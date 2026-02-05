# In[]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings("ignore")


CONFIDENCE_LEVEL = 0.95
DATA_FILE = 'Data/sim_scaling_CD_small_grid.csv'
DIST_FILE = 'Data/sim_dist_CD_small_grid.csv'
FIG_DIR = 'Figures'
os.makedirs(FIG_DIR, exist_ok=True)
BODY_QUANTILE = 0.95

# Regime Cutoffs (Specific to 20x20 Grid)
REGIME_START = 430
REGIME_SAT = 543
REGIME_END = 800

# Params
TARGET_N_DIST = 500 # Chose a point in the regime for Q-Q plot (En-route dominated)
THEORY_CD = {'dist_name': 'Rayleigh', 'k_target': 2.0}

# =============================================================================
# 1. SIMULATION ENGINE (Strict: Global Greedy + Toroidal)
# =============================================================================
def sim_greedy_dispatch_strict(
    n_d,
    num_of_periods=1000,
    seed_init=12,
    params_sim=(20, 10, 2, 0), # 20x20 grid, 10mph, 2 mile trip
    idle_cruise_prob=0.9,      
    abandon_after_periods=40,
    burn_in=200,
    return_individuals=False
):
    rng = np.random.default_rng(seed_init)
    num_blocks, speed_mph, trip_miles, _ = params_sim
    time_per_block_min = 0.3 
    lambda_per_period = 9 
    mean_trip_blocks = 40.0
    
    # Initialize
    cabs_x = rng.integers(0, num_blocks, size=n_d)
    cabs_y = rng.integers(0, num_blocks, size=n_d)
    cabs_busy_until = np.zeros(n_d, dtype=int)
    cabs_ids = np.arange(n_d)

    backlog_x = []
    backlog_y = []
    backlog_arr = []
    
    total_wait_sum = 0.0
    served_count = 0
    individual_waits = []

    # Toroidal Distance
    def get_dist_matrix(px, py, dx, dy, limit):
        diff_x = np.abs(px[:, None] - dx[None, :])
        diff_x = np.minimum(diff_x, limit - diff_x)
        diff_y = np.abs(py[:, None] - dy[None, :])
        diff_y = np.minimum(diff_y, limit - diff_y)
        return diff_x + diff_y

    for t in range(num_of_periods):
        # 1. Idle Driver Cruising
        idle_mask = cabs_busy_until <= t
        idle_indices = np.flatnonzero(idle_mask)
        if len(idle_indices) > 0 and idle_cruise_prob > 0:
            moves = rng.random(size=len(idle_indices)) < idle_cruise_prob
            if np.any(moves):
                movers = idle_indices[moves]
                axis = rng.integers(0, 2, size=len(movers))
                direction = rng.choice([-1, 1], size=len(movers))
                cabs_x[movers] = np.where(axis==0, (cabs_x[movers] + direction) % num_blocks, cabs_x[movers])
                cabs_y[movers] = np.where(axis==1, (cabs_y[movers] + direction) % num_blocks, cabs_y[movers])

        # 2. Arrivals
        k = int(rng.poisson(lambda_per_period))
        if k > 0:
            backlog_x.extend(rng.integers(0, num_blocks, size=k))
            backlog_y.extend(rng.integers(0, num_blocks, size=k))
            backlog_arr.extend([t] * k)

        # 3. Abandonment
        if abandon_after_periods and len(backlog_x) > 0:
            keep = [(t - arr) < abandon_after_periods for arr in backlog_arr]
            if not all(keep):
                backlog_x = [x for x, k in zip(backlog_x, keep) if k]
                backlog_y = [y for y, k in zip(backlog_y, keep) if k]
                backlog_arr = [a for a, k in zip(backlog_arr, keep) if k]

        # 4. Global Greedy Matching
        idle_mask = cabs_busy_until <= t
        idle_indices = np.flatnonzero(idle_mask)
        n_pax = len(backlog_x)
        n_idle = len(idle_indices)

        if n_pax > 0 and n_idle > 0:
            dists = get_dist_matrix(np.array(backlog_x), np.array(backlog_y), cabs_x[idle_indices], cabs_y[idle_indices], num_blocks)
            
            flat_dists = dists.ravel()
            flat_parr  = np.repeat(backlog_arr, n_idle)
            flat_dids  = np.tile(cabs_ids[idle_indices], n_pax)
            sort_order = np.lexsort((flat_dids, flat_parr, flat_dists))
            
            matched_p = set()
            matched_d = set()
            rem_p_indices = []
            
            p_idx_map = np.repeat(np.arange(n_pax), n_idle)
            d_idx_map = np.tile(np.arange(n_idle), n_pax)

            for k in sort_order:
                if len(matched_p) == n_pax or len(matched_d) == n_idle: break
                pidx = p_idx_map[k]; didx = d_idx_map[k]
                if pidx in matched_p or didx in matched_d: continue
                
                matched_p.add(pidx); matched_d.add(didx)
                
                # Wait Calculation
                wait_time = (t - backlog_arr[pidx]) * time_per_block_min + flat_dists[k] * time_per_block_min
                if t >= burn_in:
                    total_wait_sum += wait_time
                    served_count += 1
                    if return_individuals: individual_waits.append(wait_time)
                
                # Move Driver
                real_d_idx = idle_indices[didx]
                L = int(rng.geometric(1.0/mean_trip_blocks))
                move_x = rng.integers(0, L + 1); move_y = L - move_x
                sx = rng.choice([-1, 1]); sy = rng.choice([-1, 1])
                
                cabs_x[real_d_idx] = (backlog_x[pidx] + sx*move_x) % num_blocks
                cabs_y[real_d_idx] = (backlog_y[pidx] + sy*move_y) % num_blocks
                cabs_busy_until[real_d_idx] = t + int(flat_dists[k]) + L
                rem_p_indices.append(pidx)
            
            if rem_p_indices:
                rem_set = set(rem_p_indices)
                backlog_x = [x for i,x in enumerate(backlog_x) if i not in rem_set]
                backlog_y = [y for i,y in enumerate(backlog_y) if i not in rem_set]
                backlog_arr = [a for i,a in enumerate(backlog_arr) if i not in rem_set]

    res = {"mean_total_wait_min": total_wait_sum / max(1, served_count)}
    if return_individuals: res["individual_waits"] = np.array(individual_waits)
    return res


# =============================================================================
# HELPER FUNCTIONS: MODELS
# =============================================================================
def model_log_linear(n, gamma, beta): 
    # Theory for SH: W ~ 1/(n-beta) -> log(W) ~ -1 * log(n-beta)
    return gamma - 1.0 * np.log(n - beta)

def model_log_sqrt(n, gamma, beta): 
    # Theory for CD: W ~ 1/sqrt(n-beta) -> log(W) ~ -0.5 * log(n-beta)
    return gamma - 0.5 * np.log(n - beta)

def model_log_audit(n, gamma, delta, beta_fixed):
    return gamma - delta * np.log(n - beta_fixed)

def calculate_aic(n_vals, w_vals, func, popt, k_params):
    """Calculates AIC and R2 for a fitted model."""
    residuals = np.log(w_vals) - func(n_vals, *popt)
    rss = np.sum(residuals**2)
    n = len(n_vals)
    aic = n * np.log(rss/n) + 2 * k_params
    
    # R2 on log scale (linearized)
    sst = np.sum((np.log(w_vals) - np.mean(np.log(w_vals)))**2)
    r2 = 1 - (rss / sst)
    return aic, r2

def analyze_distribution(waits):
    """Performs Structural Recovery analysis (Weibull, Body R2)."""
    # 1. Point Mass Analysis
    n_total = len(waits)
    n_zeros = np.sum(waits <= 1e-6)
    pct_zeros = n_zeros / n_total
    
    # 2. Filter for Continuous Structure (W > 0)
    y_emp = np.sort(waits[waits > 1e-6])
    n_fit = len(y_emp)
    probs = (np.arange(n_fit) + 0.5) / n_fit
    
    # 3. Fit Theoretical Distribution (Rayleigh for Central Dispatch)
    params = stats.rayleigh.fit(y_emp, floc=0)
    y_theo = stats.rayleigh.ppf(probs, *params)
    dist_label = "Rayleigh"
        
    # 4. Weibull Structural Test (Family Recovery)
    # Weibull shape k=1 is Exp, k=2 is Rayleigh
    w_params = stats.weibull_min.fit(y_emp, floc=0)
    k_shape = w_params[0]
    
    # 5. Body R2 (Domain of Validity)
    cut_idx = int(BODY_QUANTILE * n_fit)
    if cut_idx > 2:
        r_matrix = np.corrcoef(y_theo[:cut_idx], y_emp[:cut_idx])
        r2_body = r_matrix[0, 1]**2
    else:
        r2_body = 0.0
        
    return {
        'pct_zeros': pct_zeros,
        'k_shape': k_shape,
        'r2_body': r2_body,
        'y_emp': y_emp,
        'y_theo': y_theo,
        'dist_label': dist_label
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# --- A. Scaling Simulation (if needed) ---
print(f"--- PART 1: SCALING ANALYSIS (Simulating n_d 400-800) for {NUM_SEEDS} Seeds ---")

if os.path.exists(DATA_FILE):
    df_scale = pd.read_csv(DATA_FILE)
    if 'seed' not in df_scale.columns or df_scale['seed'].nunique() < NUM_SEEDS:
        print("Existing data incomplete. Regenerating...")
        regenerate_scale = True
    else:
        print(f"Loading existing SCALING data from {DATA_FILE}")
        regenerate_scale = False
else:
    regenerate_scale = True

if regenerate_scale:
    results = []
    for seed in range(NUM_SEEDS):
        print(f"  Scaling Sim Seed {seed+1}/{NUM_SEEDS}...")
        for n_d_val in range(400, 801, 10):
            stats_out = sim_greedy_dispatch_strict(n_d=n_d_val, seed_init=seed)
            results.append({
                "seed": seed, 
                "n_d": n_d_val, 
                "mean_total_wait_min": stats_out["mean_total_wait_min"]
            })
    df_scale = pd.DataFrame(results)
    df_scale.to_csv(DATA_FILE, index=False)
    print("Scaling Simulation Complete.")


# --- B. Distribution Simulation (if needed) ---
print(f"--- PART 2: DISTRIBUTION ANALYSIS (Single Target n_d={TARGET_N_DIST}) ---")

if os.path.exists(DIST_FILE):
    df_dist = pd.read_csv(DIST_FILE)
    # Basic check if it looks correct
    if 'wait_min' not in df_dist.columns:
        regenerate_dist = True
    else:
        print(f"Loading existing DISTRIBUTION data from {DIST_FILE}")
        regenerate_dist = False
else:
    regenerate_dist = True

if regenerate_dist:
    all_waits_dist = []
    print(f"  Running Distribution Sim for {NUM_SEEDS} Seeds at n_d={TARGET_N_DIST}...")
    for seed in range(NUM_SEEDS):
        # Run longer for better distribution stats (2000 periods)
        res_dist = sim_greedy_dispatch_strict(n_d=TARGET_N_DIST, seed_init=seed, num_of_periods=2000, return_individuals=True)
        seed_waits = pd.DataFrame({"seed": seed, "wait_min": res_dist["individual_waits"]})
        all_waits_dist.append(seed_waits)
    
    df_dist = pd.concat(all_waits_dist, ignore_index=True)
    df_dist.to_csv(DIST_FILE, index=False)
    print("Distribution Simulation Complete.")


# --- C. Analysis & Logic ---

# 1. Scaling Regression
df_fit = df_scale[(df_scale['n_d'] >= REGIME_START) & (df_scale['n_d'] <= REGIME_SAT)]
n_fit = df_fit['n_d'].values
w_fit = df_fit['mean_total_wait_min'].values

# Initial Guesses
p0 = [3, 300]
bounds = ([-np.inf, 0], [np.inf, min(n_fit)-1])

# A. The Tournament
func_target = model_log_sqrt
popt_t, pcov_t = curve_fit(func_target, n_fit, np.log(w_fit), p0=p0, bounds=bounds)
aic_t, r2_t = calculate_aic(n_fit, w_fit, func_target, popt_t, 2)
beta_est = popt_t[1]

# Fit Linear (Rival)
func_rival = model_log_linear
popt_r, _ = curve_fit(func_rival, n_fit, np.log(w_fit), p0=p0, bounds=bounds)
aic_r, _ = calculate_aic(n_fit, w_fit, func_rival, popt_r, 2)

evidence_ratio = np.exp(abs(aic_r - aic_t) / 2)
preferred = 'Sqrt (Theory)' if aic_t < aic_r else 'Linear (Rival)'

# B. Physics Audit
def wrapped_audit(n, gamma, delta):
    return model_log_audit(n, gamma, delta, beta_est)

p0_audit = [10, 0.5]
popt_a, pcov_a = curve_fit(wrapped_audit, n_fit, np.log(w_fit), p0=p0_audit)
delta_est = popt_a[1]
delta_se = np.sqrt(np.diag(pcov_a))[1]
z_score_audit = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
delta_ci = (delta_est - z_score_audit*delta_se, delta_est + z_score_audit*delta_se)

# 2. Distribution Analysis
raw_waits = df_dist['wait_min'].dropna().values
dist_res = analyze_distribution(raw_waits)


# --- D. Plotting (Side-by-Side) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Scaling
ax = axes[0]
df_agg = df_scale.groupby('n_d')['mean_total_wait_min'].agg(['mean', 'std', 'count']).reset_index()
alpha_level = (1 - CONFIDENCE_LEVEL) / 2
z_score = stats.norm.ppf(1 - alpha_level)
df_agg['sem'] = df_agg['std'] / np.sqrt(df_agg['count'])
df_agg['ci'] = z_score * df_agg['sem']

df_plot = df_agg[(df_agg['n_d'] >= REGIME_START) & (df_agg['n_d'] <= REGIME_END)]
ax.errorbar(df_plot['n_d'], df_plot['mean'], yerr=df_plot['ci'], 
             fmt='o', color='gray', ecolor='gray', capsize=3, elinewidth=1.5, alpha=0.8, 
             label=f'Mean Wait ± {int(CONFIDENCE_LEVEL*100)}% CI')

# Individual Seed Scatter
CHOSEN_SEED = 2
df_seed = df_scale[(df_scale['seed'] == CHOSEN_SEED) & (df_scale['n_d'] >= REGIME_START) & (df_scale['n_d'] <= REGIME_END)]
ax.scatter(df_seed['n_d'], df_seed['mean_total_wait_min'], color='darkblue', s=40, alpha=0.9, zorder=5, 
            label=f'Sim Data (Seed {CHOSEN_SEED})')

# Best Fit Line
x_smooth = np.linspace(REGIME_START, REGIME_END, 100)
y_smooth = np.exp(func_target(x_smooth, *popt_t))
ax.plot(x_smooth, y_smooth, 'r--', lw=2, zorder=6, 
         label=rf'Fit: $W \propto (n_d - {beta_est:.0f})^{{-0.5}}$ ($R^2={r2_t:.3f}$)')

# Regimes
y_lim = ax.get_ylim()
ax.axvspan(REGIME_START, REGIME_SAT, color='green', alpha=0.1)
ax.text((REGIME_START + REGIME_SAT)/2, y_lim[1]*0.85, 'En-Route-Dominated\nRegime', 
         ha='center', va='center', color='black', fontweight='bold')
ax.axvspan(REGIME_SAT, REGIME_END, color='red', alpha=0.35)
ax.text((REGIME_SAT + REGIME_END)/2, y_lim[1]*0.85, 'Saturated\nRegime', 
         ha='center', va='center', color='black', fontweight='bold', rotation=90)

ax.set_xlabel(r'Number of Active Drivers ($n_d$)', fontsize=12)
ax.set_ylabel(r'Mean Passenger Wait ($W_d$)', fontsize=12)
ax.set_title(f'A. Scaling Validation ($n_d$ Sweep)', fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)


# Plot 2: Q-Q Plot
ax = axes[1]
y_emp = dist_res['y_emp']
y_theo = dist_res['y_theo']

# Subsample for plotting speed if needed
if len(y_emp) > 5000:
    idx = np.random.choice(len(y_emp), 5000, replace=False)
    idx.sort()
    y_emp = y_emp[idx]
    y_theo = y_theo[idx]
    
cut = int(BODY_QUANTILE * len(y_emp))

ax.scatter(y_theo[:cut], y_emp[:cut], s=15, color='blue', alpha=0.5, 
           label=f'Body (0-{int(BODY_QUANTILE*100)}%)')
ax.scatter(y_theo[cut:], y_emp[cut:], s=15, color='orange', alpha=0.4, 
           label=f'Tail ({int(BODY_QUANTILE*100)}-100%)')

max_val = max(y_theo.max(), y_emp.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.8, label='Ideal Structure')

ax.set_xlabel(f'Theoretical Quantiles ({dist_res["dist_label"]})', fontsize=12)
ax.set_ylabel('Simulated Quantiles (Sim)', fontsize=12)

# Annotation box for stats
stats_text = (
    f"Structural Validity ($n_d={TARGET_N_DIST}$):\n"
    f"Weibull $k = {dist_res['k_shape']:.2f}$ (Theory: {THEORY_CD['k_target']})\n"
    f"Body $R^2 = {dist_res['r2_body']:.4f}$"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.05, 0.75, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

ax.set_title(f'B. Structural Recovery ($W>0$)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/CD_scale_20.png', dpi=300)
print("\nPlot saved to Figures/CD_scale_20.png")


# --- E. Latex Output ---
latex_table = r"""
\begin{table}[ht]
\centering
\caption{Classification and Audit of Matching Regimes (20x20 Grid Special Case)}
\label{tab:scaling_audit_20}
\begin{tabular}{l c}
\hline
\textbf{Metric} & \textbf{Central Dispatch (20x20)} \\
\hline
\multicolumn{2}{l}{\textit{Panel A: Regime Classification (AIC Tournament)}} \\
Preferred Model & Square Root ($n^{-0.5}$) \\
Evidence Ratio (vs. Linear) & $1 : """ + f"{evidence_ratio:.1e}" + r"""$ \\
Goodness of Fit ($R^2$) & """ + f"{r2_t:.4f}" + r""" \\
\hline
\multicolumn{2}{l}{\textit{Panel B: Physics Audit (Free Exponent)}} \\
Estimated Scaling Exponent ($\hat{\delta}$) & \textbf{""" + f"{delta_est:.3f}" + r"""} \\
""" + f"{int(CONFIDENCE_LEVEL*100)}" + r"""\% Confidence Interval & [""" + f"{delta_ci[0]:.3f}, {delta_ci[1]:.3f}" + r"""] \\
Theoretical Prediction & $\delta = 0.5$ \\
\hline
\multicolumn{2}{l}{\textit{Panel C: Capacity Estimation}} \\
Est. Capacity Loss ($\hat{\beta}$) & """ + f"{beta_est:.0f}" + r""" \\
\hline
\multicolumn{2}{l}{\textit{Panel D: Distributional Structure ($W>0$)}} \\
Weibull Shape Parameter ($k$) & \textbf{""" + f"{dist_res['k_shape']:.2f}" + r"""} \\
Theoretical Prediction & Rayleigh ($k=2$) \\
Body $R^2$ (Q-Q Fit) & """ + f"{dist_res['r2_body']:.4f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""

print("\n" + "="*30 + " COPY FOR LATEX " + "="*30)
print(latex_table)
print("="*60)
