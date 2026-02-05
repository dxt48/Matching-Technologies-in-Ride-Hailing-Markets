import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings
import os
from collections import deque

warnings.filterwarnings("ignore")

# Configuration
NUM_SEEDS = 10



warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "Data"
FIG_DIR = "Figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Analysis Parameters
BODY_QUANTILE = 0.95
CONFIDENCE_LEVEL = 0.95
NUM_SEEDS = 10 
CHOSEN_SEED = 2 # User can change this (0-9 typically) 

# Regime Cutoffs (derived from visual inspection of saturation points)
SH_REGIME = {'start': 6600, 'sat': 8500, 'end': 8800}
CD_REGIME = {'start': 5300, 'sat': 7200, 'end': 7600}
CD_BATCHED_REGIME = {'start': 5300, 'sat': 7200, 'end': 7600} # Maintaining same regime for comparison

# Simulation Parameters (for Data Generation)
#SH_SIM_PARAMS = {'n_range': range(6400, 8800, 100), 'target_n': 7200}
#CD_SIM_PARAMS = {'n_range': range(5300, 7600, 100), 'target_n': 6000}

SH_SIM_PARAMS = {'n_range': range(6400, 8800, 100), 'target_n': 6783}
CD_SIM_PARAMS = {'n_range': range(5300, 7600, 100), 'target_n': 6783}


# Theoretical Expectations (for Table Comparison)
THEORY = {
    'SH': {'beta': 4836, 'delta': 1.0, 'dist_name': 'Exponential', 'k_target': 1.0},
    'CD': {'beta': 4719, 'delta': 0.5, 'dist_name': 'Rayleigh', 'k_target': 2.0},
    'CD_BATCHED': {'beta': 4719, 'delta': 0.5, 'dist_name': 'Rayleigh', 'k_target': 2.0}
}

# =============================================================================
# HELPER FUNCTIONS: MODELS
# =============================================================================

# 1. The Models (Log-Linearized to handle heteroscedasticity)
def model_log_linear(n, gamma, beta): 
    # Theory for SH: W ~ 1/(n-beta) -> log(W) ~ -1 * log(n-beta)
    return gamma - 1.0 * np.log(n - beta)

def model_log_sqrt(n, gamma, beta): 
    # Theory for CD: W ~ 1/sqrt(n-beta) -> log(W) ~ -0.5 * log(n-beta)
    return gamma - 0.5 * np.log(n - beta)

# 2. The Physics Audit Model (Free Exponent, Fixed Beta)
# We fix Beta to the best-fit value from the constrained model to isolate the scaling exponent.
def model_log_audit(n, gamma, delta, beta_fixed):
    return gamma - delta * np.log(n - beta_fixed)

# =============================================================================
# HELPER FUNCTIONS: STATISTICS
# =============================================================================

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

def analyze_distribution(waits, mode):
    """Performs Structural Recovery analysis (Weibull, Body R2)."""
    # 1. Point Mass Analysis
    n_total = len(waits)
    n_zeros = np.sum(waits <= 1e-6)
    pct_zeros = n_zeros / n_total
    
    # 2. Filter for Continuous Structure (W > 0)
    y_emp = np.sort(waits[waits > 1e-6])
    n_fit = len(y_emp)
    probs = (np.arange(n_fit) + 0.5) / n_fit
    
    # 3. Fit Theoretical Distribution
    if mode == 'SH':
        # Exponential
        params = stats.expon.fit(y_emp, floc=0)
        y_theo = stats.expon.ppf(probs, *params)
        dist_label = "Exponential"
    else:
        # Rayleigh
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
# MAIN ANALYSIS ENGINE
# =============================================================================

latex_results = {}

for mode in ['SH', 'CD', 'CD_BATCHED']:
    print(f"--- Running Structural Analysis for {mode} ---")
    
    # Load Data (assuming CSV files are available)
    if mode == 'CD_BATCHED':
        scale_file = os.path.join(DATA_DIR, "sim_scaling_CD_batched_w2.csv")
        waits_file = os.path.join(DATA_DIR, "sim_waits_CD_batched_w2.csv")
        n_key = "n_d"
    else:
        scale_file = os.path.join(DATA_DIR, f"sim_scaling_{mode}.csv")
        waits_file = os.path.join(DATA_DIR, f"sim_waits_{mode}.csv")
        n_key = "n_s" if mode == 'SH' else "n_d"
    
    # Load data directly from CSV files
    df_scale = pd.read_csv(scale_file)
    df_dist = pd.read_csv(waits_file)

    # -------------------------------------------------------------------------
    # PART 1: SCALING ANALYSIS (The Tournament & Audit)
    # -------------------------------------------------------------------------
    
    # Settings
    # Settings
    if mode == 'SH':
        regime = SH_REGIME
        n_col = 'n_s'
    elif mode == 'CD':
        regime = CD_REGIME
        n_col = 'n_d'
    else: # CD_BATCHED
        regime = CD_BATCHED_REGIME
        n_col = 'n_d'
    
    # Filter Data (Matching-Dominated / En-Route Regime)
    df_fit = df_scale[(df_scale[n_col] >= regime['start']) & (df_scale[n_col] <= regime['sat'])]
    n_fit = df_fit[n_col].values
    w_fit = df_fit['wait'].values if 'wait' in df_fit.columns else df_fit['mean_total_wait_min'].values
    
    # Define Candidates
    # Define Candidates
    if mode == 'SH':
        func_target = model_log_linear # Theory: 1/n
        func_rival  = model_log_sqrt   # Rival: 1/sqrt(n)
        p0 = [10, 5000]
    else: # CD or CD_BATCHED
        func_target = model_log_sqrt   # Theory: 1/sqrt(n)
        func_rival  = model_log_linear # Rival: 1/n
        p0 = [5, 4000]
        
    bounds = ([-np.inf, 0], [np.inf, min(n_fit)-1])

    # A. The Tournament (Constrained fit)
    popt_t, pcov_t = curve_fit(func_target, n_fit, np.log(w_fit), p0=p0, bounds=bounds)
    aic_t, r2_t = calculate_aic(n_fit, w_fit, func_target, popt_t, 2)
    beta_est = popt_t[1] # Extract Beta for the Audit

    popt_r, _ = curve_fit(func_rival, n_fit, np.log(w_fit), p0=p0, bounds=bounds)
    aic_r, _ = calculate_aic(n_fit, w_fit, func_rival, popt_r, 2)
    
    evidence_ratio = np.exp(abs(aic_r - aic_t) / 2)
    
    # B. The Physics Audit (Free Exponent, Fixed Beta)
    def wrapped_audit(n, gamma, delta):
        return model_log_audit(n, gamma, delta, beta_est)
    
    # Guess delta=1 for SH, delta=0.5 for CD/CD_BATCHED
    p0_audit = [10, 1.0 if mode=='SH' else 0.5] 
    popt_a, pcov_a = curve_fit(wrapped_audit, n_fit, np.log(w_fit), p0=p0_audit)
    
    delta_est = popt_a[1]
    delta_se = np.sqrt(np.diag(pcov_a))[1]
    z_score_audit = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
    delta_ci = (delta_est - z_score_audit*delta_se, delta_est + z_score_audit*delta_se)

    # -------------------------------------------------------------------------
    # PART 2: DISTRIBUTION ANALYSIS (Structural Recovery)
    # -------------------------------------------------------------------------
    
    # Get raw waits (aggregated or single seed - utilizing all for robustness)
    raw_waits = df_dist['wait_min'].dropna().values
    dist_res = analyze_distribution(raw_waits, mode)
    
    # -------------------------------------------------------------------------
    # PART 3: GENERATE PLOTS (Side-by-Side)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot A: Scaling (Left) ---
    ax = axes[0]
    
    # 1. Aggregate Simulation Data (Mean +/- 95% CI)
    df_agg = df_scale.groupby(n_col)[w_fit.dtype.names if w_fit.dtype.names else 'wait' if 'wait' in df_scale.columns else 'mean_total_wait_min'].agg(['mean', 'std', 'count']).reset_index()
    # Handle column name difference
    y_col = 'mean'
    alpha_level = (1 - CONFIDENCE_LEVEL) / 2
    z_score = stats.norm.ppf(1 - alpha_level) 
    
    df_agg['sem'] = df_agg['std'] / np.sqrt(df_agg['count'])
    # Confidence Interval: z_score * SEM
    df_agg['ci'] = z_score * df_agg['sem']
    
    # Filter for plot range
    df_plot = df_agg[(df_agg[n_col] >= regime['start']) & (df_agg[n_col] <= regime['end'])]
    
    ax.errorbar(df_plot[n_col], df_plot[y_col], yerr=df_plot['ci'], 
                fmt='o', color='gray', ecolor='gray', capsize=3, elinewidth=1.5, alpha=0.8, label=f'Mean ± {int(CONFIDENCE_LEVEL*100)}% CI (All Seeds)')

    # 1b. Plot Specific Seed Scatter
    if 'seed' in df_scale:
        df_seed = df_scale[(df_scale['seed'] == CHOSEN_SEED) & (df_scale[n_col] >= regime['start']) & (df_scale[n_col] <= regime['end'])]
        # Ensure we use the correct column name for wait
        w_col_seed = 'wait' if 'wait' in df_scale.columns else 'mean_total_wait_min'
        ax.scatter(df_seed[n_col], df_seed[w_col_seed], color='darkblue', s=40, alpha=0.9, zorder=5, label=f'Simulated Data (Seed {CHOSEN_SEED})')
    
    # 2. Plot Fit Line
    x_smooth = np.linspace(regime['start'], regime['end'], 100)
    y_smooth = np.exp(func_target(x_smooth, *popt_t))
    
    fit_eqn = r'$\bar{W} \propto (n - \beta)^{-1}$' if mode == 'SH' else r'$\bar{W} \propto (n - \beta)^{-0.5}$'
    ax.plot(x_smooth, y_smooth, 'r--', lw=2, label=f'Best Fit: {fit_eqn}\n($R^2={r2_t:.3f}$, AIC={aic_t:.1f})')

    # 3. Add Fit Confidence Bands (Delta Method / Monte Carlo approx)
    try:
        # Sample parameters from multivariate normal (approx posterior)
        mc_params = np.random.multivariate_normal(popt_t, pcov_t, 1000)
        # Evaluate model for all samples
        mc_curves = np.array([np.exp(func_target(x_smooth, *p)) for p in mc_params])
        # Calculate CI based on CONFIDENCE_LEVEL
        pct_low = (1 - CONFIDENCE_LEVEL) / 2 * 100
        pct_high = (1 + CONFIDENCE_LEVEL) / 2 * 100
        lower_bound = np.percentile(mc_curves, pct_low, axis=0)
        upper_bound = np.percentile(mc_curves, pct_high, axis=0)
        
        ax.fill_between(x_smooth, lower_bound, upper_bound, color='red', alpha=0.2, label=f'Fit {int(CONFIDENCE_LEVEL*100)}% CI')
    except Exception as e:
        print(f"Warning: Could not plot fit intervals: {e}")
    
    # 3. Regime Shading & Labels
    trans = ax.get_xaxis_transform() # x in data coordinates, y in axes coordinates (0-1)
    
    # Stable/Dominated Regime
    #ax.axvspan(regime['start'], regime['sat'], color='green' if 'CD' in mode else #'lightblue', alpha=0.15)
    #regime_name = "En-Route-Dominated" if 'CD' in mode else "Matching-Dominated"
    #ax.text((regime['start'] + regime['sat'])/2, 0.6, regime_name, transform=trans,
    #        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # Saturated Regime
    #ax.axvspan(regime['sat'], regime['end'], color='red', alpha=0.15)
    #ax.text((regime['sat'] + regime['end'])/2, 0.5, "Saturated", transform=trans,
    #        ha='center', va='center', fontsize=10, fontweight='bold', color='black', rotation=90)
    
    ax.set_xlabel(r'Number of Active Drivers', fontsize=12)
    ax.set_ylabel(r'Mean Passenger Wait Time (Minutes)', fontsize=12)
    if mode == "SH":
        full_name = "Street-Hailing"
    elif mode == "CD":
        full_name = "Central Dispatch"
    else:
        full_name = "Central Dispatch (Batched)"
        
    ax.set_title(f'A. {full_name} Scaling Validation', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # --- Plot B: Q-Q Plot (Right) ---
    ax = axes[1]
    
    # Subsample for plotting speed if needed
    y_emp = dist_res['y_emp']
    y_theo = dist_res['y_theo']
    if len(y_emp) > 5000:
        idx = np.random.choice(len(y_emp), 5000, replace=False)
        idx.sort()
        y_emp = y_emp[idx]
        y_theo = y_theo[idx]
        
    cut = int(BODY_QUANTILE * len(y_emp))
    
    # Plot Body
    ax.scatter(y_theo[:cut], y_emp[:cut], s=15, color='blue', alpha=0.5, 
               label=f'Body (0-{int(BODY_QUANTILE*100)}%)')
    # Plot Tail
    ax.scatter(y_theo[cut:], y_emp[cut:], s=15, color='orange', alpha=0.4, 
               label=f'Tail ({int(BODY_QUANTILE*100)}-100%)')
    # Identity Line
    max_val = max(y_theo.max(), y_emp.max())
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.8, label='Ideal Structure')
    
    ax.set_xlabel(f'Theoretical Quantiles ({dist_res["dist_label"]})', fontsize=12)
    ax.set_ylabel('Simulated Quantiles (Sim)', fontsize=12)
    
    # Annotation box for stats
    stats_text = (
        f"Structural Validity:\n"
        f"Weibull $k = {dist_res['k_shape']:.2f}$ (Theory: {THEORY[mode]['k_target']})\n"
        f"Body $R^2 = {dist_res['r2_body']:.4f}$"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.75, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    ax.set_title(f'B. {full_name} Structural Recovery ($W>0$)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{mode}_Structural_Analysis.png"), dpi=300)
    # plt.show() # Comment out if running in batch
    print(f"Figures saved to {FIG_DIR}/{mode}_Structural_Analysis.png")
    
    # -------------------------------------------------------------------------
    # STORE RESULTS FOR LATEX
    # -------------------------------------------------------------------------
    latex_results[mode] = {
        'aic_pref': aic_t,
        'evidence': evidence_ratio,
        'r2': r2_t,
        'delta_est': delta_est,
        'delta_low': delta_ci[0],
        'delta_high': delta_ci[1],
        'beta_est': beta_est,
        'pct_zeros': dist_res['pct_zeros'],
        'k_shape': dist_res['k_shape'],
        'r2_body': dist_res['r2_body']
    }

# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

sh = latex_results['SH']
cd = latex_results['CD']
cd_b = latex_results['CD_BATCHED']

combined_tex = r"""
\begin{table}[ht]
\centering
\caption{Classification, Audit, and Structural Validation of Matching Regimes}
\label{tab:scaling_audit_combined}
\resizebox{\textwidth}{!}{
\begin{tabular}{l c c c}
\hline
\textbf{Metric} & \textbf{Street-Hailing} & \textbf{Central Dispatch} & \textbf{Central Dispatch (0.6 min batch window)} \\
\hline
\multicolumn{4}{l}{\textit{Panel A: Regime Classification (AIC Tournament)}} \\
Preferred Model & Linear ($n^{-1}$) & Square Root ($n^{-0.5}$) & Square Root ($n^{-0.5}$) \\
Evidence Ratio (vs. Alternative) & $1 : """ + f"{sh['evidence']:.1e}" + r"""$ & $1 : """ + f"{cd['evidence']:.1e}" + r"""$ & $1 : """ + f"{cd_b['evidence']:.1e}" + r"""$ \\
Goodness of Fit ($R^2$) & """ + f"{sh['r2']:.4f}" + r""" & """ + f"{cd['r2']:.4f}" + r""" & """ + f"{cd_b['r2']:.4f}" + r""" \\
\hline
\multicolumn{4}{l}{\textit{Panel B: Physics Audit (Free Exponent)}} \\
Estimated Scaling Exponent ($\hat{\delta}$) & \textbf{""" + f"{sh['delta_est']:.3f}" + r"""} & \textbf{""" + f"{cd['delta_est']:.3f}" + r"""} & \textbf{""" + f"{cd_b['delta_est']:.3f}" + r"""} \\
95\% Confidence Interval & [""" + f"{sh['delta_low']:.3f}, {sh['delta_high']:.3f}" + r"""] & [""" + f"{cd['delta_low']:.3f}, {cd['delta_high']:.3f}" + r"""] & [""" + f"{cd_b['delta_low']:.3f}, {cd_b['delta_high']:.3f}" + r"""] \\
Theoretical Prediction & $\delta = 1.0$ & $\delta = 0.5$ & $\delta = 0.5$ \\
\hline
\multicolumn{4}{l}{\textit{Panel C: Capacity Estimation}} \\
Est. Capacity Loss ($\hat{\beta}$) & """ + f"{sh['beta_est']:.0f}" + r""" & """ + f"{cd['beta_est']:.0f}" + r""" & """ + f"{cd_b['beta_est']:.0f}" + r""" \\
Theoretical Prediction & $\approx 4836$ & $\approx 4719$ & $\approx 4719$ \\
\hline
\multicolumn{4}{l}{\textit{Panel D: Structural Recovery (Distribution)}} \\
\% Immediate Pickups ($W=0$) & """ + f"{sh['pct_zeros']*100:.1f}" + r"""\% & """ + f"{cd['pct_zeros']*100:.1f}" + r"""\% & """ + f"{cd_b['pct_zeros']*100:.1f}" + r"""\% \\
Weibull Shape Parameter ($k$) & \textbf{""" + f"{sh['k_shape']:.2f}" + r"""} & \textbf{""" + f"{cd['k_shape']:.2f}" + r"""} & \textbf{""" + f"{cd_b['k_shape']:.2f}" + r"""} \\
Target Family & Exponential ($k=1$) & Rayleigh ($k=2$) & Rayleigh ($k=2$) \\
Body $R^2$ (Q-Q Linearity) & """ + f"{sh['r2_body']:.4f}" + r""" & """ + f"{cd['r2_body']:.4f}" + r""" & """ + f"{cd_b['r2_body']:.4f}" + r""" \\
\hline
\end{tabular}
}
\\ \footnotesize{\textit{Note:} In Panel B, $\beta$ is constrained to the best-fit value to strictly isolate the scaling exponent $\delta$. In Panel D, structural metrics ($k$, $R^2$) are calculated on positive wait times to validate the travel-time component independent of the zero-wait point mass. Distribution fit testing uses $n_s=7000$ for Street-Hailing and $n_d=6000$ for Central Dispatch regimes.}
\end{table}
"""

figure_tex = r"""
\begin{figure}[ht]
    \caption{Street-Hailing: Structural Validity}
    \label{fig:sh_structure}
    \centering
    \includegraphics[width=1.0\textwidth]{Figures/SH_Structural_Analysis.png}
    
    \vspace{0.5em}
    \footnotesize{\textit{Note:} (A) The Physics Audit confirms the $1/n$ scaling law within the matching-dominated regime. (B) The Q-Q plot confirms the Exponential memoryless property for the vast majority of passengers (Body), with expected deviations only in the extreme tail due to finite grid effects. Distribution fit testing uses $n_s=7000$.}
\end{figure}

\begin{figure}[ht]
    \caption{Central Dispatch: Structural Validity}
    \label{fig:cd_structure}
    \centering
    \includegraphics[width=1.0\textwidth]{Figures/CD_Structural_Analysis.png}
    
    \vspace{0.5em}
    \footnotesize{\textit{Note:}  (A) The Physics Audit confirms the $1/\sqrt{n}$ scaling law predicted by spatial coverage theory. (B) The Q-Q plot confirms the Rayleigh-like structure of wait times when $W>0$, distinguishing the spatial travel component from immediate pickup events. Distribution fit testing uses $n_d=6000$.}
\end{figure}

\begin{figure}[ht]
    \caption{Central Dispatch with Batching: Structural Validity}
    \label{fig:cd_batched_structure}
    \centering
    \includegraphics[width=1.0\textwidth]{Figures/CD_BATCHED_Structural_Analysis.png}
    
    \vspace{0.5em}
    \footnotesize{\textit{Note:}  (A) The Physics Audit confirms the $1/\sqrt{n}$ scaling law persists under batch matching, albeit with improved efficiency (lower effective $\beta$). (B) The Q-Q plot confirms the Rayleigh-like structure of wait times remains the dominant spatial physics. Distribution fit testing uses $n_d=6000$. The batch window is 0.6 minutes.}
\end{figure}
"""

print("\n" + "="*30 + " COPY FOR MAIN BODY (COMBINED TABLE) " + "="*30)
print(combined_tex)
print("\n" + "="*30 + " COPY FOR FIGURES (.TEX) " + "="*30)
print(figure_tex)
print("="*60)



