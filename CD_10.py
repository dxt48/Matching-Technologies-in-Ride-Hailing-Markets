# In[]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
SCALING_FILE = "Data/sim_scaling_CD.csv"
WAITS_FILE   = "Data/sim_waits_CD.csv"

# Simulation Settings
NUM_SEEDS = 10
N_D_RANGE = range(5300, 7600, 100)
TARGET_N_D_DIST = 6000

# Analysis Cutoffs (En-Route Regime)
N_START, N_SAT, N_END = 5300, 7200, 7600

# Force Rerun? (Set to True to regenerate data)
FORCE_RERUN_SCALING = False  # Set to True to regenerate scaling data
FORCE_RERUN_WAITS = False     # Set to True to regenerate distribution data 

# =============================================================================
# 1. SIMULATION ENGINE
# =============================================================================
def sim_greedy_dispatch_strict(n_d, num_of_periods=1000, seed_init=12, 
                               params_sim=(80, 10, 2, 0), idle_cruise_prob=0.9,      
                               abandon_after_periods=40, burn_in=200, return_individuals=False):
    rng = np.random.default_rng(seed_init)
    num_blocks, speed_mph, trip_miles, _ = params_sim
    time_per_block_min = 0.3 
    lambda_per_period = 117.96 
    mean_trip_blocks = 40.0
    
    cabs_x = rng.integers(0, num_blocks, size=n_d)
    cabs_y = rng.integers(0, num_blocks, size=n_d)
    cabs_busy_until = np.zeros(n_d, dtype=int)
    cabs_ids = np.arange(n_d)
    backlog_x = []; backlog_y = []; backlog_arr = []
    total_wait_sum = 0.0; served_count = 0; individual_waits = []

    def get_dist_matrix(px, py, dx, dy, limit):
        diff_x = np.abs(px[:, None] - dx[None, :]); diff_x = np.minimum(diff_x, limit - diff_x)
        diff_y = np.abs(py[:, None] - dy[None, :]); diff_y = np.minimum(diff_y, limit - diff_y)
        return diff_x + diff_y

    for t in range(num_of_periods):
        idle_mask = cabs_busy_until <= t; idle_indices = np.flatnonzero(idle_mask)
        if len(idle_indices) > 0 and idle_cruise_prob > 0:
            moves = rng.random(size=len(idle_indices)) < idle_cruise_prob
            if np.any(moves):
                movers = idle_indices[moves]; axis = rng.integers(0, 2, size=len(movers))
                direction = rng.choice([-1, 1], size=len(movers))
                cabs_x[movers] = np.where(axis==0, (cabs_x[movers] + direction) % num_blocks, cabs_x[movers])
                cabs_y[movers] = np.where(axis==1, (cabs_y[movers] + direction) % num_blocks, cabs_y[movers])
        
        k = int(rng.poisson(lambda_per_period))
        if k > 0:
            backlog_x.extend(rng.integers(0, num_blocks, size=k)); backlog_y.extend(rng.integers(0, num_blocks, size=k))
            backlog_arr.extend([t] * k)
            
        if abandon_after_periods and len(backlog_x) > 0:
            keep = [(t - arr) < abandon_after_periods for arr in backlog_arr]
            if not all(keep):
                backlog_x = [x for x, k in zip(backlog_x, keep) if k]; backlog_y = [y for y, k in zip(backlog_y, keep) if k]
                backlog_arr = [a for a, k in zip(backlog_arr, keep) if k]
                
        idle_mask = cabs_busy_until <= t; idle_indices = np.flatnonzero(idle_mask)
        n_pax = len(backlog_x); n_idle = len(idle_indices)
        
        if n_pax > 0 and n_idle > 0:
            dists = get_dist_matrix(np.array(backlog_x), np.array(backlog_y), cabs_x[idle_indices], cabs_y[idle_indices], num_blocks)
            flat_dists = dists.ravel(); flat_parr = np.repeat(backlog_arr, n_idle); flat_dids = np.tile(cabs_ids[idle_indices], n_pax)
            sort_order = np.lexsort((flat_dids, flat_parr, flat_dists))
            matched_p = set(); matched_d = set(); rem_p_indices = []
            p_idx_map = np.repeat(np.arange(n_pax), n_idle); d_idx_map = np.tile(np.arange(n_idle), n_pax)
            
            for k in sort_order:
                if len(matched_p) == n_pax or len(matched_d) == n_idle: break
                pidx = p_idx_map[k]; didx = d_idx_map[k]
                if pidx in matched_p or didx in matched_d: continue
                matched_p.add(pidx); matched_d.add(didx)
                wait_time = (t - backlog_arr[pidx]) * time_per_block_min + flat_dists[k] * time_per_block_min
                if t >= burn_in:
                    total_wait_sum += wait_time; served_count += 1
                    if return_individuals: individual_waits.append(wait_time)
                real_d_idx = idle_indices[didx]; L = int(rng.geometric(1.0/mean_trip_blocks))
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

# In[]:
# =============================================================================
# 2A. LOOP 1: Generate Scaling Data (Runs independently)
# =============================================================================
if not FORCE_RERUN_SCALING and os.path.exists(SCALING_FILE):
    print("Found existing Scaling Data. Loading...")
    df_scaling = pd.read_csv(SCALING_FILE)
    print(f"Scaling data loaded from {SCALING_FILE}")
else:
    print(f"\n{'='*60}")
    print("LOOP 1: Generating Scaling Data")
    print(f"{'='*60}")
    print(f"Running {NUM_SEEDS} seeds across n_d range {N_D_RANGE.start}-{N_D_RANGE.stop-1}...")
    
    scaling_data = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Scaling Sweep Seed {seed+1}/{NUM_SEEDS}")
        for n_d in N_D_RANGE:
            res = sim_greedy_dispatch_strict(n_d=n_d, seed_init=seed)
            scaling_data.append({
                "seed": seed, "n_d": n_d, "wait": res["mean_total_wait_min"]
            })
    
    # Save Scaling Data
    df_scaling = pd.DataFrame(scaling_data)
    df_scaling.to_csv(SCALING_FILE, index=False)
    print(f"✓ Scaling data saved to {SCALING_FILE}")

# In[]:
# =============================================================================
# 2B. LOOP 2: Generate Distribution Data (Runs independently)
# =============================================================================
if not FORCE_RERUN_WAITS and os.path.exists(WAITS_FILE):
    print("Found existing Distribution Data. Loading...")
    df_waits = pd.read_csv(WAITS_FILE)
    print(f"Distribution data loaded from {WAITS_FILE}")
else:
    print(f"\n{'='*60}")
    print("LOOP 2: Generating Distribution Data")
    print(f"{'='*60}")
    print(f"Running {NUM_SEEDS} seeds at n_d={TARGET_N_D_DIST} (2000 periods each)...")
    
    all_waits_6000 = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Distribution Deep Dive Seed {seed+1}/{NUM_SEEDS} (n_d={TARGET_N_D_DIST})")
        # We collect raw waits to do the Q-Q plot properly in the analysis phase
        res_dist = sim_greedy_dispatch_strict(n_d=TARGET_N_D_DIST, num_of_periods=2000, 
                                              seed_init=seed, return_individuals=True)
        # Store with seed ID
        seed_waits = pd.DataFrame({"seed": seed, "wait_min": res_dist["individual_waits"]})
        all_waits_6000.append(seed_waits)
    
    # Save Distribution Data
    df_waits = pd.concat(all_waits_6000, ignore_index=True)
    df_waits.to_csv(WAITS_FILE, index=False)
    print(f"✓ Distribution data saved to {WAITS_FILE}")

print(f"\n{'='*60}")
print("Data Ready for Analysis")
print(f"{'='*60}")

