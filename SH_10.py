# In[]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings
from collections import deque
import logging
import os

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================
SCALING_FILE = "Data/sim_scaling_SH.csv"
WAITS_FILE   = "Data/sim_waits_SH.csv"

# Simulation Settings
NUM_SEEDS = 10
# Range chosen to cover Matching-Dominated Regime (Prop 1)
# Note: Street Hailing requires more drivers than CD.
N_S_RANGE = range(6400, 8800, 100) 
TARGET_N_S_DIST = 7000  # Middle of stable regime

# Analysis Cutoffs (Matching-Dominated Regime)
N_START, N_SAT, N_END = 6600, 8500, 8800

# Force Rerun? (Set to True to regenerate data)
FORCE_RERUN_SCALING = False  # Set to True to regenerate scaling data
FORCE_RERUN_WAITS = False     # Set to True to regenerate distribution data 

# =============================================================================
# 1. SIMULATION ENGINE (Street Hailing)
# =============================================================================
CITY_SIDE_MILES = 4.0   
BASE_ARRIVALS_PER_MIN = 393.2

def sim_street_hailing_with_queue(
    n_s,
    num_of_periods=1000,
    seed_init=12,
    params_sim=(80, 10, 2, 0),        
    demand_rate_scale=1,
    mean_trip_blocks=40.0,            
    idle_cruise_prob=0.5,             
    abandon_after_periods=40,       
    return_individual_waits=False,    
    burn_in=200
):
    # --- RNG & Setup ---
    rng = np.random.default_rng(int(seed_init))
    num_blocks, speed_mph, trip_len_miles, _ = params_sim
    
    city_side_miles   = CITY_SIDE_MILES
    block_len_miles   = city_side_miles / num_blocks
    time_per_blockmin = 60.0 * (block_len_miles / speed_mph) # ~0.3 min

    mu_blocks = float(mean_trip_blocks)
    mu_blocks = max(mu_blocks, 1e-6)

    lam = float(demand_rate_scale) * time_per_blockmin * BASE_ARRIVALS_PER_MIN

    # Helpers
    def wrap_step(pos, step, limit):
        return (int(pos) + int(step)) % int(limit)

    def sample_trip_blocks():
        p = min(1.0, max(1e-6, 1.0 / mu_blocks))
        return int(rng.geometric(p))

    def cell_index(x, y):
        return int(x) * num_blocks + int(y)

    # State
    xC = rng.integers(0, num_blocks, size=int(n_s), dtype=np.int64)
    yC = rng.integers(0, num_blocks, size=int(n_s), dtype=np.int64)
    busy_until = np.zeros(int(n_s), dtype=np.int64)
    corner_queues = [deque() for _ in range(num_blocks * num_blocks)]

    served = 0
    sum_queue_min  = 0.0
    individual_waits = [] 

    # --- Main Loop ---
    for t in range(int(num_of_periods)):
        # 1. Arrivals
        k = int(rng.poisson(lam))
        if k > 0:
            xP = rng.integers(0, num_blocks, size=k)
            yP = rng.integers(0, num_blocks, size=k)
            for xi, yi in zip(xP, yP):
                corner_queues[cell_index(xi, yi)].append(int(t))

        # 2. Abandonment
        if abandon_after_periods is not None:
            thresh = int(t) - int(abandon_after_periods)
            if thresh >= 0:
                for q in corner_queues:
                    while q and int(q[0]) <= thresh:
                        q.popleft()

        # 3. Immediate Pickups & Relocation
        idle_mask = (busy_until <= t)
        idle_idx  = np.flatnonzero(idle_mask)
        
        # Shuffle for fairness (optional but good practice)
        if len(idle_idx) > 0:
             rng.shuffle(idle_idx)

        # Helper to process pickup
        def try_pickup(cab, use_move=False):
            nonlocal served, sum_queue_min
            # Pick up from current location
            q = corner_queues[cell_index(xC[cab], yC[cab])]
            if q:
                a = q.popleft()
                L = sample_trip_blocks()
                busy_until[cab] = t + L
                
                # Relocation (Wrap)
                move_x = rng.integers(0, L + 1); move_y = L - move_x
                sx = rng.choice([-1, 1]); sy = rng.choice([-1, 1])
                xC[cab] = wrap_step(wrap_step(xC[cab], sx*move_x, num_blocks), 0, num_blocks)
                yC[cab] = wrap_step(wrap_step(yC[cab], sy*move_y, num_blocks), 0, num_blocks)

                if t >= burn_in:
                    wait = float((t - a) * time_per_blockmin)
                    if return_individual_waits: individual_waits.append(wait)
                    sum_queue_min += wait
                    served += 1
                return True
            return False

        # Phase 3: Immediate Pickups (Pre-Cruise)
        if idle_idx.size:
            for cab in idle_idx:
                try_pickup(cab)

        # 4. Cruising
        # Re-check idle (some got busy in step 3)
        idle_mask = (busy_until <= t)
        idle_idx  = np.flatnonzero(idle_mask)
        
        if idle_idx.size and idle_cruise_prob > 0.0:
            will_move = rng.random(idle_idx.size) < idle_cruise_prob
            axes  = rng.integers(0, 2, size=idle_idx.size)
            steps = rng.choice([-1, 1], size=idle_idx.size)
            for k, cab in enumerate(idle_idx):
                if will_move[k]:
                    if axes[k] == 0: xC[cab] = wrap_step(xC[cab], steps[k], num_blocks)
                    else:            yC[cab] = wrap_step(yC[cab], steps[k], num_blocks)

        # 5. Post-Move Pickups
        idle_mask = (busy_until <= t)
        idle_idx  = np.flatnonzero(idle_mask)
        if idle_idx.size:
             # Shuffle again
            rng.shuffle(idle_idx)
            for cab in idle_idx:
                try_pickup(cab)

    mean_w = sum_queue_min / max(1, served)
    res = {"mean_total_wait_min": mean_w}
    if return_individual_waits: res["individual_waits"] = np.array(individual_waits)
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
    print(f"Running {NUM_SEEDS} seeds across n_s range {N_S_RANGE.start}-{N_S_RANGE.stop-1}...")
    
    scaling_data = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Scaling Sweep Seed {seed+1}/{NUM_SEEDS}")
        for n_s in N_S_RANGE:
            res = sim_street_hailing_with_queue(n_s=n_s, seed_init=seed)
            scaling_data.append({
                "seed": seed, "n_s": n_s, "wait": res["mean_total_wait_min"]
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
    print(f"Running {NUM_SEEDS} seeds at n_s={TARGET_N_S_DIST} (2000 periods each)...")
    
    all_waits_dist = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Distribution Deep Dive Seed {seed+1}/{NUM_SEEDS} (n_s={TARGET_N_S_DIST})")
        # We collect raw waits to do the Q-Q plot properly in the analysis phase
        res_dist = sim_street_hailing_with_queue(n_s=TARGET_N_S_DIST, num_of_periods=2000, 
                                              seed_init=seed, return_individual_waits=True)
        # Store with seed ID
        seed_waits = pd.DataFrame({"seed": seed, "wait_min": res_dist["individual_waits"]})
        all_waits_dist.append(seed_waits)
    
    # Save Distribution Data
    df_waits = pd.concat(all_waits_dist, ignore_index=True)
    df_waits.to_csv(WAITS_FILE, index=False)
    print(f"✓ Distribution data saved to {WAITS_FILE}")

print(f"\n{'='*60}")
print("Data Ready for Analysis")
print(f"{'='*60}")

