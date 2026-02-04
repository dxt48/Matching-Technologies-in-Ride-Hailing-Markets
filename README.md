# Matching-Technologies-in-Ride-Hailing-Markets

This repository contains the simulation code, data, and analysis scripts used for the empirical validation section of the paper "Project Matching". The code validates the theoretical scaling laws and distributional properties of Street-Hailing (SH) and Central Dispatch (CD) matching mechanisms.

## Repository Contents

### 1. Analysis Scripts (Reproducing Figures & Tables)
These scripts generate the figures and statistical results reported in the paper using the provided pre-computed data.

*   **`Master_Analysis.py`**: The primary analysis script. It performs the following:
    *   **Scaling Analysis**: Fits theoretical models ($n^{-1}$ for SH, $n^{-0.5}$ for CD) to simulation data.
    *   **Model Selection**: Runs an AIC "Tournament" to statistically distinguish between regimes.
    *   **Structural Recovery**: Validates the underlying wait-time distributions (Exponential vs. Rayleigh) using Q-Q plots and Weibull family recovery.
    *   **Outputs**: Generates Figure 3 (Street-Hailing), Figure 4A (Central Dispatch), and the Batched Analysis figures.

*   **`Lem_4_Prop_3.py`**: Performs the formal statistical test for **Proposition 3** (Tail Risk).
    *   **Calibration**: Calibrates fleet sizes $n_s$ and $n_d$ to achieve equal average wait times ($E[W] = 1.5$ min).
    *   **Bootstrap Test**: Uses empirical bootstrap to calculate confidence intervals for the required fleet sizes.
    *   **Visualization**: Plots the cumulative distribution functions (CDFs) to illustrate the "crossing point" $w'$ where Central Dispatch exhibits higher tail risk.

### 2. Simulation Scripts (Data Generation)
These scripts run the discrete agent-based simulations (grid-world). They are pre-configured to generate the CSV data files found in the `Data/` folder.

*   **`CD_10.py`**: Standard Central Dispatch simulation (greedy matching).
*   **`SH_10.py`**: Street-Hailing simulation (random cruising + FIFO queues).
*   **`CD_10_Batch.py`**: Batched Central Dispatch simulation (matches agents in parameterized time windows).

### 3. Data Files
The `Data/` directory contains the simulation outputs used by the analysis scripts.

*   **Scaling Data** (`sim_scaling_*.csv`): Contains mean wait times for a wide range of fleet sizes ($n$). Used for "Physics Audit" and scaling law validation.
    *   `sim_scaling_SH.csv`
    *   `sim_scaling_CD.csv`
    *   `sim_scaling_CD_batched_w2.csv` (Batch window approx 0.6 min)

*   **Distribution Data** (`sim_waits_*.csv`): Contains individual wait times for specific fleet sizes ($n_s=7000, n_d=6000$). Used for Q-Q plots and distributional analysis.
    *   `sim_waits_SH.csv`
    *   `sim_waits_CD.csv`
    *   `sim_waits_CD_batched_w2.csv`

## Requirements

The code requires a standard Python 3 environment with the following scientific computing libraries:

```bash
pip install numpy pandas matplotlib scipy
```

## Usage Instructions

### Reproducing Figures
To reproduce the figures and tables from the paper, run the analysis scripts directly. They will read the data from the `Data/` directory.

1.  **Run the Main Analysis:**
    ```bash
    python Master_Analysis.py
    ```
    *   Generates `Figures/*_Structural_Analysis.png`.
    *   Prints Table 3 (Classification, Audit, and Validation) statistics to the console.

2.  **Run the Tail Risk / Proposition 3 Test:**
    ```bash
    python Lem_4_Prop_3.py
    ```
    *   Outputs the Formal Statistical Test results (Confidence Intervals for $n^*$).
    *   Generates `Figures/wait_time_comparison.png`.

### Regenerating Simulation Data
If you wish to re-run the simulations from scratch (warning: this is computationally intensive and may take hours depending on the number of seeds):

1.  Open the desired simulation script (e.g., `CD_10.py`).
2.  Set the flags `FORCE_RERUN_SCALING = True` or `FORCE_RERUN_WAITS = True` at the top of the file.
3.  Run the script:
    ```bash
    python CD_10.py
    ```
