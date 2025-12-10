# USE: nohup python paper_figure_error_distance.py 2 STUDY [RI|WI] argmax [0.15|none] > logs_paper_error_distance.out 2>&1 &
# If omega is specified (e.g., 0.15), generates figure for that specific omega value
# If omega is 'none' or omitted, generates figure aggregated over all omegas
# Always aggregates over all epsilons

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].lower() != 'none' else None  # Specific omega value to analyze, or None for all omegas

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"
RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"

def analyze_multidigit_module(raw_dir, figures_dir, omega_value, param_type):
    os.makedirs(figures_dir, exist_ok=True)

    # Load combined_results.csv to get error distance data
    combined_results_path = os.path.join(raw_dir, "combined_results.csv")
    if not os.path.exists(combined_results_path):
        print(f"No combined_results.csv found in {raw_dir}")
        return
    
    # Read CSV with error distance information
    combined_df = pd.read_csv(combined_results_path, low_memory=False)
    print(f"Combined results loaded from: {combined_results_path}")
    print(f"Initial number of rows in combined results: {len(combined_df)}")
    
    # Clean and convert data types
    combined_df['omega'] = pd.to_numeric(combined_df['omega'], errors='coerce')
    combined_df['error_distance'] = pd.to_numeric(combined_df['error_distance'], errors='coerce')
    
    # Ensure error column exists
    if 'error' not in combined_df.columns:
        print("No 'error' column found in combined_results.csv")
        return
    
    # Filter for param_init_type and optionally omega
    if omega_value is not None:
        # Filter for specific omega (aggregated over all epsilons)
        subset_param = combined_df[
            (combined_df['param_init_type'] == param_type) & 
            (combined_df['omega'] == omega_value) &
            (combined_df['error'] == True)
        ]
        filter_desc = f"omega={omega_value}, param={param_type}"
    else:
        # Aggregate over all omegas and all epsilons
        subset_param = combined_df[
            (combined_df['param_init_type'] == param_type) &
            (combined_df['error'] == True)
        ]
        filter_desc = f"all omegas, param={param_type}"
    
    if subset_param.empty:
        print(f"No error data found for {filter_desc}")
        return
    
    print(f"Found {len(subset_param)} error rows for {filter_desc}")
    
    # --- Figure: Errors by distance (normalized/proportion) ---
    # Calculate normalized error distance distribution
    overall_error = subset_param['error_distance'].value_counts(normalize=True).sort_index()
    # Convert the normalized values to percentages for better readability
    overall_error = overall_error * 100  # Convert to percentage
    
    if overall_error.empty:
        print(f"No error distance data available")
        return
    
    # Create the figure with improved styling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to show only distances up to a reasonable maximum (e.g., 20 or max if smaller)
    max_distance = min(25, int(overall_error.index.max()))
    overall_error_filtered = overall_error[overall_error.index <= max_distance]
    
    # Create bar plot with a nice color scheme
    bars = ax.bar(overall_error_filtered.index, overall_error_filtered.values, 
                  color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # Set labels and title with appropriate font sizes
    ax.set_xlabel('Error Distance', fontsize=32)
    ax.set_ylabel('Percentage of Errors (%)', fontsize=32)
    ax.tick_params(axis='both', labelsize=28)
    
    # Set x-axis to show only selected distances (not all)
    # Show distances: 1, 5, 10, 15, 20, etc.
    x_ticks = [1]
    if max_distance >= 5:
        x_ticks.extend(range(5, max_distance + 1, 5))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    
    # Set y-axis limits
    ax.set_ylim(0, max(overall_error_filtered.values) * 1.1)
    
    # Add grid for better readability
    plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_axisbelow(True)
    
    # Save the figure
    if omega_value is not None:
        safe_om = str(omega_value).replace('.', '_')
        fname = os.path.join(figures_dir, f"errors_by_distance_omega_{safe_om}.png")
    else:
        fname = os.path.join(figures_dir, f"errors_by_distance_all_omegas.png")
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Error distance figure saved to: {fname}")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)