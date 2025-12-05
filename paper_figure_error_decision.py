# USE: nohup python paper_figure_error_decision.py 2 FOURTH_STUDY WI argmax > logs_paper_error_decision.out 2>&1 &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
XLIMS = [1000, 2000, 4000]  # X-axis limit for the plots

# Hardcoded omega values to plot
OMEGA_VALUES = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

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

def analyze_multidigit_module(raw_dir, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        return
    
    # Read CSV with low_memory=False to avoid dtype warning
    combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
    print(f"Combined logs loaded from: {combined_logs_path}")
    print(f"Initial number of rows in combined logs: {len(combined_logs)}")
    
    # Clean and convert data types
    combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
    combined_logs['accuracy'] = pd.to_numeric(combined_logs['accuracy'], errors='coerce')
    combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')
    
    # Remove rows with NaN values in critical columns
    combined_logs = combined_logs.dropna(subset=['epoch', 'accuracy', 'omega'])
    
    print(f"After cleaning: {len(combined_logs)} rows remaining")
        
    # Create single figure with lines for each hardcoded omega
    plt.figure(figsize=(16, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(OMEGA_VALUES)))
    colors = colors[::-1]  # Reverse colors for better visibility

    for i, omega in enumerate(OMEGA_VALUES):
        # Filter data for this omega value
        omega_data = combined_logs[combined_logs['omega'] == omega]
        
        if omega_data.empty:
            print(f"No data found for omega = {omega}")
            continue
        
        # Group by epoch and calculate mean and std across runs
        epoch_stats = omega_data.groupby('epoch')['accuracy'].agg(['mean', 'std']).reset_index()
        epoch_stats['error'] = 100 - epoch_stats['mean'] * 100 # Convert to error percentage

        # Calculate smoothed error using rolling window
        window_size = min(10, len(epoch_stats) // 100)  # Adaptive window size
        if window_size < 5:
            window_size = 5
        epoch_stats['smoothed_error'] = epoch_stats['error'].rolling(window=window_size, center=True, min_periods=1).mean()
        epoch_stats['smoothed_std'] = epoch_stats['std'].rolling(window=window_size, center=True, min_periods=1).mean()

        epoch_stats['std'] = 20 * epoch_stats['std'] + (20 + 2 * epoch_stats['smoothed_error']) * epoch_stats['smoothed_std']  # Convert std to error percentage scale

        # Plot line with error bars using smoothed error
        plt.plot(epoch_stats['epoch'], epoch_stats['error'], 
                label=f'ω = {omega}', color=colors[i], linewidth=3)
        plt.fill_between(epoch_stats['epoch'], 
                       epoch_stats['smoothed_error'] - epoch_stats['std'],
                       epoch_stats['smoothed_error'] + epoch_stats['std'],
                       color=colors[i], alpha=0.2)
    
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel('Averaged Error (%)', fontsize=30)
    plt.legend(loc='center right', 
               bbox_to_anchor=(1.32, 0.5), 
               title='Magnitude Noise', 
               fontsize=28, 
               title_fontsize=30
               )
    plt.grid(True, color='gray', alpha=0.7)
    plt.ylim(-1, 101)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    
    # Save the figure
    for XLIM in XLIMS:
        plt.xlim(0, XLIM)
        output_filename = f"error_by_omega_{STUDY_NAME}_{PARAM_TYPE}_{MODEL_TYPE}_{XLIM}.png"
        output_path = os.path.join(figures_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    plt.close()
    print(f"Figure saved to: {output_path}")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR)