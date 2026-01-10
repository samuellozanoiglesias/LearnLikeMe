# USE: nohup python paper_figure_effects.py 2 STUDY WI argmax 0.15 500 > logs_paper_effects.out 2>&1 &

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
OMEGA_VALUE = float(sys.argv[5])  # Specific omega value to analyze
EPOCH = int(sys.argv[6]) if len(sys.argv) > 6 else "last"  # Specific epoch for barplot analysis

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
    combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')
    
    # Convert accuracy columns to numeric
    acc_cols = [
        "test_pairs_no_carry_small_accuracy",
        "test_pairs_carry_small_accuracy",
        "test_pairs_no_carry_large_accuracy",
        "test_pairs_carry_large_accuracy"
    ]
    for col in acc_cols:
        if col in combined_logs.columns:
            combined_logs[col] = pd.to_numeric(combined_logs[col], errors='coerce')
    
    print(f"After cleaning: {len(combined_logs)} rows remaining")
    
    # Filter for the specific omega and param_init_type
    subset_param = combined_logs[
        (combined_logs['param_init_type'] == param_type) & 
        (combined_logs['omega'] == omega_value)
    ]
    
    if subset_param.empty:
        print(f"No data found for param_type={param_type} and omega={omega_value}")
        return
    
    print(f"Found {len(subset_param)} rows for omega={omega_value}, param={param_type}")
    
    # --- Figure 1: Errors over epochs (aggregated across all epsilons) ---
    acc_labels = ["Small - No Carry", "Small - Carry", "Large - No Carry", "Large - Carry"]
    colors = ["#5FA8FF", "#1C6CE5", "#F46A6A", "#D62828"]  # Sky blue, light pink, steel blue, indian red
    
    pw = {}
    for col in acc_cols:
        if col in subset_param.columns:
            pw[col] = subset_param.pivot_table(index='epoch', columns='run', values=col)
    
    if pw:
        plt.figure(figsize=(12, 7))
        for col, lbl, colcol in zip(acc_cols, acc_labels, colors):
            if col not in pw:
                continue
            df_runs = pw[col]
            if df_runs is None or df_runs.empty:
                continue
            mean = df_runs.mean(axis=1)
            std = df_runs.std(axis=1).fillna(0)
            # Convert to numpy float arrays for plotting
            mean = pd.to_numeric(mean, errors='coerce').astype(float).to_numpy()
            std = pd.to_numeric(std, errors='coerce').astype(float).to_numpy()
            idx = np.array(df_runs.index, dtype=float)
            # Remove non-finite values
            mask = np.isfinite(mean) & np.isfinite(std) & np.isfinite(idx)
            mean = mean[mask]
            std = std[mask]
            idx = idx[mask]
            if len(mean) == 0:
                continue
            # Convert accuracy to error
            error_mean = 100 - mean
            error_std = np.sqrt(std)  # std remains the same
            plt.plot(idx, error_mean, label=lbl, color=colcol, linewidth=2.5)
            plt.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colcol, alpha=0.2)
        
        plt.xlabel('Batch', fontsize=32)
        plt.ylabel('Mean Error Rate (%)', fontsize=32)
        plt.ylim(-5, 105)
        plt.tick_params(axis='both', labelsize=28)
        plt.legend(loc='best', fontsize=30)
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
        
        safe_om = str(omega_value).replace('.', '_')
        fname = os.path.join(figures_dir, f"errors_epochs_omega_{safe_om}_all_eps.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Figure 1 saved to: {fname}")
    
    # --- Figure 2: Barplot for a specific epoch (averaged over all epsilons) ---
    # Use the last epoch available in the data
    if not subset_param.empty:
        if EPOCH == "last":
            selected_epoch = subset_param['epoch'].max()
        else:
            selected_epoch = EPOCH
        epoch_data = subset_param[subset_param['epoch'] == selected_epoch]
        
        if not epoch_data.empty:
            # Calculate mean error for each test category at the selected epoch
            # Order: small_no_carry, small_carry, large_no_carry, large_carry
            cols_ordered = [
                "test_pairs_no_carry_small_accuracy",
                "test_pairs_carry_small_accuracy",
                "test_pairs_no_carry_large_accuracy",
                "test_pairs_carry_large_accuracy"
            ]
                        
            means = []
            stds = []
            for col in cols_ordered:
                if col in epoch_data.columns:
                    values = pd.to_numeric(epoch_data[col], errors='coerce').dropna()
                    # Convert accuracy to error
                    error_values = 100 - values
                    means.append(error_values.mean())
                    stds.append(error_values.std())
                else:
                    means.append(0)
                    stds.append(0)
            
            # Create grouped bar positions: 0, 0.8 for small; 2.2, 3.0 for large
            x_positions = [0, 0.8, 2.2, 3.0]
            safe_om = str(omega_value).replace('.', '_')
            
            # --- Barplot WITHOUT experimental RT ---
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(x_positions, means, yerr=np.sqrt(stds), capsize=5, color=colors, 
                         alpha=0.8, edgecolor='black', linewidth=1.5, width=0.7)
            
            ax.set_ylabel('Mean Error Rate (%)', fontsize=32)
            ax.set_xticks([0.4, 2.6])
            ax.set_xticklabels(['Small', 'Large'], fontsize=28)
            ax.set_xlabel('Problem Size', fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
            
            # Add legend
            legend_elements = [
                Patch(facecolor=colors[0], edgecolor='black', label='Small - No Carry', alpha=0.8),
                Patch(facecolor=colors[1], edgecolor='black', label='Small - Carry', alpha=0.8),
                Patch(facecolor=colors[2], edgecolor='black', label='Large - No Carry', alpha=0.8),
                Patch(facecolor=colors[3], edgecolor='black', label='Large - Carry', alpha=0.8)
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=30, framealpha=0.95)
            
            fname2 = os.path.join(figures_dir, f"barplot_errors_omega_{safe_om}_epoch_{int(selected_epoch)}.png")
            plt.savefig(fname2, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Figure 2 (without RT) saved to: {fname2}")
            
            # --- Barplot WITH experimental RT ---
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(x_positions, means, yerr=np.sqrt(stds), capsize=5, color=colors, 
                         alpha=0.8, edgecolor='black', linewidth=1.5, width=0.7)
            
            # Experimental reaction times data
            min_RT = 1250  # Minimum RT for scaling
            max_RT = 4137.5  # Maximum RT for scaling
            rt_values = [1400, 1800, 2700, 3200]  # RT(SNC), RT(SC), RT(LNC), RT(LC)
            
            # Create secondary y-axis for reaction times
            ax2 = ax.twinx()
            ax2.set_ylabel('Reaction Time (ms)', fontsize=32)
            ax2.set_ylim(min_RT, max_RT)
            ax2.tick_params(axis='y', labelsize=28)
            
            # Normalize RT values to the error axis scale (0-105) for plotting
            rt_normalized = [(rt - min_RT) / (max_RT - min_RT) * 105 for rt in rt_values]
            
            # Plot experimental data as black stars on the primary axis
            ax.scatter(x_positions, rt_normalized, marker='*', s=500, color='green', edgecolors='darkgreen', linewidth=1, 
                      zorder=5, label='Experimental RTs')
            
            ax.set_ylabel('Mean Error Rate (%)', fontsize=32)
            ax.set_xticks([0.4, 2.6])
            ax.set_xticklabels(['Small', 'Large'], fontsize=28)
            ax.set_xlabel('Problem Size', fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
            
            # Add legend with experimental RT
            legend_elements_with_rt = [
                Patch(facecolor=colors[0], edgecolor='black', label='Small - No Carry', alpha=0.8),
                Patch(facecolor=colors[1], edgecolor='black', label='Small - Carry', alpha=0.8),
                Patch(facecolor=colors[2], edgecolor='black', label='Large - No Carry', alpha=0.8),
                Patch(facecolor=colors[3], edgecolor='black', label='Large - Carry', alpha=0.8),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markeredgecolor='darkgreen',
                       markersize=15, label='Experimental RTs')
            ]
            ax.legend(handles=legend_elements_with_rt, loc='upper left', fontsize=32, framealpha=0.95)
            
            fname3 = os.path.join(figures_dir, f"barplot_errors_omega_{safe_om}_epoch_{int(selected_epoch)}_with_RT.png")
            plt.savefig(fname3, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Figure 2 (with RT) saved to: {fname3}")
        else:
            print(f"No data found for last epoch")
    else:
        print("No data available for barplot")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)