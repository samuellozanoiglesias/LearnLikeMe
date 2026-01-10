# USE: nohup python paper_figure_dissoluted_problem_size.py 2 STUDY1 STUDY2 WI argmax 0.10 > logs_paper_dissoluted_problem_size.out 2>&1 &

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
STUDY_NAME_1 = str(sys.argv[2]).upper()  # Name of the first study
STUDY_NAME_2 = str(sys.argv[3]).upper()  # Name of the second study
PARAM_TYPE = str(sys.argv[4]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[5]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[6])  # Specific omega value to analyze

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/comparison_{STUDY_NAME_1}_vs_{STUDY_NAME_2}"
RAW_DIR_1 = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME_1}/{PARAM_TYPE}/{MODEL_TYPE}_version"
RAW_DIR_2 = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME_2}/{PARAM_TYPE}/{MODEL_TYPE}_version"

def analyze_multidigit_module(raw_dir, omega_value, param_type):
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
    
def analyze_multidigit_module(raw_dir, omega_value, param_type):
    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        return None
    
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
        return None
    
    print(f"Found {len(subset_param)} rows for omega={omega_value}, param={param_type}")
    
    # Prepare pivot tables
    pw = {}
    for col in acc_cols:
        if col in subset_param.columns:
            pw[col] = subset_param.pivot_table(index='epoch', columns='run', values=col)
    
    return pw

def plot_study_data(ax, pw, study_title, is_upper=False):
    """Plot data for a single study on a given axis
    
    Args:
        ax: matplotlib axis
        pw: pivot table data
        study_title: title for the subplot
        is_upper: if True, hide x-axis labels (for upper subplot)
    """
    acc_cols = [
        "test_pairs_no_carry_small_accuracy",
        "test_pairs_carry_small_accuracy",
        "test_pairs_no_carry_large_accuracy",
        "test_pairs_carry_large_accuracy"
    ]
    acc_labels = ["Small - No Carry", "Small - Carry", "Large - No Carry", "Large - Carry"]
    colors = ["#5FA8FF", "#1C6CE5", "#F46A6A", "#D62828"]  # Sky blue, light pink, steel blue, indian red
    
    if not pw:
        return
    
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
        ax.plot(idx, error_mean, label=lbl, color=colcol, linewidth=2.5)
        ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colcol, alpha=0.2)
    
    # X-axis configuration
    if is_upper:
        ax.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels but keep ticks
    else:
        ax.set_xlabel('Batch', fontsize=32)
    
    ax.set_ylabel('Mean Error Rate (%)', fontsize=32)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(loc='best', fontsize=32)
    ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_title(study_title, fontsize=36, fontweight='bold')

# Main execution
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data for both studies
print(f"\n{'='*60}")
print(f"Loading data for Study 1: {STUDY_NAME_1}")
print(f"{'='*60}")
pw_study1 = analyze_multidigit_module(RAW_DIR_1, OMEGA_VALUE, PARAM_TYPE)

print(f"\n{'='*60}")
print(f"Loading data for Study 2: {STUDY_NAME_2}")
print(f"{'='*60}")
pw_study2 = analyze_multidigit_module(RAW_DIR_2, OMEGA_VALUE, PARAM_TYPE)

# Create double plot
if pw_study1 or pw_study2:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Hardcoded titles - modify these as needed
    title_study1 = "Balanced stimuli"
    title_study2 = "Exponentially decayed stimuli"
    
    if pw_study1:
        plot_study_data(ax1, pw_study1, title_study1, is_upper=True)
    else:
        ax1.text(0.5, 0.5, f'No data available for {STUDY_NAME_1}', 
                ha='center', va='center', fontsize=32, transform=ax1.transAxes)
    
    if pw_study2:
        plot_study_data(ax2, pw_study2, title_study2, is_upper=False)
    else:
        ax2.text(0.5, 0.5, f'No data available for {STUDY_NAME_2}', 
                ha='center', va='center', fontsize=32, transform=ax2.transAxes)
    
    plt.tight_layout()
    
    safe_om = str(OMEGA_VALUE).replace('.', '_')
    fname = os.path.join(FIGURES_DIR, f"comparison_errors_epochs_omega_{safe_om}_all_eps.png")
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n{'='*60}")
    print(f"Comparison figure saved to: {fname}")
    print(f"{'='*60}")
else:
    print("\nNo data available for either study. No figure created.")