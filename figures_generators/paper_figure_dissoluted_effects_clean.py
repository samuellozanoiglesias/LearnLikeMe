# USE: nohup python paper_figure_dissoluted_effects_clean.py 2 STUDY1 STUDY2 RI argmax 0.10 > logs_paper_dissoluted_effects_clean.out 2>&1 &

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

def plot_problem_size_effect_clean(ax, pw_balanced, pw_exponential, fixed_batch_bal, fixed_batch_exp, colors_balanced, colors_exponential, colors_dashed, is_upper=False):
    """Plot problem size effect (small vs large) for both regimes - CLEAN VERSION without annotations"""
    
    # Store data for fancy line drawing
    balanced_small_data = None
    balanced_large_data = None
    exponential_small_data = None
    exponential_large_data = None
    
    # Plot balanced regime (dashed lines)
    if pw_balanced:
        # Small problems (average of no carry and carry)
        small_cols = ["test_pairs_no_carry_small_accuracy", "test_pairs_carry_small_accuracy"]
        small_data = []
        for col in small_cols:
            if col in pw_balanced and pw_balanced[col] is not None and not pw_balanced[col].empty:
                small_data.append(pw_balanced[col])
        
        if small_data:
            # Average across the two small problem types
            small_combined = pd.concat(small_data, axis=1)
            small_mean = small_combined.mean(axis=1)
            small_std = small_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            small_mean = pd.to_numeric(small_mean, errors='coerce').astype(float).to_numpy()
            small_std = pd.to_numeric(small_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(small_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(small_mean) & np.isfinite(small_std) & np.isfinite(idx)
            small_mean = small_mean[mask]
            small_std = small_std[mask]
            idx = idx[mask]
            
            if len(small_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - small_mean
                error_std = np.sqrt(small_std)
                balanced_small_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='Small (Uniform)', color=colors_balanced[0], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_balanced[0], alpha=0.2)
        
        # Large problems (average of no carry and carry)
        large_cols = ["test_pairs_no_carry_large_accuracy", "test_pairs_carry_large_accuracy"]
        large_data = []
        for col in large_cols:
            if col in pw_balanced and pw_balanced[col] is not None and not pw_balanced[col].empty:
                large_data.append(pw_balanced[col])
        
        if large_data:
            # Average across the two large problem types
            large_combined = pd.concat(large_data, axis=1)
            large_mean = large_combined.mean(axis=1)
            large_std = large_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            large_mean = pd.to_numeric(large_mean, errors='coerce').astype(float).to_numpy()
            large_std = pd.to_numeric(large_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(large_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(large_mean) & np.isfinite(large_std) & np.isfinite(idx)
            large_mean = large_mean[mask]
            large_std = large_std[mask]
            idx = idx[mask]
            
            if len(large_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - large_mean
                error_std = np.sqrt(large_std)
                balanced_large_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='Large (Uniform)', color=colors_balanced[1], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_balanced[1], alpha=0.2)
    
    # Plot exponential regime (solid lines)
    if pw_exponential:
        # Small problems (average of no carry and carry)
        small_cols = ["test_pairs_no_carry_small_accuracy", "test_pairs_carry_small_accuracy"]
        small_data = []
        for col in small_cols:
            if col in pw_exponential and pw_exponential[col] is not None and not pw_exponential[col].empty:
                small_data.append(pw_exponential[col])
        
        if small_data:
            # Average across the two small problem types
            small_combined = pd.concat(small_data, axis=1)
            small_mean = small_combined.mean(axis=1)
            small_std = small_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            small_mean = pd.to_numeric(small_mean, errors='coerce').astype(float).to_numpy()
            small_std = pd.to_numeric(small_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(small_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(small_mean) & np.isfinite(small_std) & np.isfinite(idx)
            small_mean = small_mean[mask]
            small_std = small_std[mask]
            idx = idx[mask]
            
            if len(small_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - small_mean
                error_std = np.sqrt(small_std)
                exponential_small_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='Small (Naturalistic)', color=colors_exponential[0], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_exponential[0], alpha=0.2)
        
        # Large problems (average of no carry and carry)
        large_cols = ["test_pairs_no_carry_large_accuracy", "test_pairs_carry_large_accuracy"]
        large_data = []
        for col in large_cols:
            if col in pw_exponential and pw_exponential[col] is not None and not pw_exponential[col].empty:
                large_data.append(pw_exponential[col])
        
        if large_data:
            # Average across the two large problem types
            large_combined = pd.concat(large_data, axis=1)
            large_mean = large_combined.mean(axis=1)
            large_std = large_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            large_mean = pd.to_numeric(large_mean, errors='coerce').astype(float).to_numpy()
            large_std = pd.to_numeric(large_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(large_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(large_mean) & np.isfinite(large_std) & np.isfinite(idx)
            large_mean = large_mean[mask]
            large_std = large_std[mask]
            idx = idx[mask]
            
            if len(large_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - large_mean
                error_std = np.sqrt(large_std)
                exponential_large_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='Large (Naturalistic)', color=colors_exponential[1], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_exponential[1], alpha=0.2)
    
    # Add fancy decorative vertical lines at fixed batch values
    # Uniform regime: vertical line at batch 400
    if balanced_small_data is not None and balanced_large_data is not None:
        idx_small, error_small = balanced_small_data
        idx_large, error_large = balanced_large_data
        
        # Check if batch 400 is within the range of both datasets
        if (idx_small.min() <= fixed_batch_bal <= idx_small.max() and 
            idx_large.min() <= fixed_batch_bal <= idx_large.max()):
            small_y = np.interp(fixed_batch_bal, idx_small, error_small)
            large_y = np.interp(fixed_batch_bal, idx_large, error_large)
            
            # Draw elegant dashed line with decorative caps
            ax.plot([fixed_batch_bal, fixed_batch_bal], [small_y, large_y], 
                   color=colors_dashed[0], linewidth=4, linestyle='--', alpha=0.85, solid_capstyle='round')
            # Add decorative diamond markers at endpoints
            ax.plot(fixed_batch_bal, small_y, marker='D', markersize=8, color=colors_dashed[0], markerfacecolor='white', markeredgewidth=2)
            ax.plot(fixed_batch_bal, large_y, marker='D', markersize=8, color=colors_dashed[0], markerfacecolor='white', markeredgewidth=2)
    
    # Naturalistic regime: vertical line at batch 500
    if exponential_small_data is not None and exponential_large_data is not None:
        idx_small, error_small = exponential_small_data
        idx_large, error_large = exponential_large_data
        
        # Check if batch 500 is within the range of both datasets
        if (idx_small.min() <= fixed_batch_exp <= idx_small.max() and 
            idx_large.min() <= fixed_batch_exp <= idx_large.max()):
            small_y = np.interp(fixed_batch_exp, idx_small, error_small)
            large_y = np.interp(fixed_batch_exp, idx_large, error_large)
            
            # Draw elegant dashed line with decorative caps
            ax.plot([fixed_batch_exp, fixed_batch_exp], [small_y, large_y], 
                   color=colors_dashed[1], linewidth=4, linestyle='--', alpha=0.85, solid_capstyle='round')
            # Add decorative diamond markers at endpoints
            ax.plot(fixed_batch_exp, small_y, marker='D', markersize=8, color=colors_dashed[1], markerfacecolor='white', markeredgewidth=2)
            ax.plot(fixed_batch_exp, large_y, marker='D', markersize=8, color=colors_dashed[1], markerfacecolor='white', markeredgewidth=2)

    # Configure axis
    if is_upper:
        ax.tick_params(axis='x', labelbottom=False)
    else:
        ax.set_xlabel('Batch', fontsize=32)
    
    ax.set_ylabel('Mean Error Rate (%)', fontsize=32)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(loc='best', fontsize=24)
    ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_title('Problem Size Effect', fontsize=36, fontweight='bold')

def plot_carry_effect_clean(ax, pw_balanced, pw_exponential, fixed_batch_bal, fixed_batch_exp, colors_balanced, colors_exponential, colors_dashed, is_upper=False):
    """Plot carry effect (no carry vs carry) for both regimes - CLEAN VERSION without annotations"""
        
    # Store data for fancy line drawing
    balanced_no_carry_data = None
    balanced_carry_data = None
    exponential_no_carry_data = None
    exponential_carry_data = None
    
    # Plot balanced regime (dashed lines)
    if pw_balanced:
        # No carry problems (average of small and large)
        no_carry_cols = ["test_pairs_no_carry_small_accuracy", "test_pairs_no_carry_large_accuracy"]
        no_carry_data = []
        for col in no_carry_cols:
            if col in pw_balanced and pw_balanced[col] is not None and not pw_balanced[col].empty:
                no_carry_data.append(pw_balanced[col])
        
        if no_carry_data:
            # Average across the two no carry problem types
            no_carry_combined = pd.concat(no_carry_data, axis=1)
            no_carry_mean = no_carry_combined.mean(axis=1)
            no_carry_std = no_carry_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            no_carry_mean = pd.to_numeric(no_carry_mean, errors='coerce').astype(float).to_numpy()
            no_carry_std = pd.to_numeric(no_carry_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(no_carry_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(no_carry_mean) & np.isfinite(no_carry_std) & np.isfinite(idx)
            no_carry_mean = no_carry_mean[mask]
            no_carry_std = no_carry_std[mask]
            idx = idx[mask]
            
            if len(no_carry_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - no_carry_mean
                error_std = np.sqrt(no_carry_std)
                balanced_no_carry_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='No Carry (Uniform)', color=colors_balanced[0], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_balanced[0], alpha=0.2)
        
        # Carry problems (average of small and large)
        carry_cols = ["test_pairs_carry_small_accuracy", "test_pairs_carry_large_accuracy"]
        carry_data = []
        for col in carry_cols:
            if col in pw_balanced and pw_balanced[col] is not None and not pw_balanced[col].empty:
                carry_data.append(pw_balanced[col])
        
        if carry_data:
            # Average across the two carry problem types
            carry_combined = pd.concat(carry_data, axis=1)
            carry_mean = carry_combined.mean(axis=1)
            carry_std = carry_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            carry_mean = pd.to_numeric(carry_mean, errors='coerce').astype(float).to_numpy()
            carry_std = pd.to_numeric(carry_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(carry_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(carry_mean) & np.isfinite(carry_std) & np.isfinite(idx)
            carry_mean = carry_mean[mask]
            carry_std = carry_std[mask]
            idx = idx[mask]
            
            if len(carry_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - carry_mean
                error_std = np.sqrt(carry_std)
                balanced_carry_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='Carry (Uniform)', color=colors_balanced[1], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_balanced[1], alpha=0.2)
    
    # Plot exponential regime (solid lines)
    if pw_exponential:
        # No carry problems (average of small and large)
        no_carry_cols = ["test_pairs_no_carry_small_accuracy", "test_pairs_no_carry_large_accuracy"]
        no_carry_data = []
        for col in no_carry_cols:
            if col in pw_exponential and pw_exponential[col] is not None and not pw_exponential[col].empty:
                no_carry_data.append(pw_exponential[col])
        
        if no_carry_data:
            # Average across the two no carry problem types
            no_carry_combined = pd.concat(no_carry_data, axis=1)
            no_carry_mean = no_carry_combined.mean(axis=1)
            no_carry_std = no_carry_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            no_carry_mean = pd.to_numeric(no_carry_mean, errors='coerce').astype(float).to_numpy()
            no_carry_std = pd.to_numeric(no_carry_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(no_carry_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(no_carry_mean) & np.isfinite(no_carry_std) & np.isfinite(idx)
            no_carry_mean = no_carry_mean[mask]
            no_carry_std = no_carry_std[mask]
            idx = idx[mask]
            
            if len(no_carry_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - no_carry_mean
                error_std = np.sqrt(no_carry_std)
                exponential_no_carry_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='No Carry (Naturalistic)', color=colors_exponential[0], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_exponential[0], alpha=0.2)
        
        # Carry problems (average of small and large)
        carry_cols = ["test_pairs_carry_small_accuracy", "test_pairs_carry_large_accuracy"]
        carry_data = []
        for col in carry_cols:
            if col in pw_exponential and pw_exponential[col] is not None and not pw_exponential[col].empty:
                carry_data.append(pw_exponential[col])
        
        if carry_data:
            # Average across the two carry problem types
            carry_combined = pd.concat(carry_data, axis=1)
            carry_mean = carry_combined.mean(axis=1)
            carry_std = carry_combined.std(axis=1).fillna(0)
            
            # Convert to numpy arrays
            carry_mean = pd.to_numeric(carry_mean, errors='coerce').astype(float).to_numpy()
            carry_std = pd.to_numeric(carry_std, errors='coerce').astype(float).to_numpy()
            idx = np.array(carry_combined.index, dtype=float)
            
            # Remove non-finite values
            mask = np.isfinite(carry_mean) & np.isfinite(carry_std) & np.isfinite(idx)
            carry_mean = carry_mean[mask]
            carry_std = carry_std[mask]
            idx = idx[mask]
            
            if len(carry_mean) > 0:
                # Convert accuracy to error
                error_mean = 100 - carry_mean
                error_std = np.sqrt(carry_std)
                exponential_carry_data = (idx, error_mean)
                ax.plot(idx, error_mean, label='Carry (Naturalistic)', color=colors_exponential[1], linewidth=2.5, linestyle='-')
                ax.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colors_exponential[1], alpha=0.2)
    
    # Add fancy decorative vertical lines at fixed batch values
    # Uniform regime: vertical line at batch 400
    if balanced_no_carry_data is not None and balanced_carry_data is not None:
        idx_no_carry, error_no_carry = balanced_no_carry_data
        idx_carry, error_carry = balanced_carry_data
        
        # Check if batch 400 is within the range of both datasets
        if (idx_no_carry.min() <= fixed_batch_bal <= idx_no_carry.max() and 
            idx_carry.min() <= fixed_batch_bal <= idx_carry.max()):
            no_carry_y = np.interp(fixed_batch_bal, idx_no_carry, error_no_carry)
            carry_y = np.interp(fixed_batch_bal, idx_carry, error_carry)
            
            # Draw elegant dashed line with decorative caps
            ax.plot([fixed_batch_bal, fixed_batch_bal], [no_carry_y, carry_y], 
                   color=colors_dashed[0], linewidth=4, linestyle='--', alpha=0.85, solid_capstyle='round')
            # Add decorative diamond markers at endpoints
            ax.plot(fixed_batch_bal, no_carry_y, marker='D', markersize=8, color=colors_dashed[0], markerfacecolor='white', markeredgewidth=2)
            ax.plot(fixed_batch_bal, carry_y, marker='D', markersize=8, color=colors_dashed[0], markerfacecolor='white', markeredgewidth=2)
    
    # Naturalistic regime: vertical line at batch 500
    if exponential_no_carry_data is not None and exponential_carry_data is not None:
        idx_no_carry, error_no_carry = exponential_no_carry_data
        idx_carry, error_carry = exponential_carry_data
        
        # Check if batch 500 is within the range of both datasets
        if (idx_no_carry.min() <= fixed_batch_exp <= idx_no_carry.max() and 
            idx_carry.min() <= fixed_batch_exp <= idx_carry.max()):
            no_carry_y = np.interp(fixed_batch_exp, idx_no_carry, error_no_carry)
            carry_y = np.interp(fixed_batch_exp, idx_carry, error_carry)
            
            # Draw elegant dashed line with decorative caps
            ax.plot([fixed_batch_exp, fixed_batch_exp], [no_carry_y, carry_y], 
                   color=colors_dashed[1], linewidth=4, linestyle='--', alpha=0.85, solid_capstyle='round')
            # Add decorative diamond markers at endpoints
            ax.plot(fixed_batch_exp, no_carry_y, marker='D', markersize=8, color=colors_dashed[1], markerfacecolor='white', markeredgewidth=2)
            ax.plot(fixed_batch_exp, carry_y, marker='D', markersize=8, color=colors_dashed[1], markerfacecolor='white', markeredgewidth=2)

    # Configure axis
    if is_upper:
        ax.tick_params(axis='x', labelbottom=False)
    else:
        ax.set_xlabel('Batch', fontsize=32)
    
    ax.set_ylabel('Mean Error Rate (%)', fontsize=32)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(loc='best', fontsize=24)
    ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_title('Carry-over Effect', fontsize=36, fontweight='bold')

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

# Uniform: blue (darker=small, lighter=large), Naturalistic: red (darker=small, lighter=large)
colors_balanced = ["#87c387", "#259225"]  # Darker blue (no carry), lighter blue (carry)
colors_exponential = ["#b2a6d1", "#7F6BB3"]  # Darker red (no carry), lighter red (carry)
colors_dashed = ["#1E561E", "#383051"]

# Create clean version with fancy vertical lines
if pw_study1 or pw_study2:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    fixed_batch_bal = 450
    fixed_batch_exp = 550

    # Upper graph: Problem Size Effect (clean)
    plot_problem_size_effect_clean(ax1, pw_study1, pw_study2, fixed_batch_bal, fixed_batch_exp, colors_balanced, colors_exponential, colors_dashed, is_upper=True)
    
    # Lower graph: Carry Effect (clean) 
    plot_carry_effect_clean(ax2, pw_study1, pw_study2, fixed_batch_bal, fixed_batch_exp, colors_balanced, colors_exponential, colors_dashed, is_upper=False)
    
    safe_om = str(OMEGA_VALUE).replace('.', '_')
    fname_clean = os.path.join(FIGURES_DIR, f"comparison_effects_clean_omega_{safe_om}.png")
    plt.savefig(fname_clean, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n{'='*60}")
    print(f"Clean comparison figure with fancy lines saved to: {fname_clean}")
    print(f"{'='*60}")
else:
    print("\nNo data available for either study. No figure created.")