# USE: nohup python paper_anova_effects_sweep.py 2 STUDY RI argmax 0.15 100 500 50 > log_anova_sweep.out 2>&1 &

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[5])  # Specific omega value to analyze
EPOCH_START = int(sys.argv[6])  # Starting epoch for sweep
EPOCH_END = int(sys.argv[7])  # Ending epoch for sweep
EPOCH_STEP = int(sys.argv[8]) if len(sys.argv) > 8 else 50  # Step size for epoch sweep
N_REPETITIONS_PER_SAMPLE = 5 # Number of repetitions averaged per accuracy value (e.g., 10)

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"
FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"

def analyze_single_epoch(combined_logs, param_type, omega_value, epoch, n_repetitions_per_sample=1):
    """
    Analyze a single epoch and return effect sizes and statistics.
    Note: Each accuracy value represents the average of 10 repetitions,
    which affects sample size calculations and degrees of freedom.
    """
    # Filter for the specific omega and param_init_type
    subset_param = combined_logs[
        (combined_logs['param_init_type'] == param_type) & 
        (combined_logs['omega'] == omega_value)
    ]
    
    if subset_param.empty:
        return None
    
    epoch_data = subset_param[subset_param['epoch'] == epoch]
    
    if epoch_data.empty:
        return None
    
    # Extract error values (convert accuracy to error: 100 - accuracy)
    small_no_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_no_carry_small_accuracy"], errors='coerce').dropna()
    small_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_carry_small_accuracy"], errors='coerce').dropna()
    large_no_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_no_carry_large_accuracy"], errors='coerce').dropna()
    large_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_carry_large_accuracy"], errors='coerce').dropna()
    
    if len(small_no_carry_errors) == 0 or len(small_carry_errors) == 0 or len(large_no_carry_errors) == 0 or len(large_carry_errors) == 0:
        return None
    
    # ========================================
    # CARRY-OVER EFFECT ANOVA
    # ========================================
    no_carry_all = pd.concat([small_no_carry_errors, large_no_carry_errors]).reset_index(drop=True)
    carry_all = pd.concat([small_carry_errors, large_carry_errors]).reset_index(drop=True)
    
    # Expand each observation by repeating it n_repetitions_per_sample times
    no_carry_expanded = np.repeat(no_carry_all.values, n_repetitions_per_sample)
    carry_expanded = np.repeat(carry_all.values, n_repetitions_per_sample)
    
    # ANOVA using expanded data
    f_stat_carry, p_value_carry = stats.f_oneway(no_carry_expanded, carry_expanded)
    
    # Effect size (eta-squared) - calculated from original data
    grand_mean = pd.concat([no_carry_all, carry_all]).mean()
    ss_between = len(no_carry_all) * (no_carry_all.mean() - grand_mean)**2 + len(carry_all) * (carry_all.mean() - grand_mean)**2
    ss_total = np.sum((pd.concat([no_carry_all, carry_all]) - grand_mean)**2)
    eta_squared_carry = ss_between / ss_total if ss_total > 0 else 0
    
    # ========================================
    # PROBLEM SIZE EFFECT ANOVA
    # ========================================
    small_all = pd.concat([small_no_carry_errors, small_carry_errors]).reset_index(drop=True)
    large_all = pd.concat([large_no_carry_errors, large_carry_errors]).reset_index(drop=True)
    
    # Expand each observation by repeating it n_repetitions_per_sample times
    small_expanded = np.repeat(small_all.values, n_repetitions_per_sample)
    large_expanded = np.repeat(large_all.values, n_repetitions_per_sample)
    
    # ANOVA using expanded data
    f_stat_size, p_value_size = stats.f_oneway(small_expanded, large_expanded)
    
    # Effect size (eta-squared) - calculated from original data
    grand_mean_size = pd.concat([small_all, large_all]).mean()
    ss_between_size = len(small_all) * (small_all.mean() - grand_mean_size)**2 + len(large_all) * (large_all.mean() - grand_mean_size)**2
    ss_total_size = np.sum((pd.concat([small_all, large_all]) - grand_mean_size)**2)
    eta_squared_size = ss_between_size / ss_total_size if ss_total_size > 0 else 0
    
    # ========================================
    # TWO-WAY ANOVA (Problem Size × Carry)
    # ========================================
    # Expand each observation by repeating it n_repetitions_per_sample times
    data_list = []
    for val in small_no_carry_errors:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Size': 'Small', 'Carry': 'No Carry'})
    for val in small_carry_errors:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Size': 'Small', 'Carry': 'Carry'})
    for val in large_no_carry_errors:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Size': 'Large', 'Carry': 'No Carry'})
    for val in large_carry_errors:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Size': 'Large', 'Carry': 'Carry'})
    
    df_anova = pd.DataFrame(data_list)
    groups = df_anova.groupby(['Size', 'Carry'])['Error'].apply(list)
    
    small_no_carry = np.array(groups[('Small', 'No Carry')])
    small_carry = np.array(groups[('Small', 'Carry')])
    large_no_carry = np.array(groups[('Large', 'No Carry')])
    large_carry = np.array(groups[('Large', 'Carry')])
    
    all_data = np.concatenate([small_no_carry, small_carry, large_no_carry, large_carry])
    grand_mean_2way = np.mean(all_data)
    n_total = len(all_data)
    n_per_cell = len(small_no_carry)  # Already expanded
    
    # Main effect of Size
    mean_small = np.mean(np.concatenate([small_no_carry, small_carry]))
    mean_large = np.mean(np.concatenate([large_no_carry, large_carry]))
    ss_size = 2 * n_per_cell * ((mean_small - grand_mean_2way)**2 + (mean_large - grand_mean_2way)**2)
    df_size = 1
    ms_size = ss_size / df_size
    
    # Main effect of Carry
    mean_no_carry = np.mean(np.concatenate([small_no_carry, large_no_carry]))
    mean_carry = np.mean(np.concatenate([small_carry, large_carry]))
    ss_carry = 2 * n_per_cell * ((mean_no_carry - grand_mean_2way)**2 + (mean_carry - grand_mean_2way)**2)
    df_carry = 1
    ms_carry = ss_carry / df_carry
    
    # Interaction effect
    mean_small_no_carry = np.mean(small_no_carry)
    mean_small_carry = np.mean(small_carry)
    mean_large_no_carry = np.mean(large_no_carry)
    mean_large_carry = np.mean(large_carry)
    
    ss_interaction = n_per_cell * (
        (mean_small_no_carry - mean_small - mean_no_carry + grand_mean_2way)**2 +
        (mean_small_carry - mean_small - mean_carry + grand_mean_2way)**2 +
        (mean_large_no_carry - mean_large - mean_no_carry + grand_mean_2way)**2 +
        (mean_large_carry - mean_large - mean_carry + grand_mean_2way)**2
    )
    df_interaction = 1
    ms_interaction = ss_interaction / df_interaction
    
    # Within-group (error) sum of squares
    ss_within = (np.sum((small_no_carry - mean_small_no_carry)**2) +
                 np.sum((small_carry - mean_small_carry)**2) +
                 np.sum((large_no_carry - mean_large_no_carry)**2) +
                 np.sum((large_carry - mean_large_carry)**2))
    df_within = n_total - 4
    ms_within = ss_within / df_within
    
    # F-statistics
    f_size_2way = ms_size / ms_within
    f_carry_2way = ms_carry / ms_within
    f_interaction = ms_interaction / ms_within
    
    # P-values
    p_size_2way = 1 - stats.f.cdf(f_size_2way, df_size, df_within)
    p_carry_2way = 1 - stats.f.cdf(f_carry_2way, df_carry, df_within)
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_within)
    
    return {
        'epoch': epoch,
        'n_samples': n_per_cell // n_repetitions_per_sample,  # Original sample size
        'n_samples_expanded': n_per_cell,  # Expanded sample size
        'carry_f': f_stat_carry,
        'carry_p': p_value_carry,
        'carry_eta2': eta_squared_carry,
        'size_f': f_stat_size,
        'size_p': p_value_size,
        'size_eta2': eta_squared_size,
        'interaction_f': f_interaction,
        'interaction_p': p_interaction,
        'carry_2way_f': f_carry_2way,
        'carry_2way_p': p_carry_2way,
        'size_2way_f': f_size_2way,
        'size_2way_p': p_size_2way,
        'mean_small_no_carry': small_no_carry_errors.mean(),
        'mean_small_carry': small_carry_errors.mean(),
        'mean_large_no_carry': large_no_carry_errors.mean(),
        'mean_large_carry': large_carry_errors.mean()
    }

def run_epoch_sweep(raw_dir, figures_dir, omega_value, param_type, epoch_start, epoch_end, epoch_step, n_repetitions_per_sample=1):
    """
    Run ANOVA analysis across multiple epochs and find the epoch with highest combined effects.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        return
    
    # Read CSV with low_memory=False to avoid dtype warning
    combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
    print(f"Combined logs loaded from: {combined_logs_path}")
    
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
    
    print(f"Starting epoch sweep from {epoch_start} to {epoch_end} with step {epoch_step}")
    
    # Run analysis for each epoch
    results = []
    epochs = range(epoch_start, epoch_end + 1, epoch_step)
    
    for epoch in epochs:
        print(f"Analyzing epoch {epoch}...")
        result = analyze_single_epoch(combined_logs, param_type, omega_value, epoch, n_repetitions_per_sample)
        if result is not None:
            results.append(result)
    
    if not results:
        print("No valid results found across the epoch range")
        return
    
    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    
    # Calculate combined effect metric (product of both eta-squared values)
    df_results['combined_eta2'] = df_results['carry_eta2'] * df_results['size_eta2']
    
    # Calculate combined effect metric (sum of both eta-squared values)
    df_results['sum_eta2'] = df_results['carry_eta2'] + df_results['size_eta2']
    
    # Find epoch with maximum combined effect
    max_combined_idx = df_results['combined_eta2'].idxmax()
    max_sum_idx = df_results['sum_eta2'].idxmax()
    best_epoch_product = df_results.loc[max_combined_idx, 'epoch']
    best_epoch_sum = df_results.loc[max_sum_idx, 'epoch']
    
    # Save results to text file
    safe_om = str(omega_value).replace('.', '_')
    output_filename = f"ANOVA_sweep_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.txt"
    output_path = os.path.join(figures_dir, output_filename)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANOVA EFFECT SIZE SWEEP ACROSS EPOCHS\n")
        f.write("="*80 + "\n")
        f.write(f"\nConfiguration: {NUMBER_SIZE}-digit, {STUDY_NAME}, {PARAM_TYPE}, {MODEL_TYPE}\n")
        f.write(f"Omega: {omega_value}\n")
        f.write(f"Epoch range: {epoch_start} to {epoch_end} (step: {epoch_step})\n")
        f.write(f"Total epochs analyzed: {len(results)}\n")
        f.write(f"\nNote: P-values adjusted for true sample size (each observation = 10 repetitions)\n")
        f.write("\n" + "="*80 + "\n")
        f.write("BEST EPOCHS FOR COMBINED EFFECTS\n")
        f.write("="*80 + "\n")
        f.write(f"\nBest epoch (product of η²): {int(best_epoch_product)}\n")
        f.write(f"  Carry η²: {df_results.loc[max_combined_idx, 'carry_eta2']:.4f}\n")
        f.write(f"  Size η²: {df_results.loc[max_combined_idx, 'size_eta2']:.4f}\n")
        f.write(f"  Product: {df_results.loc[max_combined_idx, 'combined_eta2']:.6f}\n")
        f.write(f"\nBest epoch (sum of η²): {int(best_epoch_sum)}\n")
        f.write(f"  Carry η²: {df_results.loc[max_sum_idx, 'carry_eta2']:.4f}\n")
        f.write(f"  Size η²: {df_results.loc[max_sum_idx, 'size_eta2']:.4f}\n")
        f.write(f"  Sum: {df_results.loc[max_sum_idx, 'sum_eta2']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS BY EPOCH\n")
        f.write("="*80 + "\n\n")
        
        for _, row in df_results.iterrows():
            f.write(f"Epoch {int(row['epoch'])}:\n")
            f.write(f"  Carry Effect: F={row['carry_f']:.4f}, p={row['carry_p']:.6f}, η²={row['carry_eta2']:.4f}")
            if row['carry_p'] < 0.001:
                f.write(" ***\n")
            elif row['carry_p'] < 0.01:
                f.write(" **\n")
            elif row['carry_p'] < 0.05:
                f.write(" *\n")
            else:
                f.write(" n.s.\n")
            
            f.write(f"  Size Effect: F={row['size_f']:.4f}, p={row['size_p']:.6f}, η²={row['size_eta2']:.4f}")
            if row['size_p'] < 0.001:
                f.write(" ***\n")
            elif row['size_p'] < 0.01:
                f.write(" **\n")
            elif row['size_p'] < 0.05:
                f.write(" *\n")
            else:
                f.write(" n.s.\n")
            
            f.write(f"  Interaction: F={row['interaction_f']:.4f}, p={row['interaction_p']:.6f}")
            if row['interaction_p'] < 0.001:
                f.write(" ***\n")
            elif row['interaction_p'] < 0.01:
                f.write(" **\n")
            elif row['interaction_p'] < 0.05:
                f.write(" *\n")
            else:
                f.write(" n.s.\n")
            
            f.write(f"  Combined η² (product): {row['combined_eta2']:.6f}\n")
            f.write(f"  Combined η² (sum): {row['sum_eta2']:.4f}\n")
            f.write(f"  Sample size per condition: {int(row['n_samples'])} observations ({int(row['n_samples_expanded'])} expanded)\n\n")
    
    print(f"\nResults saved to: {output_path}")
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Effect sizes over epochs
    ax1 = axes[0, 0]
    ax1.plot(df_results['epoch'], df_results['carry_eta2'], 'o-', label='Carry Effect', 
             color='#1C6CE5', linewidth=2.5, markersize=8)
    ax1.plot(df_results['epoch'], df_results['size_eta2'], 's-', label='Problem Size Effect', 
             color='#D62828', linewidth=2.5, markersize=8)
    ax1.axvline(best_epoch_product, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Best (product): {int(best_epoch_product)}')
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Effect Size (η²)', fontsize=20)
    ax1.set_title('Effect Sizes Across Epochs', fontsize=22)
    ax1.legend(fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=16)
    
    # Plot 2: Combined effect metrics
    ax2 = axes[0, 1]
    ax2.plot(df_results['epoch'], df_results['combined_eta2'], 'o-', 
             color='purple', linewidth=2.5, markersize=8, label='Product of η²')
    ax2.axvline(best_epoch_product, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Max: {int(best_epoch_product)}')
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('Combined η² (Product)', fontsize=20)
    ax2.set_title('Combined Effect Size (Product)', fontsize=22)
    ax2.legend(fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=16)
    
    # Plot 3: F-statistics over epochs
    ax3 = axes[1, 0]
    ax3.plot(df_results['epoch'], df_results['carry_f'], 'o-', label='Carry F-stat', 
             color='#1C6CE5', linewidth=2.5, markersize=8)
    ax3.plot(df_results['epoch'], df_results['size_f'], 's-', label='Size F-stat', 
             color='#D62828', linewidth=2.5, markersize=8)
    ax3.axvline(best_epoch_product, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=20)
    ax3.set_ylabel('F-statistic', fontsize=20)
    ax3.set_title('F-statistics Across Epochs', fontsize=22)
    ax3.legend(fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=16)
    
    # Plot 4: Sum of effect sizes
    ax4 = axes[1, 1]
    ax4.plot(df_results['epoch'], df_results['sum_eta2'], 'o-', 
             color='orange', linewidth=2.5, markersize=8, label='Sum of η²')
    ax4.axvline(best_epoch_sum, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Max: {int(best_epoch_sum)}')
    ax4.set_xlabel('Epoch', fontsize=20)
    ax4.set_ylabel('Combined η² (Sum)', fontsize=20)
    ax4.set_title('Combined Effect Size (Sum)', fontsize=22)
    ax4.legend(fontsize=16)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = f"ANOVA_sweep_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.png"
    fig_path = os.path.join(figures_dir, fig_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {fig_path}")
    
    # Save CSV with all results
    csv_filename = f"ANOVA_sweep_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.csv"
    csv_path = os.path.join(figures_dir, csv_filename)
    df_results.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nNote: P-values adjusted for true sample size (10 repetitions per observation)")
    print(f"\nBest epoch (product of η²): {int(best_epoch_product)}")
    print(f"  Carry η²: {df_results.loc[max_combined_idx, 'carry_eta2']:.4f}")
    print(f"  Size η²: {df_results.loc[max_combined_idx, 'size_eta2']:.4f}")
    print(f"  Product: {df_results.loc[max_combined_idx, 'combined_eta2']:.6f}")
    print(f"\nBest epoch (sum of η²): {int(best_epoch_sum)}")
    print(f"  Carry η²: {df_results.loc[max_sum_idx, 'carry_eta2']:.4f}")
    print(f"  Size η²: {df_results.loc[max_sum_idx, 'size_eta2']:.4f}")
    print(f"  Sum: {df_results.loc[max_sum_idx, 'sum_eta2']:.4f}")

# Run epoch sweep
run_epoch_sweep(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE, EPOCH_START, EPOCH_END, EPOCH_STEP, N_REPETITIONS_PER_SAMPLE)
