# USE: nohup python paper_anova_effects.py 2 STUDY RI argmax 0.15 500 > log_anova_effects.out 2>&1 &

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[5])  # Specific omega value to analyze
EPOCH = int(sys.argv[6]) if len(sys.argv) > 6 else "last"  # Specific epoch for analysis
N_REPETITIONS_PER_SAMPLE = 2 # Number of repetitions averaged per accuracy value (e.g., 10)

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

def perform_anova_analysis(raw_dir, figures_dir, omega_value, param_type, epoch, n_repetitions_per_sample=1):
    """
    Perform ANOVA tests for carry-over effect and problem size effect.
    
    Carry-over effect: Compare problems with carry vs without carry
    Problem size effect: Compare large vs small problems
    
    Note: Each accuracy value represents the average of 10 repetitions,
    which affects sample size calculations and degrees of freedom.
    """
    # Create output directory
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create output file
    safe_om = str(omega_value).replace('.', '_')
    if epoch == "last":
        output_filename = f"ANOVA_results_{STUDY_NAME}_omega_{safe_om}_last_epoch.txt"
    else:
        output_filename = f"ANOVA_results_{STUDY_NAME}_omega_{safe_om}_epoch_{epoch}.txt"
    output_path = os.path.join(figures_dir, output_filename)
    
    # Open output file
    output_file = open(output_path, 'w')
    
    def print_both(*args, **kwargs):
        """Print to both console and file"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_file)
    
    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print_both(f"No combined_logs.csv found in {raw_dir}")
        output_file.close()
        return
    
    # Read CSV with low_memory=False to avoid dtype warning
    combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
    print_both(f"Combined logs loaded from: {combined_logs_path}")
    print_both(f"Initial number of rows in combined logs: {len(combined_logs)}")
    
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
    
    print_both(f"After cleaning: {len(combined_logs)} rows remaining")
    
    # Filter for the specific omega and param_init_type
    subset_param = combined_logs[
        (combined_logs['param_init_type'] == param_type) & 
        (combined_logs['omega'] == omega_value)
    ]
    
    if subset_param.empty:
        print_both(f"No data found for param_type={param_type} and omega={omega_value}")
        output_file.close()
        return
    
    print_both(f"Found {len(subset_param)} rows for omega={omega_value}, param={param_type}")
    
    # Select the epoch for analysis
    if epoch == "last":
        selected_epoch = subset_param['epoch'].max()
    else:
        selected_epoch = epoch
    
    epoch_data = subset_param[subset_param['epoch'] == selected_epoch]
    
    if epoch_data.empty:
        print_both(f"No data found for epoch {selected_epoch}")
        output_file.close()
        return
    
    print_both(f"\nAnalyzing epoch: {selected_epoch}")
    print_both(f"Number of samples (runs x epsilons): {len(epoch_data)}")
    print_both(f"Note: Each sample represents 10 repetitions")
    
    # Extract error values (convert accuracy to error: 100 - accuracy)
    small_no_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_no_carry_small_accuracy"], errors='coerce').dropna()
    small_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_carry_small_accuracy"], errors='coerce').dropna()
    large_no_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_no_carry_large_accuracy"], errors='coerce').dropna()
    large_carry_errors = 100 - pd.to_numeric(epoch_data["test_pairs_carry_large_accuracy"], errors='coerce').dropna()
    
    # Actual sample sizes accounting for 10 repetitions per measurement
    n_small_no_carry = len(small_no_carry_errors) * n_repetitions_per_sample
    n_small_carry = len(small_carry_errors) * n_repetitions_per_sample
    n_large_no_carry = len(large_no_carry_errors) * n_repetitions_per_sample
    n_large_carry = len(large_carry_errors) * n_repetitions_per_sample
    
    print_both("\n" + "="*80)
    print_both("DESCRIPTIVE STATISTICS")
    print_both("="*80)
    
    print_both(f"\nSmall - No Carry: Mean = {small_no_carry_errors.mean():.2f}%, SD = {small_no_carry_errors.std():.2f}%, N = {len(small_no_carry_errors)} ({n_small_no_carry} total reps)")
    print_both(f"Small - Carry: Mean = {small_carry_errors.mean():.2f}%, SD = {small_carry_errors.std():.2f}%, N = {len(small_carry_errors)} ({n_small_carry} total reps)")
    print_both(f"Large - No Carry: Mean = {large_no_carry_errors.mean():.2f}%, SD = {large_no_carry_errors.std():.2f}%, N = {len(large_no_carry_errors)} ({n_large_no_carry} total reps)")
    print_both(f"Large - Carry: Mean = {large_carry_errors.mean():.2f}%, SD = {large_carry_errors.std():.2f}%, N = {len(large_carry_errors)} ({n_large_carry} total reps)")
    
    # ========================================
    # CARRY-OVER EFFECT ANOVA
    # ========================================
    print_both("\n" + "="*80)
    print_both("CARRY-OVER EFFECT ANOVA")
    print_both("="*80)
    print_both("Comparing problems WITH carry vs WITHOUT carry (collapsing across problem size)")
    
    # Combine small and large for no carry
    no_carry_all = pd.concat([small_no_carry_errors, large_no_carry_errors]).reset_index(drop=True)
    # Combine small and large for carry
    carry_all = pd.concat([small_carry_errors, large_carry_errors]).reset_index(drop=True)
    
    print_both(f"\nNo Carry (all sizes): Mean = {no_carry_all.mean():.2f}%, SD = {no_carry_all.std():.2f}%, N = {len(no_carry_all)} ({len(no_carry_all) * n_repetitions_per_sample} total reps)")
    print_both(f"Carry (all sizes): Mean = {carry_all.mean():.2f}%, SD = {carry_all.std():.2f}%, N = {len(carry_all)} ({len(carry_all) * n_repetitions_per_sample} total reps)")
    
    # Expand each observation by repeating it n_repetitions_per_sample times
    no_carry_expanded = np.repeat(no_carry_all.values, n_repetitions_per_sample)
    carry_expanded = np.repeat(carry_all.values, n_repetitions_per_sample)
    
    # One-way ANOVA for carry effect with expanded data
    f_stat_carry, p_value_carry = stats.f_oneway(no_carry_expanded, carry_expanded)
    
    print_both(f"\nOne-way ANOVA Results:")
    print_both(f"F-statistic: {f_stat_carry:.4f}")
    print_both(f"P-value: {p_value_carry:.6f}")
    
    if p_value_carry < 0.001:
        print_both("Significance: *** (p < 0.001)")
    elif p_value_carry < 0.01:
        print_both("Significance: ** (p < 0.01)")
    elif p_value_carry < 0.05:
        print_both("Significance: * (p < 0.05)")
    else:
        print_both("Significance: n.s. (not significant)")
    
    # Effect size (eta-squared)
    grand_mean = pd.concat([no_carry_all, carry_all]).mean()
    ss_between = len(no_carry_all) * (no_carry_all.mean() - grand_mean)**2 + len(carry_all) * (carry_all.mean() - grand_mean)**2
    ss_total = np.sum((pd.concat([no_carry_all, carry_all]) - grand_mean)**2)
    eta_squared_carry = ss_between / ss_total if ss_total > 0 else 0
    
    print_both(f"Effect size (η²): {eta_squared_carry:.4f}")
    
    # ========================================
    # PROBLEM SIZE EFFECT ANOVA
    # ========================================
    print_both("\n" + "="*80)
    print_both("PROBLEM SIZE EFFECT ANOVA")
    print_both("="*80)
    print_both("Comparing SMALL vs LARGE problems (collapsing across carry condition)")
    
    # Combine no carry and carry for small
    small_all = pd.concat([small_no_carry_errors, small_carry_errors]).reset_index(drop=True)
    # Combine no carry and carry for large
    large_all = pd.concat([large_no_carry_errors, large_carry_errors]).reset_index(drop=True)
    
    print_both(f"\nSmall (all carry conditions): Mean = {small_all.mean():.2f}%, SD = {small_all.std():.2f}%, N = {len(small_all)} ({len(small_all) * n_repetitions_per_sample} total reps)")
    print_both(f"Large (all carry conditions): Mean = {large_all.mean():.2f}%, SD = {large_all.std():.2f}%, N = {len(large_all)} ({len(large_all) * n_repetitions_per_sample} total reps)")
    
    # Expand each observation by repeating it n_repetitions_per_sample times
    small_expanded = np.repeat(small_all.values, n_repetitions_per_sample)
    large_expanded = np.repeat(large_all.values, n_repetitions_per_sample)
    
    # One-way ANOVA for problem size effect with expanded data
    f_stat_size, p_value_size = stats.f_oneway(small_expanded, large_expanded)
    
    print_both(f"\nOne-way ANOVA Results:")
    print_both(f"F-statistic: {f_stat_size:.4f}")
    print_both(f"P-value: {p_value_size:.6f}")
    
    if p_value_size < 0.001:
        print_both("Significance: *** (p < 0.001)")
    elif p_value_size < 0.01:
        print_both("Significance: ** (p < 0.01)")
    elif p_value_size < 0.05:
        print_both("Significance: * (p < 0.05)")
    else:
        print_both("Significance: n.s. (not significant)")
    
    # Effect size (eta-squared)
    grand_mean_size = pd.concat([small_all, large_all]).mean()
    ss_between_size = len(small_all) * (small_all.mean() - grand_mean_size)**2 + len(large_all) * (large_all.mean() - grand_mean_size)**2
    ss_total_size = np.sum((pd.concat([small_all, large_all]) - grand_mean_size)**2)
    eta_squared_size = ss_between_size / ss_total_size if ss_total_size > 0 else 0
    
    print_both(f"Effect size (η²): {eta_squared_size:.4f}")
    
    # ========================================
    # TWO-WAY ANOVA (Problem Size × Carry)
    # ========================================
    print_both("\n" + "="*80)
    print_both("TWO-WAY ANOVA (Problem Size × Carry Condition)")
    print_both("="*80)
    
    # Prepare data for two-way ANOVA
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
    
    # Perform two-way ANOVA using scipy
    # Group data by conditions
    groups = df_anova.groupby(['Size', 'Carry'])['Error'].apply(list)
    
    small_no_carry = np.array(groups[('Small', 'No Carry')])
    small_carry = np.array(groups[('Small', 'Carry')])
    large_no_carry = np.array(groups[('Large', 'No Carry')])
    large_carry = np.array(groups[('Large', 'Carry')])
    
    # Calculate grand mean
    all_data = np.concatenate([small_no_carry, small_carry, large_no_carry, large_carry])
    grand_mean_2way = np.mean(all_data)
    n_total = len(all_data)
    n_per_cell = len(small_no_carry)  # Equal n per cell (already expanded)
    
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
    
    print_both("\nMain Effect - Problem Size:")
    print_both(f"  F({df_size}, {df_within}) = {f_size_2way:.4f}, p = {p_size_2way:.6f}")
    if p_size_2way < 0.001:
        print_both("  Significance: *** (p < 0.001)")
    elif p_size_2way < 0.01:
        print_both("  Significance: ** (p < 0.01)")
    elif p_size_2way < 0.05:
        print_both("  Significance: * (p < 0.05)")
    else:
        print_both("  Significance: n.s. (not significant)")
    
    print_both("\nMain Effect - Carry Condition:")
    print_both(f"  F({df_carry}, {df_within}) = {f_carry_2way:.4f}, p = {p_carry_2way:.6f}")
    if p_carry_2way < 0.001:
        print_both("  Significance: *** (p < 0.001)")
    elif p_carry_2way < 0.01:
        print_both("  Significance: ** (p < 0.01)")
    elif p_carry_2way < 0.05:
        print_both("  Significance: * (p < 0.05)")
    else:
        print_both("  Significance: n.s. (not significant)")
    
    print_both("\nInteraction Effect (Size × Carry):")
    print_both(f"  F({df_interaction}, {df_within}) = {f_interaction:.4f}, p = {p_interaction:.6f}")
    if p_interaction < 0.001:
        print_both("  Significance: *** (p < 0.001)")
    elif p_interaction < 0.01:
        print_both("  Significance: ** (p < 0.01)")
    elif p_interaction < 0.05:
        print_both("  Significance: * (p < 0.05)")
    else:
        print_both("  Significance: n.s. (not significant)")
    
    print_both("\n" + "="*80)
    print_both("SUMMARY")
    print_both("="*80)
    print_both(f"\nConfiguration: {NUMBER_SIZE}-digit, {STUDY_NAME}, {PARAM_TYPE}, {MODEL_TYPE}")
    print_both(f"Omega: {omega_value}, Epoch: {selected_epoch}")
    print_both(f"Total samples per condition: {n_per_cell // n_repetitions_per_sample} observations ({n_per_cell} total after expansion)")
    print_both(f"Note: Each observation repeated {n_repetitions_per_sample} times for accurate ANOVA")
    print_both(f"\nCarry-over Effect: F = {f_stat_carry:.4f}, p = {p_value_carry:.6f}, η² = {eta_squared_carry:.4f}")
    print_both(f"Problem Size Effect: F = {f_stat_size:.4f}, p = {p_value_size:.6f}, η² = {eta_squared_size:.4f}")
    print_both(f"Interaction Effect: F = {f_interaction:.4f}, p = {p_interaction:.6f}")
    
    # Close output file
    output_file.close()
    print(f"\nResults saved to: {output_path}")

# Run analysis
perform_anova_analysis(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE, EPOCH, N_REPETITIONS_PER_SAMPLE)
