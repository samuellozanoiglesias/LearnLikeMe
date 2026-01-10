# USE: nohup python paper_anova_effects_comparison.py 2 STUDY1 STUDY2 RI argmax 0.10 > log_anova_effects_comparison.out 2>&1 &

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME_1 = str(sys.argv[2]).upper()  # Name of the first study (analyzed at batch 400)
STUDY_NAME_2 = str(sys.argv[3]).upper()  # Name of the second study (analyzed at batch 500)
PARAM_TYPE = str(sys.argv[4]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[5]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[6])  # Specific omega value to analyze
N_REPETITIONS_PER_SAMPLE = 1  # Number of repetitions averaged per accuracy value (e.g., 5)

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR_1 = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME_1}/{PARAM_TYPE}/{MODEL_TYPE}_version"
RAW_DIR_2 = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME_2}/{PARAM_TYPE}/{MODEL_TYPE}_version"
FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/comparison_{STUDY_NAME_1}_vs_{STUDY_NAME_2}"

def load_study_data(raw_dir, omega_value, param_type, fixed_batch):
    """
    Load and filter data for a specific study at a fixed batch.
    
    Args:
        raw_dir: Directory containing the combined_logs.csv
        omega_value: Omega value to filter for
        param_type: Parameter initialization type to filter for
        fixed_batch: The specific batch/epoch to extract data from
    
    Returns:
        Dictionary with error values for each condition, or None if no data found
    """
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
    
    # Filter for the specific batch
    epoch_data = subset_param[subset_param['epoch'] == fixed_batch]
    
    if epoch_data.empty:
        print(f"No data found for batch {fixed_batch}")
        return None
    
    print(f"Found {len(epoch_data)} samples for batch {fixed_batch}")
    
    # Extract error values (convert accuracy to error: 100 - accuracy)
    data = {}
    data['small_no_carry_errors'] = 100 - pd.to_numeric(epoch_data["test_pairs_no_carry_small_accuracy"], errors='coerce').dropna()
    data['small_carry_errors'] = 100 - pd.to_numeric(epoch_data["test_pairs_carry_small_accuracy"], errors='coerce').dropna()
    data['large_no_carry_errors'] = 100 - pd.to_numeric(epoch_data["test_pairs_no_carry_large_accuracy"], errors='coerce').dropna()
    data['large_carry_errors'] = 100 - pd.to_numeric(epoch_data["test_pairs_carry_large_accuracy"], errors='coerce').dropna()
    data['batch'] = fixed_batch
    
    return data

def perform_comparative_anova_analysis(data1, data2, study1_name, study2_name, figures_dir, omega_value, n_repetitions_per_sample=1):
    """
    Perform comparative ANOVA tests between two studies for carry-over effect and problem size effect.
    
    Args:
        data1: Data dictionary from first study (at batch 400)
        data2: Data dictionary from second study (at batch 500)
        study1_name: Name of first study
        study2_name: Name of second study
        figures_dir: Directory to save results
        omega_value: Omega value being analyzed
        n_repetitions_per_sample: Number of repetitions per observation
    """
    # Create output directory
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create output file
    safe_om = str(omega_value).replace('.', '_')
    output_filename = f"ANOVA_comparison_{study1_name}_{fixed_batch_bal}_vs_{study2_name}_{fixed_batch_exp}__omega_{safe_om}.txt"
    output_path = os.path.join(figures_dir, output_filename)
    
    # Open output file
    output_file = open(output_path, 'w')
    
    def print_both(*args, **kwargs):
        """Print to both console and file"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_file)
    
    print_both("="*100)
    print_both("COMPARATIVE ANOVA ANALYSIS BETWEEN TWO STUDIES")
    print_both("="*100)
    print_both(f"Study 1: {study1_name} (Batch {data1['batch']})")
    print_both(f"Study 2: {study2_name} (Batch {data2['batch']})")
    print_both(f"Omega: {omega_value}")
    print_both(f"Parameter Type: {PARAM_TYPE}")
    print_both(f"Model Type: {MODEL_TYPE}")
    print_both(f"Note: Each observation represents {n_repetitions_per_sample} repetitions")
    
    # ========================================
    # DESCRIPTIVE STATISTICS
    # ========================================
    print_both("\n" + "="*80)
    print_both("DESCRIPTIVE STATISTICS")
    print_both("="*80)
    
    print_both(f"\n{study1_name} (Batch {data1['batch']}):")
    print_both(f"  Small - No Carry: Mean = {data1['small_no_carry_errors'].mean():.2f}%, SD = {data1['small_no_carry_errors'].std():.2f}%, N = {len(data1['small_no_carry_errors'])}")
    print_both(f"  Small - Carry: Mean = {data1['small_carry_errors'].mean():.2f}%, SD = {data1['small_carry_errors'].std():.2f}%, N = {len(data1['small_carry_errors'])}")
    print_both(f"  Large - No Carry: Mean = {data1['large_no_carry_errors'].mean():.2f}%, SD = {data1['large_no_carry_errors'].std():.2f}%, N = {len(data1['large_no_carry_errors'])}")
    print_both(f"  Large - Carry: Mean = {data1['large_carry_errors'].mean():.2f}%, SD = {data1['large_carry_errors'].std():.2f}%, N = {len(data1['large_carry_errors'])}")
    
    print_both(f"\n{study2_name} (Batch {data2['batch']}):")
    print_both(f"  Small - No Carry: Mean = {data2['small_no_carry_errors'].mean():.2f}%, SD = {data2['small_no_carry_errors'].std():.2f}%, N = {len(data2['small_no_carry_errors'])}")
    print_both(f"  Small - Carry: Mean = {data2['small_carry_errors'].mean():.2f}%, SD = {data2['small_carry_errors'].std():.2f}%, N = {len(data2['small_carry_errors'])}")
    print_both(f"  Large - No Carry: Mean = {data2['large_no_carry_errors'].mean():.2f}%, SD = {data2['large_no_carry_errors'].std():.2f}%, N = {len(data2['large_no_carry_errors'])}")
    print_both(f"  Large - Carry: Mean = {data2['large_carry_errors'].mean():.2f}%, SD = {data2['large_carry_errors'].std():.2f}%, N = {len(data2['large_carry_errors'])}")
    
    # ========================================
    # CARRY-OVER EFFECT COMPARISON
    # ========================================
    print_both("\n" + "="*80)
    print_both("CARRY-OVER EFFECT COMPARISON")
    print_both("="*80)
    print_both("Comparing carry-over effects between studies")
    print_both("(Difference between carry and no-carry conditions)")
    
    # Calculate carry-over effects for each study
    # Study 1
    study1_no_carry = pd.concat([data1['small_no_carry_errors'], data1['large_no_carry_errors']]).reset_index(drop=True)
    study1_carry = pd.concat([data1['small_carry_errors'], data1['large_carry_errors']]).reset_index(drop=True)
    study1_carry_effect = study1_carry.mean() - study1_no_carry.mean()
    
    # Study 2  
    study2_no_carry = pd.concat([data2['small_no_carry_errors'], data2['large_no_carry_errors']]).reset_index(drop=True)
    study2_carry = pd.concat([data2['small_carry_errors'], data2['large_carry_errors']]).reset_index(drop=True)
    study2_carry_effect = study2_carry.mean() - study2_no_carry.mean()
    
    print_both(f"\n{study1_name} carry-over effect: {study1_carry_effect:.2f}% (Carry - No Carry)")
    print_both(f"{study2_name} carry-over effect: {study2_carry_effect:.2f}% (Carry - No Carry)")
    print_both(f"Difference in carry-over effects: {abs(study2_carry_effect - study1_carry_effect):.2f}%")
    
    # Expand data for proper ANOVA (accounting for repetitions)
    study1_no_carry_expanded = np.repeat(study1_no_carry.values, n_repetitions_per_sample)
    study1_carry_expanded = np.repeat(study1_carry.values, n_repetitions_per_sample)
    study2_no_carry_expanded = np.repeat(study2_no_carry.values, n_repetitions_per_sample)
    study2_carry_expanded = np.repeat(study2_carry.values, n_repetitions_per_sample)
    
    # Three-way ANOVA: Study × Carry condition × Problem size
    # Prepare data frame
    data_list = []
    
    # Study 1 data
    for val in data1['small_no_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study1_name, 'Carry': 'No Carry', 'Size': 'Small'})
    for val in data1['small_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study1_name, 'Carry': 'Carry', 'Size': 'Small'})
    for val in data1['large_no_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study1_name, 'Carry': 'No Carry', 'Size': 'Large'})
    for val in data1['large_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study1_name, 'Carry': 'Carry', 'Size': 'Large'})
    
    # Study 2 data
    for val in data2['small_no_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study2_name, 'Carry': 'No Carry', 'Size': 'Small'})
    for val in data2['small_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study2_name, 'Carry': 'Carry', 'Size': 'Small'})
    for val in data2['large_no_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study2_name, 'Carry': 'No Carry', 'Size': 'Large'})
    for val in data2['large_carry_errors']:
        for _ in range(n_repetitions_per_sample):
            data_list.append({'Error': val, 'Study': study2_name, 'Carry': 'Carry', 'Size': 'Large'})
    
    df_anova = pd.DataFrame(data_list)
    
    # Compare carry-over effects between studies using independent t-test
    carry_effect_1 = study1_carry_expanded - study1_no_carry_expanded[:len(study1_carry_expanded)]
    carry_effect_2 = study2_carry_expanded - study2_no_carry_expanded[:len(study2_carry_expanded)]
    
    t_stat_carry, p_value_carry = stats.ttest_ind(carry_effect_1, carry_effect_2)
    
    print_both(f"\nIndependent t-test for carry-over effect difference:")
    print_both(f"t-statistic: {t_stat_carry:.4f}")
    print_both(f"P-value: {p_value_carry:.6f}")
    
    if p_value_carry < 0.001:
        print_both("Significance: *** (p < 0.001)")
    elif p_value_carry < 0.01:
        print_both("Significance: ** (p < 0.01)")
    elif p_value_carry < 0.05:
        print_both("Significance: * (p < 0.05)")
    else:
        print_both("Significance: n.s. (not significant)")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(carry_effect_1, ddof=1)**2 + np.std(carry_effect_2, ddof=1)**2) / 2)
    cohens_d_carry = (np.mean(carry_effect_1) - np.mean(carry_effect_2)) / pooled_std
    print_both(f"Effect size (Cohen's d): {cohens_d_carry:.4f}")
    
    # ========================================
    # PROBLEM SIZE EFFECT COMPARISON
    # ========================================
    print_both("\n" + "="*80)
    print_both("PROBLEM SIZE EFFECT COMPARISON")
    print_both("="*80)
    print_both("Comparing problem size effects between studies")
    print_both("(Difference between large and small problems)")
    
    # Calculate problem size effects for each study
    # Study 1
    study1_small = pd.concat([data1['small_no_carry_errors'], data1['small_carry_errors']]).reset_index(drop=True)
    study1_large = pd.concat([data1['large_no_carry_errors'], data1['large_carry_errors']]).reset_index(drop=True)
    study1_size_effect = study1_large.mean() - study1_small.mean()
    
    # Study 2
    study2_small = pd.concat([data2['small_no_carry_errors'], data2['small_carry_errors']]).reset_index(drop=True)
    study2_large = pd.concat([data2['large_no_carry_errors'], data2['large_carry_errors']]).reset_index(drop=True)
    study2_size_effect = study2_large.mean() - study2_small.mean()
    
    print_both(f"\n{study1_name} problem size effect: {study1_size_effect:.2f}% (Large - Small)")
    print_both(f"{study2_name} problem size effect: {study2_size_effect:.2f}% (Large - Small)")
    print_both(f"Difference in problem size effects: {abs(study2_size_effect - study1_size_effect):.2f}%")
    
    # Expand data for proper comparison
    study1_small_expanded = np.repeat(study1_small.values, n_repetitions_per_sample)
    study1_large_expanded = np.repeat(study1_large.values, n_repetitions_per_sample)
    study2_small_expanded = np.repeat(study2_small.values, n_repetitions_per_sample)
    study2_large_expanded = np.repeat(study2_large.values, n_repetitions_per_sample)
    
    # Compare problem size effects between studies using independent t-test
    size_effect_1 = study1_large_expanded - study1_small_expanded[:len(study1_large_expanded)]
    size_effect_2 = study2_large_expanded - study2_small_expanded[:len(study2_large_expanded)]
    
    t_stat_size, p_value_size = stats.ttest_ind(size_effect_1, size_effect_2)
    
    print_both(f"\nIndependent t-test for problem size effect difference:")
    print_both(f"t-statistic: {t_stat_size:.4f}")
    print_both(f"P-value: {p_value_size:.6f}")
    
    if p_value_size < 0.001:
        print_both("Significance: *** (p < 0.001)")
    elif p_value_size < 0.01:
        print_both("Significance: ** (p < 0.01)")
    elif p_value_size < 0.05:
        print_both("Significance: * (p < 0.05)")
    else:
        print_both("Significance: n.s. (not significant)")
    
    # Effect size (Cohen's d)
    pooled_std_size = np.sqrt((np.std(size_effect_1, ddof=1)**2 + np.std(size_effect_2, ddof=1)**2) / 2)
    cohens_d_size = (np.mean(size_effect_1) - np.mean(size_effect_2)) / pooled_std_size
    print_both(f"Effect size (Cohen's d): {cohens_d_size:.4f}")
    
    # ========================================
    # THREE-WAY ANOVA COMPARISON
    # ========================================
    print_both("\n" + "="*80)
    print_both("THREE-WAY ANOVA: STUDY × CARRY × PROBLEM SIZE")
    print_both("="*80)
    
    # Group data by conditions
    groups = df_anova.groupby(['Study', 'Carry', 'Size'])['Error'].apply(list)
    
    # Get all data arrays
    s1_small_no_carry = np.array(groups[(study1_name, 'No Carry', 'Small')])
    s1_small_carry = np.array(groups[(study1_name, 'Carry', 'Small')])
    s1_large_no_carry = np.array(groups[(study1_name, 'No Carry', 'Large')])
    s1_large_carry = np.array(groups[(study1_name, 'Carry', 'Large')])
    s2_small_no_carry = np.array(groups[(study2_name, 'No Carry', 'Small')])
    s2_small_carry = np.array(groups[(study2_name, 'Carry', 'Small')])
    s2_large_no_carry = np.array(groups[(study2_name, 'No Carry', 'Large')])
    s2_large_carry = np.array(groups[(study2_name, 'Carry', 'Large')])
    
    # Simplified ANOVA using scipy for main effects
    # Study effect
    study1_all = np.concatenate([s1_small_no_carry, s1_small_carry, s1_large_no_carry, s1_large_carry])
    study2_all = np.concatenate([s2_small_no_carry, s2_small_carry, s2_large_no_carry, s2_large_carry])
    f_study, p_study = stats.f_oneway(study1_all, study2_all)
    
    print_both(f"\nMain Effect - Study:")
    print_both(f"  F-statistic: {f_study:.4f}")
    print_both(f"  P-value: {p_study:.6f}")
    if p_study < 0.001:
        print_both("  Significance: *** (p < 0.001)")
    elif p_study < 0.01:
        print_both("  Significance: ** (p < 0.01)")
    elif p_study < 0.05:
        print_both("  Significance: * (p < 0.05)")
    else:
        print_both("  Significance: n.s. (not significant)")
    
    # Study × Carry interaction
    study1_carry_effect_all = np.concatenate([s1_small_carry, s1_large_carry]) - np.concatenate([s1_small_no_carry, s1_large_no_carry])
    study2_carry_effect_all = np.concatenate([s2_small_carry, s2_large_carry]) - np.concatenate([s2_small_no_carry, s2_large_no_carry])
    f_study_carry, p_study_carry = stats.f_oneway(study1_carry_effect_all, study2_carry_effect_all)
    
    print_both(f"\nInteraction - Study × Carry:")
    print_both(f"  F-statistic: {f_study_carry:.4f}")
    print_both(f"  P-value: {p_study_carry:.6f}")
    if p_study_carry < 0.001:
        print_both("  Significance: *** (p < 0.001)")
    elif p_study_carry < 0.01:
        print_both("  Significance: ** (p < 0.01)")
    elif p_study_carry < 0.05:
        print_both("  Significance: * (p < 0.05)")
    else:
        print_both("  Significance: n.s. (not significant)")
    
    # ========================================
    # SUMMARY
    # ========================================
    print_both("\n" + "="*80)
    print_both("SUMMARY")
    print_both("="*80)
    print_both(f"\nComparison: {study1_name} (Batch {data1['batch']}) vs {study2_name} (Batch {data2['batch']})")
    print_both(f"Configuration: {NUMBER_SIZE}-digit, {PARAM_TYPE}, {MODEL_TYPE}, Omega {omega_value}")
    print_both(f"\nCarry-over Effect Comparison:")
    print_both(f"  {study1_name}: {study1_carry_effect:.2f}%")
    print_both(f"  {study2_name}: {study2_carry_effect:.2f}%") 
    print_both(f"  Difference: {abs(study2_carry_effect - study1_carry_effect):.2f}%")
    print_both(f"  Statistical test: t = {t_stat_carry:.4f}, p = {p_value_carry:.6f}, d = {cohens_d_carry:.4f}")
    
    print_both(f"\nProblem Size Effect Comparison:")
    print_both(f"  {study1_name}: {study1_size_effect:.2f}%")
    print_both(f"  {study2_name}: {study2_size_effect:.2f}%")
    print_both(f"  Difference: {abs(study2_size_effect - study1_size_effect):.2f}%")
    print_both(f"  Statistical test: t = {t_stat_size:.4f}, p = {p_value_size:.6f}, d = {cohens_d_size:.4f}")
    
    print_both(f"\nOverall Study Difference: F = {f_study:.4f}, p = {p_study:.6f}")
    
    # Close output file
    output_file.close()
    print(f"\nResults saved to: {output_path}")

fixed_batch_bal = 450  # Fixed batch for the first study
fixed_batch_exp = 600  # Fixed batch for the second study

# Main execution
print(f"\n{'='*60}")
print(f"Loading data for {STUDY_NAME_1} at batch {fixed_batch_bal}")
print(f"{'='*60}")
data1 = load_study_data(RAW_DIR_1, OMEGA_VALUE, PARAM_TYPE, fixed_batch_bal)

print(f"\n{'='*60}")
print(f"Loading data for {STUDY_NAME_2} at batch {fixed_batch_exp}")
print(f"{'='*60}")
data2 = load_study_data(RAW_DIR_2, OMEGA_VALUE, PARAM_TYPE, fixed_batch_exp)

# Perform comparative analysis
if data1 is not None and data2 is not None:
    perform_comparative_anova_analysis(data1, data2, STUDY_NAME_1, STUDY_NAME_2, 
                                     FIGURES_DIR, OMEGA_VALUE, N_REPETITIONS_PER_SAMPLE)
else:
    print("Unable to perform analysis - missing data for one or both studies.")