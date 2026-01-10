# USE: nohup python paper_figure_extractors_and_decision.py 2 STUDY WI argmax > logs_paper_figs_extractor.out 2>&1 &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"
RAW_DIR_CARRY = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/carry_extractor/{STUDY_NAME}"
RAW_DIR_UNIT = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/unit_extractor/{STUDY_NAME}"
RAW_DIR_DECISION = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"

def analyze_single_digit_modules(raw_dir_carry, raw_dir_unit, raw_dir_decision, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    run_dirs_carry = [
        os.path.join(raw_dir_carry, d) for d in os.listdir(raw_dir_carry)
        if os.path.isdir(os.path.join(raw_dir_carry, d)) and "Training_" in d
    ]
    run_dirs_unit = [
        os.path.join(raw_dir_unit, d) for d in os.listdir(raw_dir_unit)
        if os.path.isdir(os.path.join(raw_dir_unit, d)) and "Training_" in d
    ]
    
    print(f"Found {len(run_dirs_carry)} carry extractor runs")
    print(f"Found {len(run_dirs_unit)} unit extractor runs")

    all_logs_carry = []
    all_logs_unit = []
    
    # Load decision module data
    combined_logs_path = os.path.join(raw_dir_decision, "combined_logs.csv")
    decision_data = None
    decision_data_detailed = None
    if os.path.exists(combined_logs_path):
        combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
        combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
        combined_logs['accuracy'] = pd.to_numeric(combined_logs['accuracy'], errors='coerce')
        combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')
        
        valid_combined_logs = combined_logs[combined_logs['epoch'].notna() & combined_logs['accuracy'].notna()]
        if not valid_combined_logs.empty:
            last_epoch_err_decision = (
                valid_combined_logs.groupby(["omega", "param_init_type", "epsilon"])[["epoch", "accuracy"]]
                .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"] if not g.empty else np.nan)
                .reset_index(name="accuracy")
            )
            last_epoch_err_decision['error'] = 100 - last_epoch_err_decision['accuracy'] * 100
            last_epoch_err_decision = last_epoch_err_decision[last_epoch_err_decision['accuracy'].notna()]
            last_epoch_err_decision = last_epoch_err_decision[last_epoch_err_decision['epsilon'] != 0].copy()
            
            # Average over epsilon values to get error per omega (for main line)
            decision_data = last_epoch_err_decision.groupby('omega')['error'].mean().reset_index()
            # Keep detailed data for shadow band (std dev across epsilon)
            decision_data_detailed = last_epoch_err_decision.copy()
            print(f"Decision module data loaded from: {combined_logs_path}")
    else:
        print(f"No combined_logs.csv found in {raw_dir_decision}")

    for run in run_dirs_carry:
        log_path = os.path.join(run, "training_log.csv")
        config_path = os.path.join(run, "config.txt")
        if not os.path.exists(log_path):
            print(f"No training_log.csv were found in {run}")
            continue

        omega = None
        epsilon = 1.0
        with open(config_path, "r") as f:
            for line in f:
                if "Weber fraction:" in line:
                    omega = float(line.strip().split(":")[-1])
                elif "Noise Factor for Initialization Parameters (Epsilon):" in line:
                    epsilon_str = line.strip().split(":")[-1].strip()
                    if epsilon_str.lower() == "none":
                        epsilon = "none"
                    else:
                        epsilon = float(epsilon_str)

        # --- Read training_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega"] = omega
        log_df["epsilon"] = epsilon
        all_logs_carry.append(log_df)

    for run in run_dirs_unit:
        log_path = os.path.join(run, "training_log.csv")
        config_path = os.path.join(run, "config.txt")

        if not os.path.exists(log_path):
            print(f"No training_log.csv were found in {run}")
            continue

        omega = None
        with open(config_path, "r") as f:
            for line in f:
                if "Weber fraction:" in line:
                    omega = float(line.strip().split(":")[-1])
                elif "Noise Factor for Initialization Parameters (Epsilon):" in line:
                    epsilon_str = line.strip().split(":")[-1].strip()
                    if epsilon_str.lower() == "none":
                        epsilon = "none"
                    else:
                        epsilon = float(epsilon_str)

        # --- Read training_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega"] = omega
        log_df["epsilon"] = epsilon
        all_logs_unit.append(log_df)

    print(f"Loaded {len(all_logs_carry)} carry logs")
    print(f"Loaded {len(all_logs_unit)} unit logs")
    
    if all_logs_carry and all_logs_unit:
        combined_logs_carry = pd.concat(all_logs_carry, ignore_index=True)
        combined_logs_unit = pd.concat(all_logs_unit, ignore_index=True)
        
        # Initialize empty dataframes to avoid NameError
        last_epoch_err_carry = pd.DataFrame()
        last_epoch_err_unit = pd.DataFrame()
        
        # Initialize empty dataframes to avoid NameError
        last_epoch_err_carry = pd.DataFrame()
        last_epoch_err_unit = pd.DataFrame()

        # Check if epsilon is "none" for extractors
        epsilon_is_none_carry = False
        epsilon_is_none_unit = False
        if not combined_logs_carry.empty:
            first_epsilon_carry = combined_logs_carry['epsilon'].iloc[0]
            if isinstance(first_epsilon_carry, str) and first_epsilon_carry.lower() == "none":
                epsilon_is_none_carry = True
        if not combined_logs_unit.empty:
            first_epsilon_unit = combined_logs_unit['epsilon'].iloc[0]
            if isinstance(first_epsilon_unit, str) and first_epsilon_unit.lower() == "none":
                epsilon_is_none_unit = True

        # --- Plot 3: Accuracy last epoch vs. omega ---
        last_epoch_carry_results = []
        if epsilon_is_none_carry:
            # Group only by omega when epsilon is "none"
            for omega_val, group in combined_logs_carry.groupby('omega'):
                epsilon_val = "none"
                # Get the last epoch for this omega
                last_epoch_num = group['epoch'].max()
                last_epoch_data = group[group['epoch'] == last_epoch_num]
                
                if not last_epoch_data.empty:
                    # Get accuracy from the last epoch
                    accuracy = last_epoch_data['accuracy'].iloc[0]
                    last_epoch_carry_results.append({
                        'omega': omega_val,
                        'epsilon': epsilon_val,
                        'last_epoch': last_epoch_num,
                        'accuracy': accuracy
                    })
        else:
            for (omega_val, epsilon_val), group in combined_logs_carry.groupby(['omega', 'epsilon']):
                # Get the last epoch for this omega-epsilon pair
                last_epoch_num = group['epoch'].max()
                last_epoch_data = group[group['epoch'] == last_epoch_num]
                
                if not last_epoch_data.empty:
                    # Get accuracy from the last epoch
                    accuracy = last_epoch_data['accuracy'].iloc[0]
                    last_epoch_carry_results.append({
                        'omega': omega_val,
                        'epsilon': epsilon_val,
                        'last_epoch': last_epoch_num,
                        'accuracy': accuracy
                    })
        
        last_epoch_carry_df = pd.DataFrame(last_epoch_carry_results)
        print(f"Carry DataFrame shape: {last_epoch_carry_df.shape}")
        if not last_epoch_carry_df.empty:
            print(f"Carry omega values: {last_epoch_carry_df['omega'].tolist()}")
            print(f"Carry omega unique: {last_epoch_carry_df['omega'].unique()}")
            # Ensure omega is numeric
            last_epoch_carry_df['omega'] = pd.to_numeric(last_epoch_carry_df['omega'], errors='coerce')
            print(f"Carry omega after conversion: {last_epoch_carry_df['omega'].tolist()}")
            print(f"Carry omega NaN count: {last_epoch_carry_df['omega'].isna().sum()}")
            # Sort by omega only, handle epsilon separately
            if epsilon_is_none_carry:
                last_epoch_carry_df = last_epoch_carry_df.sort_values(['omega'])
            else:
                last_epoch_carry_df = last_epoch_carry_df.sort_values(['omega', 'epsilon'])
            last_epoch_carry_df["error"] = 100 - last_epoch_carry_df["accuracy"] * 100
            print(f"Carry DataFrame before groupby: {len(last_epoch_carry_df)} rows")
            
            # When epsilon is "none", use values directly without averaging
            if epsilon_is_none_carry:
                last_epoch_err_carry = last_epoch_carry_df[['omega', 'error']].copy()
            else:
                last_epoch_err_carry = last_epoch_carry_df.groupby("omega")["error"].mean().reset_index()
            print(f"Carry DataFrame after processing: {len(last_epoch_err_carry)} rows")

        last_epoch_unit_results = []
        if epsilon_is_none_unit:
            # Group only by omega when epsilon is "none"
            for omega_val, group in combined_logs_unit.groupby('omega'):
                epsilon_val = "none"
                # Get the last epoch for this omega
                last_epoch_num = group['epoch'].max()
                last_epoch_data = group[group['epoch'] == last_epoch_num]
                
                if not last_epoch_data.empty:
                    # Get accuracy from the last epoch
                    accuracy = last_epoch_data['accuracy'].iloc[0]
                    last_epoch_unit_results.append({
                        'omega': omega_val,
                        'epsilon': epsilon_val,
                        'last_epoch': last_epoch_num,
                        'accuracy': accuracy
                    })
        else:
            for (omega_val, epsilon_val), group in combined_logs_unit.groupby(['omega', 'epsilon']):
                # Get the last epoch for this omega-epsilon pair
                last_epoch_num = group['epoch'].max()
                last_epoch_data = group[group['epoch'] == last_epoch_num]
                
                if not last_epoch_data.empty:
                    # Get accuracy from the last epoch
                    accuracy = last_epoch_data['accuracy'].iloc[0]
                    last_epoch_unit_results.append({
                        'omega': omega_val,
                        'epsilon': epsilon_val,
                        'last_epoch': last_epoch_num,
                        'accuracy': accuracy
                    })
        
        last_epoch_unit_df = pd.DataFrame(last_epoch_unit_results)
        print(f"Unit DataFrame shape: {last_epoch_unit_df.shape}")
        if not last_epoch_unit_df.empty:
            print(f"Unit omega values: {last_epoch_unit_df['omega'].tolist()}")
            print(f"Unit omega unique: {last_epoch_unit_df['omega'].unique()}")
            # Ensure omega is numeric
            last_epoch_unit_df['omega'] = pd.to_numeric(last_epoch_unit_df['omega'], errors='coerce')
            print(f"Unit omega after conversion: {last_epoch_unit_df['omega'].tolist()}")
            print(f"Unit omega NaN count: {last_epoch_unit_df['omega'].isna().sum()}")
            # Sort by omega only, handle epsilon separately
            if epsilon_is_none_unit:
                last_epoch_unit_df = last_epoch_unit_df.sort_values(['omega'])
            else:
                last_epoch_unit_df = last_epoch_unit_df.sort_values(['omega', 'epsilon'])
            last_epoch_unit_df["error"] = 100 - last_epoch_unit_df["accuracy"] * 100
            print(f"Unit DataFrame before processing: {len(last_epoch_unit_df)} rows")
            
            # When epsilon is "none", use values directly without averaging
            if epsilon_is_none_unit:
                last_epoch_err_unit = last_epoch_unit_df[['omega', 'error']].copy()
            else:
                last_epoch_err_unit = last_epoch_unit_df.groupby("omega")["error"].mean().reset_index()
            print(f"Unit DataFrame after processing: {len(last_epoch_err_unit)} rows")
        
        print(f"Carry results: {len(last_epoch_carry_results)} rows")
        print(f"Unit results: {len(last_epoch_unit_results)} rows")
        print(f"Final carry data: {len(last_epoch_err_carry) if 'last_epoch_err_carry' in locals() else 0} rows")
        print(f"Final unit data: {len(last_epoch_err_unit) if 'last_epoch_err_unit' in locals() else 0} rows")

        # --- Original figure: Only extractors ---
        # Compute stats for shadow bands (assuming multiple runs per omega represent variation)
        # Skip shadow bands if epsilon is "none" for extractors
        if not epsilon_is_none_carry and not last_epoch_carry_df.empty:
            carry_stats = last_epoch_carry_df.groupby('omega')['error'].agg(['mean', 'std']).reset_index()
        if not epsilon_is_none_unit and not last_epoch_unit_df.empty:
            unit_stats = last_epoch_unit_df.groupby('omega')['error'].agg(['mean', 'std']).reset_index()
        
        # Only plot if we have data
        carry_empty = 'last_epoch_err_carry' not in locals() or last_epoch_err_carry.empty
        unit_empty = 'last_epoch_err_unit' not in locals() or last_epoch_err_unit.empty
        print(f"Carry data empty: {carry_empty}")
        print(f"Unit data empty: {unit_empty}")
        
        if not carry_empty and not unit_empty:
            plt.figure(figsize=(12, 7))
            # Plot shadow bands first (so they appear behind the lines) - only if epsilon is not "none"
            if not epsilon_is_none_carry and not last_epoch_carry_df.empty:
                plt.fill_between(carry_stats['omega'], 
                                 carry_stats['mean'] - carry_stats['std'], 
                                 carry_stats['mean'] + carry_stats['std'], 
                                 color="#FF8C2A", alpha=0.2)
            if not epsilon_is_none_unit and not last_epoch_unit_df.empty:
                plt.fill_between(unit_stats['omega'], 
                                 unit_stats['mean'] - unit_stats['std'], 
                                 unit_stats['mean'] + unit_stats['std'], 
                                 color="#825630", alpha=0.2)
            # Plot main lines
            sns.lineplot(x="omega", y="error", marker="o", data=last_epoch_err_carry, label="Carry Extractor", color="#FF8C2A", linestyle="-", linewidth=3, markersize=10)
            sns.lineplot(x="omega", y="error", marker="o", data=last_epoch_err_unit, label="Unit Extractor", color="#825630", linestyle="-", linewidth=3, markersize=10)
            plt.xlabel("Magnitude Noise (ω)", fontsize=32)
            plt.legend(fontsize=30, loc='upper left')
            plt.ylabel("Mean Error Rate (%)", fontsize=32)
            plt.ylim(-5, 45)
            plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            fname = os.path.join(figures_dir, "extractors-error_last_epoch_vs_omega.png")
            plt.savefig(fname, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved extractors figure: {fname}")
        
            # --- New figure: All modules including decision ---
            if decision_data is not None and not decision_data.empty:
                # For decision module, compute std across epsilon values
                if decision_data_detailed is not None:
                    decision_stats = decision_data_detailed.groupby('omega')['error'].agg(['mean', 'std']).reset_index()
                else:
                    decision_stats = decision_data.copy()
                    decision_stats['mean'] = decision_stats['error']
                    decision_stats['std'] = 0
                
                print(decision_stats)

                plt.figure(figsize=(12, 7))
                # Plot shadow bands first (so they appear behind the lines) - only if epsilon is not "none" for extractors
                if not epsilon_is_none_carry and not last_epoch_carry_df.empty and 'carry_stats' in locals():
                    plt.fill_between(carry_stats['omega'], 
                                     carry_stats['mean'] - carry_stats['std'], 
                                     carry_stats['mean'] + carry_stats['std'], 
                                     color="#FF8C2A", alpha=0.2)
                if not epsilon_is_none_unit and not last_epoch_unit_df.empty and 'unit_stats' in locals():
                    plt.fill_between(unit_stats['omega'], 
                                     unit_stats['mean'] - unit_stats['std'], 
                                     unit_stats['mean'] + unit_stats['std'], 
                                     color="#825630", alpha=0.2)
                plt.fill_between(decision_stats['omega'], 
                                 decision_stats['mean'] - decision_stats['std'], 
                                 decision_stats['mean'] + decision_stats['std'], 
                                 color="#7F6BB3", alpha=0.2)
                # Plot main lines
                sns.lineplot(x="omega", y="error", marker="o", data=last_epoch_err_carry, label="Carry Extractor", color="#FF8C2A", linestyle="-", linewidth=3, markersize=10)
                sns.lineplot(x="omega", y="error", marker="o", data=last_epoch_err_unit, label="Unit Extractor", color="#825630", linestyle="-", linewidth=3, markersize=10)
                sns.lineplot(x="omega", y="error", marker="o", data=decision_data, label="Integrated Model", color="#7F6BB3", linestyle="-", linewidth=3, markersize=10)
                plt.xlabel("Magnitude Noise (ω)", fontsize=32)
                plt.legend(fontsize=30, loc='upper left')
                plt.ylabel("Mean Error Rate (%)", fontsize=32)
                plt.ylim(-5, 105)
                plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                fname = os.path.join(figures_dir, "all_modules-error_last_epoch_vs_omega.png")
                plt.savefig(fname, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Saved combined figure: {fname}")

    print(f"Saved figures in: {figures_dir}")

analyze_single_digit_modules(RAW_DIR_CARRY, RAW_DIR_UNIT, RAW_DIR_DECISION, FIGURES_DIR)