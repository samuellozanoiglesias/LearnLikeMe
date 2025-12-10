# USE: nohup python analyze_training_extractor_modules.py unit_extractor STUDY > logs_analyze_training_extractor.out &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
MODULE_NAME = str(sys.argv[1]).lower()  # unit_extractor or carry_extractor
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}/{STUDY_NAME}"

def analyze_single_digit_modules(raw_dir):
    figures_dir = os.path.join(raw_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    run_dirs = [
        os.path.join(raw_dir, d) for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and "Training_" in d
    ]

    all_results = []
    all_logs = []

    for run in run_dirs:
        results_path = os.path.join(run, "training_results.csv")
        log_path = os.path.join(run, "training_log.csv")
        config_path = os.path.join(run, "config.txt")

        if not os.path.exists(results_path):
            print(f"No training_results.csv were found in {run}")
            continue
        if not os.path.exists(log_path):
            print(f"No training_log.csv were found in {run}")
            continue

        df = pd.read_csv(results_path)

        omega = None
        epsilon = None
        with open(config_path, "r") as f:
            for line in f:
                if "Weber fraction:" in line:
                    omega = float(line.strip().split(":")[-1])
                if "Noise Factor for Initialization Parameters (Epsilon):" in line:
                    epsilon = float(line.strip().split(":")[-1])

        df["omega"] = omega
        df["epsilon"] = epsilon
        df["error"] = df["y (real)"] != df["pred"]
        df["error_distance"] = np.abs(df["y (real)"] - df["pred"])
        all_results.append(df)

        # --- Read training_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega"] = omega
        log_df["epsilon"] = epsilon
        all_logs.append(log_df)

    # --- Aggregated analysis ---
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results to CSV
        combined_results_path = os.path.join(figures_dir, "combined_training_results.csv")
        combined_df.to_csv(combined_results_path, index=False)
        print(f"Saved combined training results to: {combined_results_path}")

    if all_logs:
        combined_logs = pd.concat(all_logs, ignore_index=True)
        
        # Save combined logs to CSV
        combined_logs_path = os.path.join(figures_dir, "combined_training_logs.csv")
        combined_logs.to_csv(combined_logs_path, index=False)
        print(f"Saved combined training logs to: {combined_logs_path}")

        # --- Save CSV: Last epoch accuracy for each omega-epsilon pair ---
        last_epoch_results = []
        for (omega_val, epsilon_val), group in combined_logs.groupby(['omega', 'epsilon']):
            # Get the last epoch for this omega-epsilon pair
            last_epoch_num = group['epoch'].max()
            last_epoch_data = group[group['epoch'] == last_epoch_num]
            
            if not last_epoch_data.empty:
                # Get accuracy from the last epoch
                accuracy = last_epoch_data['accuracy'].iloc[0]
                last_epoch_results.append({
                    'omega': omega_val,
                    'epsilon': epsilon_val,
                    'last_epoch': last_epoch_num,
                    'accuracy': accuracy
                })
        
        if last_epoch_results:
            last_epoch_df = pd.DataFrame(last_epoch_results)
            last_epoch_df = last_epoch_df.sort_values(['omega', 'epsilon'])
            csv_path = os.path.join(figures_dir, "last_epoch_accuracy_per_omega_epsilon.csv")
            last_epoch_df.to_csv(csv_path, index=False)
            print(f"Saved last epoch accuracy CSV to: {csv_path}")
            print(f"Total omega-epsilon pairs: {len(last_epoch_results)}")

        # --- Plot 3: Accuracy last epoch vs. omega ---
        last_epoch_acc = (
            last_epoch_df.groupby("omega")["accuracy"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(6, 4))
        sns.lineplot(x="omega", y="accuracy", marker="o", data=last_epoch_acc)
        plt.title("Last Epoch Accuracy vs. ω")
        plt.xlabel("ω")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(figures_dir, "accuracy_last_epoch_vs_omega.png"))
        plt.close()

        # --- Plot 4b: Accuracy and Loss vs. epoch for each ω and epsilon ---
        for (omega_val, epsilon_val), group in combined_logs.groupby(['omega', 'epsilon']):
            if 'loss' not in group.columns:
                continue
            fig, ax1 = plt.subplots(figsize=(10, 6))
            color_acc = 'tab:blue'
            color_loss = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy', color=color_acc)
            ax1.plot(group['epoch'], group['accuracy'], color=color_acc, marker='o', label='Accuracy')
            ax1.tick_params(axis='y', labelcolor=color_acc)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Loss', color=color_loss)
            ax2.plot(group['epoch'], group['loss'], color=color_loss, marker='x', label='Loss')
            ax2.tick_params(axis='y', labelcolor=color_loss)

            plt.title(f'Accuracy and Loss vs. Epoch (ω={omega_val}, ε={epsilon_val})')
            fig.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"accuracy_loss_vs_epoch_omega_{omega_val}_epsilon_{epsilon_val}.png"))
            plt.close(fig)

        # Use combined_df for remaining plots
        # --- Plot 5: Mean error distance vs. ω ---
        mean_error_dist = (
            combined_df.groupby("omega")["error_distance"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(6, 4))
        sns.lineplot(x="omega", y="error_distance", marker="o", data=mean_error_dist)
        plt.title("Mean Error Distance vs. ω")
        plt.xlabel("ω")
        plt.ylabel("Mean Error Distance")
        plt.savefig(os.path.join(figures_dir, "mean_error_distance_vs_omega.png"))
        plt.close()

        # --- Plot 6: Histogram of error distance over all examples ---
        # Aggregate error distances across all runs / examples and plot histogram
        try:
            # Make sure error_distance is integer-valued for binning
            combined_df['error_distance'] = combined_df['error_distance'].astype(int)
            max_dist = int(combined_df['error_distance'].max()) if not combined_df['error_distance'].empty else 0
            # Create bin edges so integers are centered
            bins = np.arange(0, max_dist + 2) - 0.5

            plt.figure(figsize=(6, 4))
            sns.histplot(data=combined_df, x='error_distance', bins=bins, discrete=True)
            plt.title('Histogram of Error Distance (all examples)')
            plt.xlabel('Error distance')
            plt.ylabel('Count')
            plt.savefig(os.path.join(figures_dir, 'error_distance_histogram_all_examples.png'))
            plt.close()

            overall_mean = combined_df['error_distance'].mean()
            overall_median = combined_df['error_distance'].median()
            print(f"Overall mean error distance: {overall_mean:.3f}, median: {overall_median}")
        except Exception as e:
            print(f"Could not create aggregated error-distance histogram: {e}")

    print(f"Saved figures in: {figures_dir}")

analyze_single_digit_modules(RAW_DIR)
