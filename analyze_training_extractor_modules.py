# USE: nohup python analyze_training_extractor_modules.py unit_extractor FOURTH_STUDY > logs_analyze_training_extractor.out &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
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
        with open(config_path, "r") as f:
            for line in f:
                if "Weber fraction:" in line:
                    omega = float(line.strip().split(":")[-1])
                    break

        df["omega"] = omega
        df["error"] = df["y (real)"] != df["pred"]
        df["error_distance"] = np.abs(df["y (real)"] - df["pred"])
        all_results.append(df)

        # --- Plot 1: Errors per distance ---
        error_df = df[df["error"]]
        plt.figure(figsize=(6, 4))
        sns.countplot(x="error_distance", data=error_df)
        plt.title(f"Errors by distance - ω={omega}")
        plt.xlabel("Distance between the correct answer and the prediction")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(figures_dir, f"errors_by_distance_omega_{omega}.png"))
        plt.close()

        # --- Plot 2: Confusion matrix ---
        cm = confusion_matrix(df["y (real)"], df["pred"], labels=range(10))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion matrix - ω={omega}")
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.savefig(os.path.join(figures_dir, f"confusion_matrix_omega_{omega}.png"))
        plt.close()

        # --- Read trainig_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega"] = omega
        all_logs.append(log_df)

    # --- Aggregated analysis ---
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        plt.figure(figsize=(10, 8))
        sns.barplot(
            x="error_distance", y="error", hue="omega",
            data=combined_df, estimator=np.mean
        )
        plt.title("Proportion of errors per distance and ω")
        plt.xlabel("Distance")
        plt.ylabel("Proportion of errors")
        plt.savefig(os.path.join(figures_dir, "error_rate_by_distance_all_omegas.png"))
        plt.close()

    if all_logs:
        combined_logs = pd.concat(all_logs, ignore_index=True)

        # --- Plot 3: Accuracy last epoch vs. omega ---
        last_epoch_acc = (
            combined_logs.groupby("omega")[["epoch", "accuracy"]]
            .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"])
            .reset_index(name="accuracy")
        )

        plt.figure(figsize=(6, 4))
        sns.lineplot(x="omega", y="accuracy", marker="o", data=last_epoch_acc)
        plt.title("Last Epoch Accuracy vs. ω")
        plt.xlabel("ω")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(figures_dir, "accuracy_last_epoch_vs_omega.png"))
        plt.close()


        # --- Plot 4: Accuracy vs. epoch per ω ---
        plt.figure(figsize=(30, 15))
        sns.lineplot(
            x="epoch", y="accuracy", hue="omega",
            data=combined_logs, marker="o"
        )
        plt.title("Accuracy vs. epoch per ω")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(title="ω")
        plt.savefig(os.path.join(figures_dir, "accuracy_vs_epoch_by_omega.png"))
        plt.close()

        # --- Plot 4b: Accuracy and Loss vs. epoch for each ω ---
        for omega_val in sorted(combined_logs['omega'].unique()):
            omega_logs = combined_logs[combined_logs['omega'] == omega_val]
            if 'loss' not in omega_logs.columns:
                continue
            fig, ax1 = plt.subplots(figsize=(10, 6))
            color_acc = 'tab:blue'
            color_loss = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy', color=color_acc)
            ax1.plot(omega_logs['epoch'], omega_logs['accuracy'], color=color_acc, marker='o', label='Accuracy')
            ax1.tick_params(axis='y', labelcolor=color_acc)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Loss', color=color_loss)
            ax2.plot(omega_logs['epoch'], omega_logs['loss'], color=color_loss, marker='x', label='Loss')
            ax2.tick_params(axis='y', labelcolor=color_loss)

            plt.title(f'Accuracy and Loss vs. Epoch (ω={omega_val})')
            fig.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"accuracy_loss_vs_epoch_omega_{omega_val}.png"))
            plt.close(fig)

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
