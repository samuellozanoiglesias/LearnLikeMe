import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Config ---
CLUSTER = "cuenca" # Cuenca, Brigit or Local
MODULE_NAME = "unit_extractor"  # unit_extractor or carry_over_extractor

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}" 

def analyze_single_digit_modules(raw_dir):
    """
    Analyze results of single-digit modules for different values of Weber's fraction.
    Saves histograms and error graphs in raw_dir/figures
    """
    figures_dir = os.path.join(raw_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    run_dirs = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir)
                if os.path.isdir(os.path.join(raw_dir, d)) and "Training_" in d]

    all_results = []

    for run in run_dirs:
        results_path = os.path.join(run, "training_results.csv")
        config_path = os.path.join(run, "config.txt")

        if not os.path.exists(results_path):
            print(f"No training_results.csv were found in {run}")
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

        # --- Plot 1: Error histograms by distance ---
        error_df = df[df["error"]]
        plt.figure(figsize=(6, 4))
        sns.countplot(x="error_distance", data=error_df)
        plt.title(f"Errors by distance - ω={omega}")
        plt.xlabel("Distance between the correct answer and the prediction")
        plt.ylabel("Frequence")
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

    # --- Aggregated analysis ---
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        plt.figure(figsize=(7, 5))
        sns.barplot(x="error_distance", y="error", hue="omega",
                    data=combined_df, estimator=np.mean)
        plt.title("Proportion of errors per distance and ω")
        plt.xlabel("Distance")
        plt.ylabel("Proportion of errors")
        plt.savefig(os.path.join(figures_dir, "error_rate_by_distance_all_omegas.png"))
        plt.close()

        print(f"Saved figures in: {figures_dir}")
    else:
        print("No results were found.")

analyze_single_digit_modules(RAW_DIR)    
