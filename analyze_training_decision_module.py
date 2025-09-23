import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
MODULE_NAME = "decision_module"

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}"

def analyze_multidigit_module(raw_dir):
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

        omega_unit = None
        omega_carry = None
        with open(config_path, "r") as f:
            for line in f:
                if "Weber fraction for Unit Extractor (Omega):" in line:
                    omega_unit = float(line.strip().split(":")[-1])
                if "Weber fraction for Carry Extractor (Omega):" in line:
                    omega_carry = float(line.strip().split(":")[-1])

        df["omega_unit"] = omega_unit
        df["omega_carry"] = omega_carry
        df["error"] = df["y (real)"] != df["pred"]
        df["error_distance"] = np.abs(df["y (real)"] - df["pred"])
        all_results.append(df)

        # --- Plot 1: Errors per distance ---
        error_df = df[df["error"]]
        plt.figure(figsize=(6, 4))
        sns.countplot(x="error_distance", data=error_df)
        plt.title(f"Errors by distance - ω_unit={omega_unit}, ω_carry={omega_carry}")
        plt.xlabel("Distance between the correct answer and the prediction")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(figures_dir, f"errors_by_distance_omega_unit_{omega_unit}_omega_carry_{omega_carry}.png"))
        plt.close()

        # --- Plot 2: Confusion matrix ---
        cm = confusion_matrix(df["y (real)"], df["pred"], labels=range(100))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion matrix - ω_unit={omega_unit}, ω_carry={omega_carry}")
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.savefig(os.path.join(figures_dir, f"confusion_matrix_omega_unit_{omega_unit}_omega_carry_{omega_carry}.png"))
        plt.close()

        # --- Read trainig_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega_unit"] = omega_unit
        log_df["omega_carry"] = omega_carry
        all_logs.append(log_df)

    # --- Aggregated analysis ---
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # --- Plot 3: Mean error distance vs. omegas (3D) ---
        mean_error_dist = (
            combined_df.groupby(["omega_unit", "omega_carry"])["error_distance"]
            .mean()
            .reset_index()
        )

        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(mean_error_dist["omega_unit"], 
                           mean_error_dist["omega_carry"], 
                           mean_error_dist["error_distance"],
                           c=mean_error_dist["error_distance"],
                           cmap='viridis')
        
        ax.set_xlabel('ω (Unit)')
        ax.set_ylabel('ω (Carry)')
        ax.set_zlabel('Mean Error Distance')
        plt.colorbar(scatter)
        plt.title("Mean Error Distance vs. ω (Unit & Carry)")
        plt.savefig(os.path.join(figures_dir, "mean_error_distance_vs_omegas_3d.png"))
        plt.close()

        # 2D plots with fixed values
        unique_omega_units = sorted(mean_error_dist["omega_unit"].unique())
        unique_omega_carries = sorted(mean_error_dist["omega_carry"].unique())

        # Fixed omega_unit plots
        plt.figure(figsize=(12, 8))
        for omega_unit in unique_omega_units:
            subset = mean_error_dist[mean_error_dist["omega_unit"] == omega_unit]
            plt.plot(subset["omega_carry"], subset["error_distance"], 
                    marker='o', label=f'ω_unit={omega_unit}')
        
        plt.title("Mean Error Distance vs. ω (Carry) for different ω (Unit)")
        plt.xlabel('ω (Carry)')
        plt.ylabel('Mean Error Distance')
        plt.legend()
        plt.savefig(os.path.join(figures_dir, "mean_error_distance_fixed_omega_unit.png"))
        plt.close()

        # Fixed omega_carry plots
        plt.figure(figsize=(12, 8))
        for omega_carry in unique_omega_carries:
            subset = mean_error_dist[mean_error_dist["omega_carry"] == omega_carry]
            plt.plot(subset["omega_unit"], subset["error_distance"], 
                    marker='o', label=f'ω_carry={omega_carry}')
        
        plt.title("Mean Error Distance vs. ω (Unit) for different ω (Carry)")
        plt.xlabel('ω (Unit)')
        plt.ylabel('Mean Error Distance')
        plt.legend()
        plt.savefig(os.path.join(figures_dir, "mean_error_distance_fixed_omega_carry.png"))
        plt.close()

    if all_logs:
        combined_logs = pd.concat(all_logs, ignore_index=True)

        # --- Plot 4: Accuracy last epoch vs. omegas (3D) ---
        last_epoch_acc = (
            combined_logs.groupby(["omega_unit", "omega_carry"])[["epoch", "accuracy"]]
            .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"])
            .reset_index(name="accuracy")
        )

        print(last_epoch_acc.head())

        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(last_epoch_acc["omega_unit"], 
                           last_epoch_acc["omega_carry"], 
                           last_epoch_acc["accuracy"],
                           c=last_epoch_acc["accuracy"],
                           cmap='viridis')
        
        ax.set_xlabel('ω (Unit)')
        ax.set_ylabel('ω (Carry)')
        ax.set_zlabel('Accuracy')
        plt.colorbar(scatter)
        plt.title("Last Epoch Accuracy vs. ω (Unit & Carry)")
        plt.savefig(os.path.join(figures_dir, "accuracy_last_epoch_vs_omegas_3d.png"))
        plt.close()

        # 2D plots with fixed values
        unique_omega_units = sorted(last_epoch_acc["omega_unit"].unique())
        unique_omega_carries = sorted(last_epoch_acc["omega_carry"].unique())

        # Fixed omega_unit plots
        plt.figure(figsize=(12, 8))
        for omega_unit in unique_omega_units:
            subset = last_epoch_acc[last_epoch_acc["omega_unit"] == omega_unit]
            plt.plot(subset["omega_carry"], subset["accuracy"], 
                    marker='o', label=f'ω_unit={omega_unit}')
        
        plt.title("Last Epoch Accuracy vs. ω (Carry) for different ω (Unit)")
        plt.xlabel('ω (Carry)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(figures_dir, "accuracy_last_epoch_fixed_omega_unit.png"))
        plt.close()

        # Fixed omega_carry plots
        plt.figure(figsize=(12, 8))
        for omega_carry in unique_omega_carries:
            subset = last_epoch_acc[last_epoch_acc["omega_carry"] == omega_carry]
            plt.plot(subset["omega_unit"], subset["accuracy"], 
                    marker='o', label=f'ω_carry={omega_carry}')
        
        plt.title("Last Epoch Accuracy vs. ω (Unit) for different ω (Carry)")
        plt.xlabel('ω (Unit)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(figures_dir, "accuracy_last_epoch_fixed_omega_carry.png"))
        plt.close()

    print(f"Saved figures in: {figures_dir}")

analyze_multidigit_module(RAW_DIR)
