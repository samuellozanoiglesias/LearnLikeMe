# USE: nohup python paper_figure_extractors.py STUDY > logs_paper_figs_extractor.out 2>&1 &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
STUDY_NAME = str(sys.argv[1]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)

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

def analyze_single_digit_modules(raw_dir_carry, raw_dir_unit, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    run_dirs_carry = [
        os.path.join(raw_dir_carry, d) for d in os.listdir(raw_dir_carry)
        if os.path.isdir(os.path.join(raw_dir_carry, d)) and "Training_" in d
    ]
    run_dirs_unit = [
        os.path.join(raw_dir_unit, d) for d in os.listdir(raw_dir_unit)
        if os.path.isdir(os.path.join(raw_dir_unit, d)) and "Training_" in d
    ]

    all_logs_carry = []
    all_logs_unit = []

    for run in run_dirs_carry:
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
                    break

        # --- Read training_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega"] = omega
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
                    break

        # --- Read training_log ---
        log_df = pd.read_csv(log_path)
        if "epoch" not in log_df.columns:
            log_df["epoch"] = np.arange(len(log_df))
        log_df["omega"] = omega
        all_logs_unit.append(log_df)

    if all_logs_carry and all_logs_unit:
        combined_logs_carry = pd.concat(all_logs_carry, ignore_index=True)
        combined_logs_unit = pd.concat(all_logs_unit, ignore_index=True)

        # --- Plot 3: Accuracy last epoch vs. omega ---
        last_epoch_acc_carry = (
            combined_logs_carry.groupby("omega")[["epoch", "accuracy"]]
            .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"])
            .reset_index(name="accuracy")
        )
        last_epoch_acc_carry["error"] = 100 - last_epoch_acc_carry["accuracy"] * 100

        last_epoch_acc_unit = (
            combined_logs_unit.groupby("omega")[["epoch", "accuracy"]]
            .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"])
            .reset_index(name="accuracy")
        )
        last_epoch_acc_unit["error"] = 100 - last_epoch_acc_unit["accuracy"] * 100

        plt.figure(figsize=(10, 6))
        sns.lineplot(x="omega", y="error", marker="o", data=last_epoch_acc_carry, label="Carry Extractor", color="brown", linestyle="-", linewidth=3, markersize=10)
        sns.lineplot(x="omega", y="error", marker="o", data=last_epoch_acc_unit, label="Unit Extractor", color="green", linestyle="-", linewidth=3, markersize=10)
        plt.xlabel("Magnitude Noise (ω)", fontsize=30)
        plt.legend(fontsize=26)
        plt.ylabel("Average Error (%)", fontsize=30)
        plt.ylim(-1, 41)
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "error_last_epoch_vs_omega.png"))
        plt.close()

    print(f"Saved figures in: {figures_dir}")

analyze_single_digit_modules(RAW_DIR_CARRY, RAW_DIR_UNIT, FIGURES_DIR)