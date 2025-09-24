# USE: nohup python analyze_training_decision_module.py WI > logs_analysis_training_decision.out 2>&1 &

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
PARAM_TYPE = str(sys.argv[1]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}/{PARAM_TYPE}"

def analyze_multidigit_module(raw_dir):
    figures_dir = os.path.join(raw_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # The expected structure is raw_dir/epsilon_{epsilon:.2f}/Training_{timestamp}
    run_dirs = []
    for d in os.listdir(raw_dir):
        dpath = os.path.join(raw_dir, d)
        if not os.path.isdir(dpath):
            continue
        # If this is a Training_ folder directly, add it
        if d.startswith("Training_"):
            run_dirs.append(dpath)
            continue
        # Otherwise, look for Training_ subfolders (this handles epsilon_{...} folders)
        try:
            for sub in os.listdir(dpath):
                subpath = os.path.join(dpath, sub)
                if os.path.isdir(subpath) and sub.startswith("Training_"):
                    run_dirs.append(subpath)
        except Exception:
            continue

    all_results = []
    all_logs = []

    for run in run_dirs:
        results_path = os.path.join(run, "training_results.csv")
        log_path = os.path.join(run, "training_log.csv")
        config_path = os.path.join(run, "config.txt")

        omega = None
        param_init_type = None
        epsilon = None
        with open(config_path, "r") as f:
            for line in f:
                if "Weber fraction (Omega):" in line and ":" in line:
                    omega = float(line.strip().split(":")[-1])
                if "Parameter Initialization Type (Wise initialization or Random initialization):" in line:
                    param_init_type = line.strip().split(":")[-1].strip()
                if "Noise Factor for Initialization Parameters (Epsilon):" in line:
                    epsilon = float(line.strip().split(":")[-1])

        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df["omega"] = omega
            df["param_init_type"] = param_init_type
            df["epsilon"] = epsilon
            df["error"] = df["y (true)"] != df["y (pred)"]
            df["error_distance"] = np.abs(df["y (true)"] - df["y (pred)"])
            safe_omega = str(omega).replace('.', '_') if omega is not None else 'None'
            safe_epsilon = str(epsilon).replace('.', '_') if epsilon is not None else 'None'
            all_results.append(df)

            # --- Plot 1: Errors per distance (normalized/proportion) ---
            error_df = df[df["error"]]
            if not error_df.empty:
                dist = error_df["error_distance"].value_counts(normalize=True).sort_index()
                plt.figure(figsize=(10, 5))
                plt.bar(dist.index.astype(str), dist.values)
                plt.title(f"Errors by distance (proportion) - ω={omega}")
                plt.xlabel("Distance between the correct answer and the prediction")
                plt.ylabel("Proportion")
                plt.ylim(0, 1)
                plt.savefig(os.path.join(figures_dir, f"errors_by_distance_omega_{safe_omega}_epsilon_{safe_epsilon}.png"))
                plt.close()
            else:
                print(f"No errors to plot for run {run}")
#
            ## --- Plot 2: Confusion matrix ---
            #cm = confusion_matrix(df["y (true)"], df["y (pred)"], labels=range(100))
            #plt.figure(figsize=(6, 5))
            #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            #plt.title(f"Confusion matrix - ω={omega}")
            #plt.xlabel("Predicted")
            #plt.ylabel("Real")
            #plt.savefig(os.path.join(figures_dir, f"confusion_matrix_omega_{safe_omega}_epsilon_{safe_epsilon}.png"))
            #plt.close()

        else:
            print(f"No training_results.csv were found in {run}")
            continue
        
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            if "epoch" not in log_df.columns:
                log_df["epoch"] = np.arange(len(log_df))
            log_df["omega"] = omega
            log_df["param_init_type"] = param_init_type
            log_df["epsilon"] = epsilon
            all_logs.append(log_df)

        else:
            print(f"No training_log.csv were found in {run}")
            continue        

    # --- Aggregated analysis ---
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_results_path = os.path.join(figures_dir, "combined_results.csv")
        combined_df.to_csv(combined_results_path, index=False)
        print(f"Combined results saved to: {combined_results_path}")
        # --- Aggregated normalized error-distance plots ---
        try:
            # Overall aggregation across all runs
            overall_error = combined_df[combined_df['error']]['error_distance'].value_counts(normalize=True).sort_index()
            if not overall_error.empty:
                plt.figure(figsize=(10, 5))
                plt.bar(overall_error.index.astype(str), overall_error.values)
                plt.title('Overall Errors by Distance (proportion) - all epsilons & omegas')
                plt.xlabel('Distance between the correct answer and the prediction')
                plt.ylabel('Proportion')
                plt.ylim(0, 1)
                plt.savefig(os.path.join(figures_dir, 'errors_by_distance_overall_proportion.png'))
                plt.close()

            # Per-epsilon aggregation (aggregating across omegas)
            if 'epsilon' in combined_df.columns:
                for eps in sorted(combined_df['epsilon'].dropna().unique()):
                    sub = combined_df[(combined_df['epsilon'] == eps) & (combined_df['error'])]
                    if sub.empty:
                        continue
                    d = sub['error_distance'].value_counts(normalize=True).sort_index()
                    plt.figure(figsize=(10, 5))
                    plt.bar(d.index.astype(str), d.values)
                    plt.title(f'Errors by Distance (proportion) - epsilon={eps} (all omegas)')
                    plt.xlabel('Distance between the correct answer and the prediction')
                    plt.ylabel('Proportion')
                    plt.ylim(0, 1)
                    fname = os.path.join(figures_dir, f'errors_by_distance_epsilon_{str(eps).replace(".","_")}_proportion.png')
                    plt.savefig(fname)
                    plt.close()

            # Per-omega aggregation (aggregating across epsilons)
            if 'omega' in combined_df.columns:
                for om in sorted(combined_df['omega'].dropna().unique()):
                    sub = combined_df[(combined_df['omega'] == om) & (combined_df['error'])]
                    if sub.empty:
                        continue
                    d = sub['error_distance'].value_counts(normalize=True).sort_index()
                    plt.figure(figsize=(10, 5))
                    plt.bar(d.index.astype(str), d.values)
                    plt.title(f'Errors by Distance (proportion) - omega={om} (all epsilons)')
                    plt.xlabel('Distance between the correct answer and the prediction')
                    plt.ylabel('Proportion')
                    plt.ylim(0, 1)
                    fname = os.path.join(figures_dir, f'errors_by_distance_omega_{str(om).replace(".","_")}_proportion.png')
                    plt.savefig(fname)
                    plt.close()
        except Exception as e:
            print(f'Could not create aggregated error-distance plots: {e}')
        # --- Plot 3: Mean error distance vs. omegas (3D) ---
        # Group by omegas and also by parameter initialization type and epsilon
        mean_error_dist = (
            combined_df.groupby(["omega", "param_init_type", "epsilon"])["error_distance"]
            .mean()
            .reset_index()
        )

        # Create plots per (param_init_type, epsilon) combination
        unique_param_combos = mean_error_dist[["param_init_type", "epsilon"]].drop_duplicates().values.tolist()
        for param_init_type, eps in unique_param_combos:
            subset = mean_error_dist[(mean_error_dist["param_init_type"] == param_init_type) & (mean_error_dist["epsilon"] == eps)]
            if subset.empty:
                continue

            # Plot: Mean error distance vs omega for this (param_init_type, epsilon)
            plt.figure(figsize=(10, 6))
            plt.plot(subset["omega"], subset["error_distance"], marker='o')
            plt.title(f"Mean Error Distance vs ω - {param_init_type}, ε={eps}")
            plt.xlabel('ω')
            plt.ylabel('Mean Error Distance')
            eps_str = str(eps).replace('.', '_') if eps is not None else 'None'
            plt.savefig(os.path.join(figures_dir, f"mean_error_distance_vs_omega_param_{param_init_type}_eps_{eps_str}.png"))
            plt.close()

        # --- Compare different epsilons in one plot ---
        try:
            eps_table = mean_error_dist.copy()
            eps_table = eps_table[eps_table["epsilon"].notnull()]
            if not eps_table.empty:
                for param_type in eps_table["param_init_type"].unique():
                    sub = eps_table[eps_table["param_init_type"] == param_type]
                    # pivot so that each omega value is a column
                    pivot = sub.pivot_table(index="epsilon", columns="omega", values="error_distance")
                    if pivot.shape[1] > 0:
                        plt.figure(figsize=(12, 8))
                        for col in pivot.columns:
                            plt.plot(pivot.index, pivot[col], marker='o', label=col)
                        plt.title(f"Mean Error Distance vs. Epsilon - param={param_type}")
                        plt.xlabel('Epsilon')
                        plt.ylabel('Mean Error Distance')
                        plt.legend(loc='best', fontsize='small', ncol=2)
                        plt.savefig(os.path.join(figures_dir, f"mean_error_distance_vs_epsilon_param_{param_type}.png"))
                        plt.close()

                    # aggregated mean ± std across omegas
                    agg = sub.groupby('epsilon')['error_distance'].agg(['mean', 'std']).reset_index()
                    if not agg.empty:
                        plt.figure(figsize=(8, 6))
                        plt.errorbar(agg['epsilon'], agg['mean'], yerr=agg['std'].fillna(0), marker='o')
                        plt.title(f"Mean Error Distance vs. Epsilon (avg over omegas) - param={param_type}")
                        plt.xlabel('Epsilon')
                        plt.ylabel('Mean Error Distance')
                        plt.savefig(os.path.join(figures_dir, f"mean_error_distance_vs_epsilon_agg_param_{param_type}.png"))
                        plt.close()
        except Exception as e:
            print(f"Could not create epsilon comparison plots for error distance: {e}")

    if all_logs:
        combined_logs = pd.concat(all_logs, ignore_index=True)
        combined_logs_path = os.path.join(figures_dir, "combined_logs.csv")
        combined_logs.to_csv(combined_logs_path, index=False)
        print(f"Combined logs saved to: {combined_logs_path}")
        
        # --- Accuracy per (omega, param_init_type, epsilon) ---
        last_epoch_acc = (
            combined_logs.groupby(["omega", "param_init_type", "epsilon"])[["epoch", "accuracy"]]
            .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"])
            .reset_index(name="accuracy")
        )

        print(last_epoch_acc.head())

        # Create plots per (param_init_type, epsilon) combination: accuracy vs omega
        unique_param_combos = last_epoch_acc[["param_init_type", "epsilon"]].drop_duplicates().values.tolist()
        for param_init_type, eps in unique_param_combos:
            subset = last_epoch_acc[(last_epoch_acc["param_init_type"] == param_init_type) & (last_epoch_acc["epsilon"] == eps)]
            if subset.empty:
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(subset["omega"], subset["accuracy"], marker='o')
            plt.title(f"Last Epoch Accuracy vs ω - {param_init_type}, ε={eps}")
            plt.xlabel('ω')
            plt.ylabel('Accuracy')
            eps_str = str(eps).replace('.', '_') if eps is not None else 'None'
            plt.savefig(os.path.join(figures_dir, f"accuracy_vs_omega_param_{param_init_type}_eps_{eps_str}.png"))
            plt.close()

        # --- 3D plot: omega (x), epsilon (y), accuracy (z) per parameter init type ---
        for param_type in last_epoch_acc['param_init_type'].unique():
            sub = last_epoch_acc[last_epoch_acc['param_init_type'] == param_type]
            if sub.empty:
                continue

            # Prepare figure and 3D axes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter the raw points first
            ax.scatter(sub['omega'], sub['epsilon'], sub['accuracy'], c=sub['accuracy'], cmap='viridis', depthshade=True)

            # Try to build a regular grid (omega x epsilon) for surface plotting
            try:
                omegas = np.array(sorted(sub['omega'].dropna().unique()))
                epss = np.array(sorted(sub['epsilon'].dropna().unique()))

                if len(omegas) > 1 and len(epss) > 1:
                    # pivot to grid of accuracies
                    pivot = sub.pivot_table(index='omega', columns='epsilon', values='accuracy')
                    # ensure full index/columns in sorted order
                    pivot = pivot.reindex(index=omegas, columns=epss)

                    # Interpolate missing values along each axis when possible
                    pivot = pivot.sort_index()
                    pivot = pivot.interpolate(axis=0, limit_direction='both').interpolate(axis=1, limit_direction='both')
                    # if still NaNs at edges, fill with nearest
                    pivot = pivot.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
                    pivot = pivot.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

                    Z = pivot.values
                    X, Y = np.meshgrid(omegas, epss, indexing='ij')

                    # plot surface (transpose or orientation should match pivot shape)
                    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
                else:
                    # Not enough distinct omegas/epsilons for a surface — keep scatter only
                    pass
            except Exception as e:
                print(f"Could not draw surface for param {param_type}: {e}")

            ax.set_xlabel('ω')
            ax.set_ylabel('ε')
            # Reverse epsilon axis so it runs from high to low (visually left→right)
            try:
                if 'epss' in locals() and len(epss) > 0:
                    ax.set_ylim(epss.max(), epss.min())
            except Exception:
                # fallback: attempt to invert y-axis
                try:
                    ax.invert_yaxis()
                except Exception:
                    pass
            ax.set_zlabel('Accuracy')
            plt.title(f"Accuracy vs ω and ε - param={param_type}")
            plt.savefig(os.path.join(figures_dir, f"accuracy_vs_omega_epsilon_3d_param_{param_type}.png"))
            plt.close()

        # --- Epsilon-vs-accuracy comparison plots ---
        acc_table = last_epoch_acc.copy()
        acc_table = acc_table[acc_table["epsilon"].notnull()]
        if not acc_table.empty:
            for param_type in acc_table["param_init_type"].unique():
                sub = acc_table[acc_table["param_init_type"] == param_type]
                pivot = sub.pivot_table(index="epsilon", columns="omega", values="accuracy")
                if pivot.shape[1] > 0:
                    plt.figure(figsize=(12, 8))
                    for col in pivot.columns:
                        plt.plot(pivot.index, pivot[col], marker='o', label=str(col))
                    plt.title(f"Last Epoch Accuracy vs. Epsilon - param={param_type}")
                    plt.xlabel('Epsilon')
                    plt.ylabel('Accuracy')
                    plt.legend(loc='best', fontsize='small', ncol=2)
                    plt.savefig(os.path.join(figures_dir, f"accuracy_vs_epsilon_param_{param_type}.png"))
                    plt.close()

                # aggregated mean ± std across omegas
                agg = sub.groupby('epsilon')['accuracy'].agg(['mean', 'std']).reset_index()
                if not agg.empty:
                    plt.figure(figsize=(8, 6))
                    plt.errorbar(agg['epsilon'], agg['mean'], yerr=agg['std'].fillna(0), marker='o')
                    plt.title(f"Last Epoch Accuracy vs. Epsilon (avg over omegas) - param={param_type}")
                    plt.xlabel('Epsilon')
                    plt.ylabel('Accuracy')
                    plt.savefig(os.path.join(figures_dir, f"accuracy_vs_epsilon_agg_param_{param_type}.png"))
                    plt.close()

    print(f"Saved figures in: {figures_dir}")

analyze_multidigit_module(RAW_DIR)
