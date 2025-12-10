# USE: nohup python analyze_training_decision_module.py 2 STUDY WI argmax > logs_analysis_training_decision.out 2>&1 &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"

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
        if not os.path.exists(config_path):
            print(f"No config.txt were found in {run}")
            continue        

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

        else:
            print(f"No training_results.csv were found in {run}")
            continue
        
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path, low_memory=False)
            # Ensure epoch and accuracy columns are numeric before any aggregation
            if "epoch" not in log_df.columns:
                log_df["epoch"] = np.arange(len(log_df))
            else:
                log_df["epoch"] = pd.to_numeric(log_df["epoch"], errors='coerce')
            
            acc_cols = [
                "test_pairs_no_carry_small_accuracy",
                "test_pairs_no_carry_large_accuracy",
                "test_pairs_carry_small_accuracy",
                "test_pairs_carry_large_accuracy"
            ]
            for col in acc_cols:
                if col in log_df.columns:
                    log_df[col] = pd.to_numeric(log_df[col], errors='coerce')
            
            # Also ensure accuracy and loss are numeric if they exist
            if 'accuracy' in log_df.columns:
                log_df['accuracy'] = pd.to_numeric(log_df['accuracy'], errors='coerce')
            if 'loss' in log_df.columns:
                log_df['loss'] = pd.to_numeric(log_df['loss'], errors='coerce')
            log_df["omega"] = omega
            log_df["param_init_type"] = param_init_type
            log_df["epsilon"] = epsilon
            # add a run identifier so we can aggregate across runs
            log_df["run"] = os.path.basename(run)
            # --- Per-run: plot accuracies over epochs for the four test splits (if present) ---
            figures_dir = os.path.join(raw_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)

            acc_cols = [
                "test_pairs_no_carry_small_accuracy",
                "test_pairs_no_carry_large_accuracy",
                "test_pairs_carry_small_accuracy",
                "test_pairs_carry_large_accuracy",
            ]
            acc_labels = [
                "no_carry_small",
                "no_carry_large",
                "carry_small",
                "carry_large",
            ]
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

            if any(c in log_df.columns for c in acc_cols):
                safe_omega = str(omega).replace('.', '_') if omega is not None else 'None'
                safe_epsilon = str(epsilon).replace('.', '_') if epsilon is not None else 'None'
                plt.figure(figsize=(10, 6))
                for col, lbl, colcol in zip(acc_cols, acc_labels, colors):
                    if col in log_df.columns:
                        # Filter out rows with NaN values in epoch or accuracy columns
                        valid_mask = log_df['epoch'].notna() & log_df[col].notna()
                        if valid_mask.any():
                            plt.plot(log_df.loc[valid_mask, 'epoch'], log_df.loc[valid_mask, col], label=lbl, color=colcol)
                plt.title(f"Accuracies over epochs - param={param_init_type}, ω={omega}, ε={epsilon}")
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                plt.ylim(-5, 105)
                plt.legend(loc='best')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(figures_dir, f"accuracies_epochs_param_{param_init_type}_omega_{safe_omega}_eps_{safe_epsilon}.png"))
                plt.close()

            all_logs.append(log_df)

        else:
            print(f"No training_log.csv were found in {run}")
            continue        

    # --- Aggregated analysis ---
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_results_path = os.path.join(raw_dir, "combined_results.csv")
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
                        plt.errorbar(agg['epsilon'], agg['mean'], yerr=agg['std'].fillna(0).infer_objects(copy=False), marker='o')
                        plt.title(f"Mean Error Distance vs. Epsilon (avg over omegas) - param={param_type}")
                        plt.xlabel('Epsilon')
                        plt.ylabel('Mean Error Distance')
                        plt.savefig(os.path.join(figures_dir, f"mean_error_distance_vs_epsilon_agg_param_{param_type}.png"))
                        plt.close()
        except Exception as e:
            print(f"Could not create epsilon comparison plots for error distance: {e}")

    if all_logs:
        combined_logs = pd.concat(all_logs, ignore_index=True)
        combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
        combined_logs.to_csv(combined_logs_path, index=False)
        print(f"Combined logs saved to: {combined_logs_path}")
        # --- Aggregated epoch plots for the four requested test accuracy columns ---
        acc_cols = [
            "test_pairs_no_carry_small_accuracy",
            "test_pairs_no_carry_large_accuracy",
            "test_pairs_carry_small_accuracy",
            "test_pairs_carry_large_accuracy",
        ]
        acc_labels = ["no_carry_small", "no_carry_large", "carry_small", "carry_large"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        def plot_mean_std(pivot_df, title, fname):
            plt.figure(figsize=(10, 6))
            for col, lbl, colcol in zip(acc_cols, acc_labels, colors):
                if col not in pivot_df:
                    continue
                mean = pivot_df[col].mean(axis=1)
                std = pivot_df[col].std(axis=1).fillna(0).infer_objects(copy=False)
                mean = pd.to_numeric(mean, errors='coerce')
                std = pd.to_numeric(std, errors='coerce')
                plt.plot(mean.index, mean.values, label=lbl, color=colcol)
                plt.fill_between(mean.index, (mean - std).values, (mean + std).values, color=colcol, alpha=0.2)
            plt.title(title)
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.ylim(0, 1)
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.savefig(fname)
            plt.close()

        # Helper to plot a mapping of acc_col -> DataFrame(runs as columns, index=epoch)
        def plot_pivot_wrapper(pw, title, fname, subplot=False, max_x=None):
            plt.figure(figsize=(10, 6))
            for col, lbl, colcol in zip(acc_cols, acc_labels, colors):
                if col not in pw:
                    continue
                df_runs = pw[col]
                if df_runs is None or df_runs.empty:
                    continue
                mean = df_runs.mean(axis=1)
                std = df_runs.std(axis=1).fillna(0)
                # Convert to numpy float arrays for plotting
                mean = pd.to_numeric(mean, errors='coerce').astype(float).to_numpy()
                std = pd.to_numeric(std, errors='coerce').astype(float).to_numpy()
                idx = np.array(df_runs.index, dtype=float)
                # Remove non-finite values
                mask = np.isfinite(mean) & np.isfinite(std) & np.isfinite(idx)
                mean = mean[mask]
                std = std[mask]
                idx = idx[mask]
                if len(mean) == 0:
                    continue
                plt.plot(idx, mean, label=lbl, color=colcol)
                plt.fill_between(idx, mean - std, mean + std, color=colcol, alpha=0.2)
            plt.title(title)
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.ylim(-5, 105)
            if subplot == True:
                plt.xlim(0, 400)
            if max_x is not None:
                plt.xlim(0, max_x)
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.savefig(fname)
            plt.close()

        # Work per param_init_type to match other plots grouping
        for param_type in combined_logs['param_init_type'].dropna().unique():
            subset_param = combined_logs[combined_logs['param_init_type'] == param_type]
            if subset_param.empty:
                continue

            # 1) Per (omega, epsilon) pair
            pairs = subset_param[['omega', 'epsilon']].drop_duplicates().values.tolist()
            for om, eps in pairs:
                sub = subset_param[(subset_param['omega'] == om) & (subset_param['epsilon'] == eps)]
                if sub.empty:
                    continue
                # build pivot per acc col: produce a DataFrame with MultiIndex epoch and columns per acc col where each column is a DataFrame of runs
                # We'll create a dict of DataFrames for each acc col where columns are run ids
                pivot_per_acc = {}
                for col in acc_cols:
                    if col in sub.columns:
                        p = sub.pivot_table(index='epoch', columns='run', values=col)
                        pivot_per_acc[col] = p
                # merge pivots into a single DataFrame with hierarchical columns by acc col
                if not pivot_per_acc:
                    continue
                # For plotting convenience, create a DataFrame where each acc col is a sub-DataFrame accessed in plot_mean_std
                # We'll pass a dict-like object; simpler is to create a DataFrame where columns are acc col names mapping to DataFrames
                # Instead, create a simple container: a DataFrame-like mapping via pandas Panel isn't available; instead, create a concat with keys
                concat = {}
                for col, dfp in pivot_per_acc.items():
                    # reindex epoch to full range
                    concat[col] = dfp

                safe_om = str(om).replace('.', '_') if om is not None else 'None'
                safe_eps = str(eps).replace('.', '_') if eps is not None else 'None'
                fname = os.path.join(figures_dir, f"accuracies_epochs_agg_param_{param_type}_omega_{safe_om}_eps_{safe_eps}.png")
                plot_pivot_wrapper(concat, f"Accuracies over epochs (mean ± std) - param={param_type}, ω={om}, ε={eps}", fname)

                # --- New plot: Accuracy (left y) and Loss (right y) vs Epoch for this (omega, epsilon) ---
                # Aggregate across runs by taking mean per epoch for accuracy and loss (if available)
                if 'accuracy' in sub.columns and 'loss' in sub.columns:
                    try:
                        # Filter out rows with invalid epoch, accuracy, or loss values
                        valid_sub = sub[sub['epoch'].notna() & sub['accuracy'].notna() & sub['loss'].notna()]
                        if not valid_sub.empty:
                            agg = valid_sub.groupby('epoch').agg({'accuracy': 'mean', 'loss': 'mean'}).reset_index()
                            color_acc = 'tab:blue'
                            color_loss = 'tab:red'
                            fig, ax1 = plt.subplots(figsize=(10, 6))
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Accuracy', color=color_acc)
                            ax1.plot(agg['epoch'], agg['accuracy'], color=color_acc, marker='o', label='Accuracy')
                            ax1.tick_params(axis='y', labelcolor=color_acc)

                            ax2 = ax1.twinx()
                            ax2.set_ylabel('Loss', color=color_loss)
                            ax2.plot(agg['epoch'], agg['loss'], color=color_loss, marker='x', label='Loss')
                            ax2.tick_params(axis='y', labelcolor=color_loss)

                            plt.title(f'Accuracy and Loss vs. Epoch - param={param_type}, ω={om}, ε={eps}')
                            fig.tight_layout()
                            fname2 = os.path.join(figures_dir, f"accuracy_loss_vs_epoch_param_{param_type}_omega_{safe_om}_eps_{safe_eps}.png")
                            plt.savefig(fname2)
                            plt.close(fig)
                        else:
                            print(f"No valid accuracy/loss data for ω={om}, ε={eps}, param={param_type}")
                    except Exception as e:
                        print(f"Could not create accuracy/loss plot for ω={om}, ε={eps}, param={param_type}: {e}")

            # 2) Per omega aggregated across epsilons
            for om in sorted(subset_param['omega'].dropna().unique()):
                sub = subset_param[subset_param['omega'] == om]
                if sub.empty:
                    continue
                pw = {}
                for col in acc_cols:
                    if col in sub.columns:
                        pw[col] = sub.pivot_table(index='epoch', columns='run', values=col)
                if not pw:
                    continue
                fname = os.path.join(figures_dir, f"accuracies_epochs_agg_param_{param_type}_omega_{str(om).replace('.','_')}_all_eps.png")
                fname_max_x = os.path.join(figures_dir, f"accuracies_epochs_agg_param_{param_type}_omega_{str(om).replace('.','_')}_all_eps_maxx1000.png")
                plot_pivot_wrapper(pw, f"Accuracies over epochs (mean ± std) - param={param_type}, ω={om} (all ε)", fname)
                plot_pivot_wrapper(pw, f"Accuracies over epochs (mean ± std) - param={param_type}, ω={om} (all ε)", fname_max_x, max_x=1000)

            # 3) Per epsilon aggregated across omegas
            for eps in sorted(subset_param['epsilon'].dropna().unique()):
                sub = subset_param[subset_param['epsilon'] == eps]
                if sub.empty:
                    continue
                pw = {}
                for col in acc_cols:
                    if col in sub.columns:
                        pw[col] = sub.pivot_table(index='epoch', columns='run', values=col)
                if not pw:
                    continue
                fname = os.path.join(figures_dir, f"accuracies_epochs_agg_param_{param_type}_eps_{str(eps).replace('.','_')}_all_omegas.png")
                plot_pivot_wrapper(pw, f"Accuracies over epochs (mean ± std) - param={param_type}, ε={eps} (all ω)", fname)

            # 4) Overall aggregated across all omegas and epsilons for this param_type
            sub = subset_param
            pw = {}
            for col in acc_cols:
                if col in sub.columns:
                    pw[col] = sub.pivot_table(index='epoch', columns='run', values=col)
            if pw:
                fname = os.path.join(figures_dir, f"accuracies_epochs_agg_param_{param_type}_all_pairs.png")
                plot_pivot_wrapper(pw, f"Accuracies over epochs (mean ± std) - param={param_type} (all ω & ε)", fname)
                fname_subplot = os.path.join(figures_dir, f"accuracies_epochs_agg_param_{param_type}_all_pairs_subplot.png")
                plot_pivot_wrapper(pw, f"Accuracies over epochs (mean ± std) - param={param_type} (all ω & ε)", fname_subplot, subplot=True)
            
        
        # --- Accuracy per (omega, param_init_type, epsilon) ---
        # Filter out rows with missing epoch or accuracy data
        valid_combined_logs = combined_logs[combined_logs['epoch'].notna() & combined_logs['accuracy'].notna()]
        if not valid_combined_logs.empty:
            last_epoch_acc = (
                valid_combined_logs.groupby(["omega", "param_init_type", "epsilon"])[["epoch", "accuracy"]]
                .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"] if not g.empty else np.nan)
                .reset_index(name="accuracy")
            )
            # Remove any rows where accuracy is NaN
            last_epoch_acc = last_epoch_acc[last_epoch_acc['accuracy'].notna()]
        else:
            print("No valid epoch/accuracy data found for final analysis")
            last_epoch_acc = pd.DataFrame()

        if not last_epoch_acc.empty:
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
                        plt.errorbar(agg['epsilon'], agg['mean'], yerr=agg['std'].fillna(0).infer_objects(copy=False), marker='o')
                        plt.title(f"Last Epoch Accuracy vs. Epsilon (avg over omegas) - param={param_type}")
                        plt.xlabel('Epsilon')
                        plt.ylabel('Accuracy')
                        plt.savefig(os.path.join(figures_dir, f"accuracy_vs_epsilon_agg_param_{param_type}.png"))
                        plt.close()
        else:
            print("No valid accuracy data found for final plots")

    print(f"Saved figures in: {figures_dir}")

analyze_multidigit_module(RAW_DIR)
