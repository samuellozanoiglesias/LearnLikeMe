# USE: nohup python analyze_test_decision_module.py WI argmax > logs_analysis_test_decision.out 2>&1 &

import os
import sys
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit, Local or Lenovo
PARAM_TYPE = str(sys.argv[1]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[2]).lower()  # 'argmax' or 'vector' version of the decision module

# --- Paths ---
if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{PARAM_TYPE}/{MODEL_TYPE}_version"
FOLDER_DIR = f"{RAW_DIR}/tests"
SAVE_DIR = f"{RAW_DIR}/figures_tests"

os.makedirs(SAVE_DIR, exist_ok=True)

# Utility: parse epsilon from filename like tests_WI_0.05.csv
def parse_epsilon_from_name(name: str):
    m = re.search(r"(?:tests)_[A-Za-z]+_([0-9]+\.?[0-9]*)", name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # try a looser match
    m = re.search(r"([0-9]+\.[0-9]+)", name)
    return float(m.group(1)) if m else None

def find_all_test_csvs(tests_dir: str, param: str):
    pattern = os.path.join(tests_dir,  f'tests_{param}_*.csv')
    paths = glob.glob(pattern)
    return sorted(paths)

def aggregate_files(paths):
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
            continue

        if 'epsilon' not in df.columns:
            eps = parse_epsilon_from_name(os.path.basename(p))
            df['epsilon'] = eps

        # ensure numeric epsilon
        df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')
        dfs.append(df)

    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined

def plot_metrics_by_epsilon(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sns.set(style='whitegrid')

    # Find all accuracy columns (suffix '_accuracy')
    accuracy_cols = [c for c in df.columns if c.endswith('_accuracy')]
    if not accuracy_cols:
        print('No accuracy columns found to plot.')
        return

    # Group by epsilon and compute mean/std
    stats = df.groupby('epsilon')[accuracy_cols].agg(['mean', 'std', 'count'])

    # For each accuracy column create mean +/- std plot
    for col in accuracy_cols:
        if ('mean' in stats[col].columns):
            means = stats[col]['mean']
            stds = stats[col]['std']
            eps = means.index.values

            plt.figure(figsize=(8, 5))
            plt.plot(eps, means, marker='o', label='mean')
            plt.fill_between(eps, means - stds, means + stds, alpha=0.25, label='±1 std')
            plt.xlabel('Epsilon')
            plt.ylabel(col)
            plt.title(col.replace('_', ' ').title())
            plt.legend()
            plt.tight_layout()
            outp = os.path.join(out_dir, f'{col}_vs_epsilon.png')
            plt.savefig(outp)
            plt.close()
            print(f'Wrote {outp}')

    # Comprehensive multi-line plot (subset top 6 metrics)
    top_cols = accuracy_cols[:6]
    plt.figure(figsize=(10, 6))
    for col in top_cols:
        # plot mean
        means = stats[col]['mean']
        eps = means.index.values
        plt.plot(eps, means, marker='o', label=col.replace('_', ' '))

    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('Comprehensive Accuracy by Epsilon')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    outp = os.path.join(out_dir, 'comprehensive_accuracy_vs_epsilon.png')
    plt.savefig(outp)
    plt.close()
    print(f'Wrote {outp}')


def plot_metrics_by_omega(df: pd.DataFrame, out_dir: str):
    """Same as plot_metrics_by_epsilon but grouped by omega."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set(style='whitegrid')

    accuracy_cols = [c for c in df.columns if c.endswith('_accuracy')]
    if not accuracy_cols:
        print('No accuracy columns found to plot by omega.')
        return

    stats = df.groupby('omega')[accuracy_cols].agg(['mean', 'std', 'count'])

    for col in accuracy_cols:
        if ('mean' in stats[col].columns):
            means = stats[col]['mean']
            stds = stats[col]['std']
            oms = means.index.values

            plt.figure(figsize=(8, 5))
            plt.plot(oms, means, marker='o', label='mean')
            plt.fill_between(oms, means - stds, means + stds, alpha=0.25, label='±1 std')
            plt.xlabel('Omega')
            plt.ylabel(col)
            plt.title(col.replace('_', ' ').title())
            plt.legend()
            plt.tight_layout()
            outp = os.path.join(out_dir, f'{col}_vs_omega.png')
            plt.savefig(outp)
            plt.close()
            print(f'Wrote {outp}')

    # Comprehensive multi-line plot (subset top 6 metrics)
    top_cols = accuracy_cols[:6]
    plt.figure(figsize=(10, 6))
    for col in top_cols:
        means = stats[col]['mean']
        oms = means.index.values
        plt.plot(oms, means, marker='o', label=col.replace('_', ' '))

    plt.xlabel('Omega')
    plt.ylabel('Accuracy')
    plt.title('Comprehensive Accuracy by Omega')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    outp = os.path.join(out_dir, 'comprehensive_accuracy_vs_omega.png')
    plt.savefig(outp)
    plt.close()
    print(f'Wrote {outp}')


def ensure_checkpoint_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Try to coerce checkpoint to integer where possible
    if 'checkpoint' in df.columns:
        df['checkpoint'] = pd.to_numeric(df['checkpoint'], errors='coerce')
        df['checkpoint'] = df['checkpoint'].fillna(0).astype(int)
    else:
        df['checkpoint'] = 0
    return df


def add_no_carry_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Create no-carry counts/totals and accuracies for large and small test pairs
    # expected existing columns:
    # test_pairs_large_count, test_pairs_large_total, test_pairs_carry_large_count, test_pairs_carry_large_total
    # test_pairs_small_count, test_pairs_small_total, test_pairs_carry_small_count, test_pairs_carry_small_total
    def safe_col(col):
        return col if col in df.columns else None

    # Only add if the large/small columns exist
    for size in ['large', 'small']:
        count_col = f'test_pairs_{size}_count'
        total_col = f'test_pairs_{size}_total'
        carry_count_col = f'test_pairs_carry_{size}_count'
        carry_total_col = f'test_pairs_carry_{size}_total'

        no_carry_count_col = f'test_pairs_no_carry_{size}_count'
        no_carry_total_col = f'test_pairs_no_carry_{size}_total'
        no_carry_acc_col = f'test_pairs_no_carry_{size}_accuracy'

        if all(c in df.columns for c in [count_col, total_col, carry_count_col, carry_total_col]):
            df[no_carry_count_col] = df[count_col] - df[carry_count_col]
            df[no_carry_total_col] = df[total_col] - df[carry_total_col]
            # avoid division by zero
            df[no_carry_acc_col] = df.apply(lambda r: round(100 * r[no_carry_count_col] / r[no_carry_total_col], 2)
                                            if r[no_carry_total_col] and r[no_carry_total_col] > 0 else None, axis=1)
        else:
            print(f"Warning: missing columns for {size} no-carry computation; skipping {size} no-carry metrics.")

    return df


def plot_checkpoint_comparison_per_combo(df: pd.DataFrame, out_dir: str):
    """For every (epsilon, omega) pair, plot accuracy over checkpoint with 4 lines:
    test_pairs_carry_large_accuracy, test_pairs_carry_small_accuracy,
    test_pairs_no_carry_large_accuracy, test_pairs_no_carry_small_accuracy
    """
    os.makedirs(out_dir, exist_ok=True)

    required_cols = [
        'test_pairs_carry_large_accuracy', 'test_pairs_carry_small_accuracy',
        'test_pairs_no_carry_large_accuracy', 'test_pairs_no_carry_small_accuracy',
        'epsilon', 'omega', 'checkpoint'
    ]

    # Drop rows missing the required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print('Skipping per-combo checkpoint plots, missing columns:', missing)
        return

    combos = df[['epsilon', 'omega']].drop_duplicates()
    for _, row in combos.iterrows():
        eps = row['epsilon']
        om = row['omega']
        sub = df[(df['epsilon'] == eps) & (df['omega'] == om)].copy()
        if sub.empty:
            continue

        # group by checkpoint and compute mean accuracy
        grp = sub.groupby('checkpoint')[[
            'test_pairs_carry_large_accuracy', 'test_pairs_carry_small_accuracy',
            'test_pairs_no_carry_large_accuracy', 'test_pairs_no_carry_small_accuracy'
        ]].mean()

        plt.figure(figsize=(8, 5))
        plt.plot(grp.index, grp['test_pairs_carry_large_accuracy'], marker='o', label='carry large')
        plt.plot(grp.index, grp['test_pairs_carry_small_accuracy'], marker='o', label='carry small')
        plt.plot(grp.index, grp['test_pairs_no_carry_large_accuracy'], marker='o', label='no carry large')
        plt.plot(grp.index, grp['test_pairs_no_carry_small_accuracy'], marker='o', label='no carry small')
        plt.xlabel('Checkpoint')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Eps={eps} Omega={om} — Accuracy vs Checkpoint')
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(out_dir, f'checkpoint_cmp_eps_{eps}_omega_{om}.png')
        plt.savefig(fname)
        plt.close()
        print(f'Wrote {fname}')

# Aggregated over all combos: mean accuracy per checkpoint (averaged over epsilon/omega)
def plot_checkpoint_comparison_aggregated(df: pd.DataFrame, out_dir: str):
    metrics = [
        'test_pairs_carry_large_accuracy', 'test_pairs_carry_small_accuracy',
        'test_pairs_no_carry_large_accuracy', 'test_pairs_no_carry_small_accuracy'
    ]

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print('Skipping aggregated checkpoint plot, missing metrics:', missing)
        return

    grp = df.groupby('checkpoint')[metrics].mean()
    if grp.empty:
        print('No data for aggregated checkpoint comparison.')
        return

    plt.figure(figsize=(9, 6))
    plt.plot(grp.index, grp[metrics[0]], marker='o', label='carry large')
    plt.plot(grp.index, grp[metrics[1]], marker='o', label='carry small')
    plt.plot(grp.index, grp[metrics[2]], marker='o', label='no carry large')
    plt.plot(grp.index, grp[metrics[3]], marker='o', label='no carry small')
    plt.xlabel('Checkpoint')
    plt.ylabel('Mean Accuracy (%)')
    plt.title('Aggregated Accuracy vs Checkpoint (mean over all epsilons & omegas)')
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, 'checkpoint_cmp_aggregated_all_combos.png')
    plt.savefig(fname)
    plt.close()
    print(f'Wrote {fname}')


def plot_checkpoint_comparison_by_epsilon(df: pd.DataFrame, out_dir: str):
    """For each epsilon, aggregate over omegas and plot mean accuracy vs checkpoint."""
    metrics = [
        'test_pairs_carry_large_accuracy', 'test_pairs_carry_small_accuracy',
        'test_pairs_no_carry_large_accuracy', 'test_pairs_no_carry_small_accuracy'
    ]

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print('Skipping per-epsilon aggregated checkpoint plots, missing metrics:', missing)
        return

    if 'epsilon' not in df.columns:
        print('No epsilon column found; skipping per-epsilon aggregated checkpoint plots.')
        return

    for eps in sorted(df['epsilon'].dropna().unique()):
        sub = df[df['epsilon'] == eps]
        if sub.empty:
            continue
        grp = sub.groupby('checkpoint')[metrics].mean()
        if grp.empty:
            continue

        plt.figure(figsize=(9, 6))
        plt.plot(grp.index, grp[metrics[0]], marker='o', label='carry large')
        plt.plot(grp.index, grp[metrics[1]], marker='o', label='carry small')
        plt.plot(grp.index, grp[metrics[2]], marker='o', label='no carry large')
        plt.plot(grp.index, grp[metrics[3]], marker='o', label='no carry small')
        plt.xlabel('Checkpoint')
        plt.ylabel('Mean Accuracy (%)')
        plt.title(f'Aggregated Accuracy vs Checkpoint (eps={eps}; mean over omegas)')
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(out_dir, f'checkpoint_cmp_aggregated_eps_{str(eps).replace(".","_")}.png')
        plt.savefig(fname)
        plt.close()
        print(f'Wrote {fname}')


def plot_checkpoint_comparison_by_omega(df: pd.DataFrame, out_dir: str):
    """For each omega, aggregate over epsilons and plot mean accuracy vs checkpoint."""
    metrics = [
        'test_pairs_carry_large_accuracy', 'test_pairs_carry_small_accuracy',
        'test_pairs_no_carry_large_accuracy', 'test_pairs_no_carry_small_accuracy'
    ]

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print('Skipping per-omega aggregated checkpoint plots, missing metrics:', missing)
        return

    if 'omega' not in df.columns:
        print('No omega column found; skipping per-omega aggregated checkpoint plots.')
        return

    for om in sorted(df['omega'].dropna().unique()):
        sub = df[df['omega'] == om]
        if sub.empty:
            continue
        grp = sub.groupby('checkpoint')[metrics].mean()
        if grp.empty:
            continue

        plt.figure(figsize=(9, 6))
        plt.plot(grp.index, grp[metrics[0]], marker='o', label='carry large')
        plt.plot(grp.index, grp[metrics[1]], marker='o', label='carry small')
        plt.plot(grp.index, grp[metrics[2]], marker='o', label='no carry large')
        plt.plot(grp.index, grp[metrics[3]], marker='o', label='no carry small')
        plt.xlabel('Checkpoint')
        plt.ylabel('Mean Accuracy (%)')
        plt.title(f'Aggregated Accuracy vs Checkpoint (omega={om}; mean over epsilons)')
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(out_dir, f'checkpoint_cmp_aggregated_omega_{str(om).replace(".","_")}.png')
        plt.savefig(fname)
        plt.close()
        print(f'Wrote {fname}')

def plot_cross_comparisons(df: pd.DataFrame, out_dir: str):
    """Create comparison plots:
    - For each omega: for the 4 metrics, plot accuracy vs checkpoint with one line per epsilon
    - For each epsilon: for the 4 metrics, plot accuracy vs checkpoint with one line per omega
    """
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        'test_pairs_carry_large_accuracy', 'test_pairs_carry_small_accuracy',
        'test_pairs_no_carry_large_accuracy', 'test_pairs_no_carry_small_accuracy'
    ]

    # Ensure omega and epsilon exist
    if 'omega' not in df.columns:
        print('No omega column found; skipping cross-comparisons by omega.')
        return

    for om in sorted(df['omega'].dropna().unique()):
        sub_om = df[df['omega'] == om].copy()
        if sub_om.empty:
            continue

        for metric in metrics:
            if metric not in sub_om.columns:
                continue

            plt.figure(figsize=(8, 5))
            for eps in sorted(sub_om['epsilon'].dropna().unique()):
                sub = sub_om[sub_om['epsilon'] == eps]
                grp = sub.groupby('checkpoint')[metric].mean()
                if grp.empty:
                    continue
                plt.plot(grp.index, grp.values, marker='o', label=f'eps={eps}')

            plt.xlabel('Checkpoint')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Omega={om} — {metric} vs Checkpoint (eps lines)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fname = os.path.join(out_dir, f'omega_{om}_{metric}_eps_compare.png')
            plt.savefig(fname)
            plt.close()
            print(f'Wrote {fname}')

    # Now for each epsilon compare omegas
    for eps in sorted(df['epsilon'].dropna().unique()):
        sub_eps = df[df['epsilon'] == eps].copy()
        if sub_eps.empty:
            continue

        for metric in metrics:
            if metric not in sub_eps.columns:
                continue

            plt.figure(figsize=(8, 5))
            for om in sorted(sub_eps['omega'].dropna().unique()):
                sub = sub_eps[sub_eps['omega'] == om]
                grp = sub.groupby('checkpoint')[metric].mean()
                if grp.empty:
                    continue
                plt.plot(grp.index, grp.values, marker='o', label=f'omega={om}')

            plt.xlabel('Checkpoint')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Eps={eps} — {metric} vs Checkpoint (omega lines)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fname = os.path.join(out_dir, f'eps_{eps}_{metric}_omega_compare.png')
            plt.savefig(fname)
            plt.close()
            print(f'Wrote {fname}')


csv_paths = find_all_test_csvs(FOLDER_DIR, PARAM_TYPE)
if not csv_paths:
    print('No test CSVs found for param', PARAM_TYPE)

print(f'Found {len(csv_paths)} CSVs. Aggregating...')
combined = aggregate_files(csv_paths)

if combined is None:
    print('No CSVs could be read. Exiting.')

mega_csv = os.path.join(RAW_DIR, f'all_eps_results_{PARAM_TYPE}.csv')
combined.to_csv(mega_csv, index=False)
print(f'Wrote CSV: {mega_csv} ({len(combined)} rows)')

# Plot grouped metrics
plot_metrics_by_epsilon(combined, SAVE_DIR)
plot_metrics_by_omega(combined, SAVE_DIR)

# Ensure checkpoint numeric and add derived no-carry metrics
combined = ensure_checkpoint_numeric(combined)
combined = add_no_carry_metrics(combined)

# Per (epsilon,omega) checkpoint plots with four lines
plot_checkpoint_comparison_per_combo(combined, SAVE_DIR)
plot_checkpoint_comparison_aggregated(combined, SAVE_DIR)
plot_checkpoint_comparison_by_epsilon(combined, SAVE_DIR)
plot_checkpoint_comparison_by_omega(combined, SAVE_DIR)

# Cross comparisons: fixed omega (multiple epsilons) and fixed epsilon (multiple omegas)
#plot_cross_comparisons(combined, SAVE_DIR)

print("Done.")
