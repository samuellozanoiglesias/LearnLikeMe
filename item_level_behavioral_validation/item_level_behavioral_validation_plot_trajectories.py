"""
Developmental Trajectory Plots: Kids vs. Adults correlation over training
============================================================================

USE:

nohup python item_level_behavioral_validation_plot_trajectories.py > item_level_behavioral_validation_plot_trajectories.log 2>&1 &

USE WITH CHECKPOINT RANGE RESTRICTION (optional):

nohup python item_level_behavioral_validation_plot_trajectories.py 200 1000 > item_level_behavioral_validation_plot_trajectories.log 2>&1 &

This script does NOT re-run the gridsearch and does NOT re-run
item_level_behavioral_validation.py. It only reads back the two master
tables that checkpoint_gridsearch_item_level_behavioral_validation.py
already wrote to disk:

    GRIDSEARCH_OUTPUT_DIR/_gridsearch_summary/master_correlation_grid_pooled.csv
    GRIDSEARCH_OUTPUT_DIR/_gridsearch_summary/master_correlation_grid_per_epsilon.csv

and turns them into a training-trajectory picture: for each of the three
human measures (RT, zRT, ER), how does the model-vs-human correlation (r)
change across training checkpoints, separately for Kids_II and Adults?

The motivating hypothesis (from the reviewer comment) is a developmental
crossover: if the model captures the development of human addition
performance, correlations with children should be relatively higher at
earlier checkpoints, and correlations with adults should be relatively
higher at later checkpoints.

Two versions of the trajectory are plotted, since they answer slightly
different questions:

1. POOLED trajectory (main answer to the reviewer's question)
   -- one r-value per (checkpoint, population, measure), from the "all
   epsilons treated as one model" pooled analysis. This is the most direct
   read of "does the correlation change with training", without also
   letting epsilon change underneath it.

2. BEST-EPSILON-PER-CHECKPOINT trajectory (robustness check)
   -- for each (checkpoint, population, measure), take whichever epsilon in
   the per-epsilon grid gives the largest |r| at that checkpoint, and plot
   that r. This asks: even if we let the "best" noise level float at every
   point in training, does the same kids-early / adults-late pattern show
   up? If trajectory 1 and trajectory 2 tell the same story, that's good
   evidence the crossover isn't an artifact of a particular epsilon choice.

Optionally restrict to a checkpoint range
------------------------------------------
Set MIN_CHECKPOINT and/or MAX_CHECKPOINT below (either can be left as None
to leave that side unbounded) to only look at training steps within
[MIN_CHECKPOINT, MAX_CHECKPOINT]. This is useful e.g. to zoom in on the
early part of training where the crossover is expected, or to exclude a
noisy tail. Whenever either bound is set, every output filename gets a
"_{min}_{max}" suffix reflecting the checkpoint range actually plotted
(taken from the bound you set, or from the min/max training_step present
in the filtered data when a bound is left as None), so range-restricted
runs never overwrite the full-range outputs.

Two tidy CSVs are also written out (with a `training_step` column appended,
extracted from the checkpoint filename) so these can be opened directly in
Excel alongside the original behavioral data files.

Outputs
-------
DEV_TRAJECTORY_OUTPUT_DIR/
    developmental_trajectory_pooled[_{min}_{max}].csv
    developmental_trajectory_best_epsilon[_{min}_{max}].csv
    trajectory_pooled_RT_zRT_ER[_{min}_{max}].png
    trajectory_best_epsilon_RT_zRT_ER[_{min}_{max}].png
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONFIG -- edit these for your setup
# --------------------------------------------------------------------------

# Must match GRIDSEARCH_OUTPUT_DIR in checkpoint_gridsearch_item_level_
# behavioral_validation.py -- this script only reads what that one wrote.
GRIDSEARCH_OUTPUT_DIR = "./checkpoint_gridsearch_results"
SUMMARY_DIR = os.path.join(GRIDSEARCH_OUTPUT_DIR, "_gridsearch_summary")

POOLED_GRID_PATH = os.path.join(SUMMARY_DIR, "master_correlation_grid_pooled.csv")
PER_EPSILON_GRID_PATH = os.path.join(SUMMARY_DIR, "master_correlation_grid_per_epsilon.csv")

DEV_TRAJECTORY_OUTPUT_DIR = "./developmental_trajectory_results"
os.makedirs(DEV_TRAJECTORY_OUTPUT_DIR, exist_ok=True)

# Optional checkpoint range to restrict the plot/CSVs to. Leave either (or
# both) as None to leave that side unbounded (i.e. use the full range of
# whatever training steps are present in the data). Both are inclusive.
#   e.g. MIN_CHECKPOINT, MAX_CHECKPOINT = 0, 600     -- only steps 0-600
#   e.g. MIN_CHECKPOINT, MAX_CHECKPOINT = 500, None  -- everything from 500 on
#   e.g. MIN_CHECKPOINT, MAX_CHECKPOINT = None, None -- full range (default)
import sys
MIN_CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else None
MAX_CHECKPOINT = sys.argv[2] if len(sys.argv) > 2 else None
if MIN_CHECKPOINT is not None:
    MIN_CHECKPOINT = int(MIN_CHECKPOINT)
if MAX_CHECKPOINT is not None:
    MAX_CHECKPOINT = int(MAX_CHECKPOINT)

HUMAN_MEASURES = ["RT", "zRT", "ER"]
POPULATIONS = ["Kids_II", "Adults"]
POP_COLORS = {"Kids_II": "tab:orange", "Adults": "tab:blue"}
SIG_ALPHA = 0.05  # threshold used only to mark points as filled vs. hollow


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def extract_training_step(checkpoint_name: str) -> int:
    """Pull the integer batch/checkpoint number out of a checkpoint filename,
    e.g. 'trained_model_checkpoint_600.pkl' -> 600. Falls back to -1 (and
    warns) if no number is found, so such rows sort first and are easy to
    spot rather than silently dropped."""
    m = re.search(r"(\d+)", checkpoint_name)
    if m is None:
        warnings.warn(f"Could not find a training step number in "
                       f"{checkpoint_name!r}; using -1.")
        return -1
    return int(m.group(1))


def load_grid(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {path}. This script expects the gridsearch "
            f"script to already have been run once, producing the master "
            f"grid CSVs under GRIDSEARCH_OUTPUT_DIR/_gridsearch_summary/. "
            f"Check that GRIDSEARCH_OUTPUT_DIR here matches the one used "
            f"in the gridsearch script."
        )
    df = pd.read_csv(path)
    df["training_step"] = df["checkpoint_name"].apply(extract_training_step)
    return df.sort_values("training_step")


def filter_checkpoint_range(df: pd.DataFrame, min_checkpoint, max_checkpoint,
                             source_label: str) -> pd.DataFrame:
    """Restrict df to training_step in [min_checkpoint, max_checkpoint]
    (either bound optional). Warns (but does not raise) if the filter
    leaves nothing behind, since that's most likely a bounds typo."""
    if min_checkpoint is None and max_checkpoint is None:
        return df
    mask = pd.Series(True, index=df.index)
    if min_checkpoint is not None:
        mask &= df["training_step"] >= min_checkpoint
    if max_checkpoint is not None:
        mask &= df["training_step"] <= max_checkpoint
    filtered = df[mask]
    if filtered.empty:
        warnings.warn(
            f"[{source_label}] Filtering to checkpoint range "
            f"[{min_checkpoint}, {max_checkpoint}] left no rows -- check "
            f"MIN_CHECKPOINT/MAX_CHECKPOINT against the training_step "
            f"values actually present in the data."
        )
    return filtered


def range_suffix(traj: pd.DataFrame, min_checkpoint, max_checkpoint) -> str:
    """Build the '_{min}_{max}' filename suffix for a checkpoint-restricted
    run. Returns '' when neither bound was set (full-range run, unchanged
    filenames). When a bound is None but the other isn't, the unbounded
    side falls back to the min/max training_step actually present in the
    (already-filtered) data, so the suffix always reflects what was really
    plotted."""
    if min_checkpoint is None and max_checkpoint is None:
        return ""
    if traj.empty or "training_step" not in traj.columns:
        used_min = min_checkpoint if min_checkpoint is not None else "na"
        used_max = max_checkpoint if max_checkpoint is not None else "na"
    else:
        used_min = min_checkpoint if min_checkpoint is not None else int(traj["training_step"].min())
        used_max = max_checkpoint if max_checkpoint is not None else int(traj["training_step"].max())
    return f"_{used_min}_{used_max}"


# --------------------------------------------------------------------------
# Build the two trajectory tables
# --------------------------------------------------------------------------

def build_pooled_trajectory(pooled_grid: pd.DataFrame) -> pd.DataFrame:
    """One row per (training_step, population, measure) straight from the
    pooled (all-epsilons-as-one-model) grid -- no epsilon selection."""
    cols = ["checkpoint_name", "training_step", "population", "measure",
            "pearson_r", "pearson_p"]
    cols = [c for c in cols if c in pooled_grid.columns]
    traj = pooled_grid[cols].drop_duplicates()
    return traj.sort_values(["measure", "population", "training_step"])


def build_best_epsilon_trajectory(per_epsilon_grid: pd.DataFrame) -> pd.DataFrame:
    """For each (checkpoint, population, measure), keep only the row with the
    largest |pearson_r| across epsilons -- i.e. the best-fitting noise level
    at that point in training."""
    rows = []
    group_cols = ["checkpoint_name", "training_step", "population", "measure"]
    for _, sub in per_epsilon_grid.groupby(group_cols):
        sub_valid = sub.dropna(subset=["pearson_r"])
        if sub_valid.empty:
            continue
        best_idx = sub_valid["pearson_r"].abs().idxmax()
        rows.append(sub_valid.loc[best_idx])
    traj = pd.DataFrame(rows)
    keep_cols = group_cols + ["epsilon", "pearson_r", "pearson_p"]
    keep_cols = [c for c in keep_cols if c in traj.columns]
    traj = traj[keep_cols]
    return traj.sort_values(["measure", "population", "training_step"])


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_trajectory(traj: pd.DataFrame, title: str, out_path: str, epsilon_in_label: bool = False):
    """3-panel line plot (one panel per measure), each panel showing the
    Kids_II and Adults r-value trajectory across training_step. Points where
    pearson_p < SIG_ALPHA are drawn as filled markers; non-significant points
    are drawn hollow, so the significance pattern is visible at a glance
    without cluttering the plot with extra annotations."""
    fig, axes = plt.subplots(1, len(HUMAN_MEASURES), figsize=(6 * len(HUMAN_MEASURES), 4.5), sharey=True)
    if len(HUMAN_MEASURES) == 1:
        axes = [axes]

    for ax, measure in zip(axes, HUMAN_MEASURES):
        sub_measure = traj[traj["measure"] == measure]
        for pop in POPULATIONS:
            sub = sub_measure[sub_measure["population"] == pop].sort_values("training_step")
            if sub.empty:
                continue
            color = POP_COLORS.get(pop, None)
            ax.plot(sub["training_step"], sub["pearson_r"], "-", color=color,
                     linewidth=1.5, label=f"{pop}", zorder=2)

            if "pearson_p" in sub.columns:
                sig = sub["pearson_p"] < SIG_ALPHA
                ax.scatter(sub.loc[sig, "training_step"], sub.loc[sig, "pearson_r"],
                            facecolor=color, edgecolor=color, s=28, zorder=3)
                ax.scatter(sub.loc[~sig, "training_step"], sub.loc[~sig, "pearson_r"],
                            facecolor="white", edgecolor=color, s=28, zorder=3)
            else:
                ax.scatter(sub["training_step"], sub["pearson_r"], color=color, s=20, zorder=3)

        ax.axhline(0, color="grey", linewidth=0.8, zorder=1)
        ax.set_title(measure)
        ax.set_xlabel("Training step (checkpoint)")

    axes[0].set_ylabel("Pearson r (model error rate vs. human measure)")
    axes[0].legend(loc="best", fontsize=9)

    subtitle = title
    if epsilon_in_label:
        subtitle += "\n(best-fitting epsilon at each checkpoint; filled = p < %.2f)" % SIG_ALPHA
    else:
        subtitle += "\n(pooled across epsilons; filled = p < %.2f)" % SIG_ALPHA
    fig.suptitle(subtitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    pooled_grid = load_grid(POOLED_GRID_PATH)
    per_epsilon_grid = load_grid(PER_EPSILON_GRID_PATH)

    # Apply the optional checkpoint-range filter up front, on the raw grids,
    # so both the CSVs and the plots below only ever see the restricted range.
    pooled_grid = filter_checkpoint_range(pooled_grid, MIN_CHECKPOINT, MAX_CHECKPOINT,
                                           source_label="pooled grid")
    per_epsilon_grid = filter_checkpoint_range(per_epsilon_grid, MIN_CHECKPOINT, MAX_CHECKPOINT,
                                                source_label="per-epsilon grid")

    # --- Pooled trajectory (main plot) ---
    pooled_traj = build_pooled_trajectory(pooled_grid)
    suffix = range_suffix(pooled_traj, MIN_CHECKPOINT, MAX_CHECKPOINT)

    pooled_traj_path = os.path.join(DEV_TRAJECTORY_OUTPUT_DIR, f"developmental_trajectory_pooled{suffix}.csv")
    pooled_traj.to_csv(pooled_traj_path, index=False)

    pooled_plot_path = os.path.join(DEV_TRAJECTORY_OUTPUT_DIR, f"trajectory_pooled_RT_zRT_ER{suffix}.png")
    plot_trajectory(
        pooled_traj,
        title="Kids vs. Adults correlation over training (pooled across epsilons)",
        out_path=pooled_plot_path,
        epsilon_in_label=False,
    )

    # --- Best-epsilon-per-checkpoint trajectory (robustness check) ---
    best_eps_traj = build_best_epsilon_trajectory(per_epsilon_grid)
    suffix_best = range_suffix(best_eps_traj, MIN_CHECKPOINT, MAX_CHECKPOINT)

    best_eps_traj_path = os.path.join(DEV_TRAJECTORY_OUTPUT_DIR,
                                       f"developmental_trajectory_best_epsilon{suffix_best}.csv")
    best_eps_traj.to_csv(best_eps_traj_path, index=False)

    best_eps_plot_path = os.path.join(DEV_TRAJECTORY_OUTPUT_DIR,
                                       f"trajectory_best_epsilon_RT_zRT_ER{suffix_best}.png")
    plot_trajectory(
        best_eps_traj,
        title="Kids vs. Adults correlation over training (best-fitting epsilon per checkpoint)",
        out_path=best_eps_plot_path,
        epsilon_in_label=True,
    )

    print("Saved:")
    for p in [pooled_traj_path, pooled_plot_path, best_eps_traj_path, best_eps_plot_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()