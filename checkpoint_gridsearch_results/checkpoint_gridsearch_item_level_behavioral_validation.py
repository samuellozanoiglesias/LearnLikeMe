"""
Checkpoint Gridsearch: Item-Level Behavioral Validation across Checkpoints
============================================================================

USE:

nohup python checkpoint_gridsearch_item_level_behavioral_validation.py > checkpoint_gridsearch.log 2>&1 &

This script does NOT reimplement the analysis -- it drives
`item_level_behavioral_validation.py` (imported as a module) once per
CHECKPOINT_NAME in CHECKPOINT_NAMES below, exactly as if you had edited
CHECKPOINT_NAME by hand and re-run the script, except each run's outputs are
kept in their own, clearly named subfolder instead of overwriting each
other. Nothing in item_level_behavioral_validation.py's own pipeline is
changed by running it this way.

Pipeline
--------
1.  For every checkpoint name in CHECKPOINT_NAMES:
        - point the imported module at that checkpoint
          (ilbv.CHECKPOINT_NAME = checkpoint_name)
        - point its OUTPUT_DIR at a dedicated subfolder of
          GRIDSEARCH_OUTPUT_DIR (one folder per checkpoint)
        - run ilbv.main(), which performs (unchanged):
            * the full per-epsilon analysis (correlation_results.csv,
              model_error_rates_by_epsilon.csv, developmental_profile.png,
              scatter_best_epsilon_*.png)
            * the pooled "all epsilons as one model" analysis
              (pooled_correlation_results.csv,
              pooled_model_error_rates_mean_std.csv,
              pooled_model_error_rates_per_init.csv,
              pooled_scatter_all_epsilons.png)
2.  Re-read every checkpoint's correlation_results.csv and
    pooled_correlation_results.csv from disk and concatenate them (with a
    new `checkpoint_name` column) into two master "gridsearch" tables:
        - master_correlation_grid_per_epsilon.csv
          (checkpoint x epsilon x population x measure)
        - master_correlation_grid_pooled.csv
          (checkpoint x population x measure)
3.  For each of the three human measures (RT, zRT, ER) and each of three
    criteria -- "Kids_II", "Adults", and "both" (defined as the mean of
    |r_kids| and |r_adults| for the same checkpoint/epsilon/measure, i.e.
    the checkpoint/epsilon that is jointly best for both populations at
    once) -- find:
        - the best (checkpoint, epsilon) combination in the per-epsilon grid
        - the best checkpoint alone in the pooled grid
    and save these as best_summary_per_epsilon.csv / best_summary_pooled.csv.
4.  Plots:
        - a checkpoint x epsilon correlation heatmap per (population, measure)
          for the per-epsilon grid -- the actual "gridsearch" visualization
        - a grouped bar chart comparing the pooled |r| (Kids_II, Adults, both)
          across checkpoints, one subplot per measure

Everything is written under GRIDSEARCH_OUTPUT_DIR; per-checkpoint raw
outputs live in GRIDSEARCH_OUTPUT_DIR/<sanitized_checkpoint_name>/, and the
cross-checkpoint comparison lives in GRIDSEARCH_OUTPUT_DIR/_gridsearch_summary/.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The actual analysis pipeline -- imported, not duplicated. Importing it does
# NOT run anything (main() is only called under `if __name__ == "__main__":`
# in that file), so it's safe to import and then drive from here.
import item_level_behavioral_validation.item_level_behavioral_validation as ilbv

# --------------------------------------------------------------------------
# CONFIG -- edit these for your setup
# --------------------------------------------------------------------------

# The list of checkpoints to gridsearch over. Each must be a valid
# CHECKPOINT_NAME as used by item_level_behavioral_validation.py, i.e. it
# will be looked for (with the same trained_model.pkl fallback logic) inside
# every kept Training_<timestamp> folder for every epsilon.
CHECKPOINT_NAMES = [
    f"trained_model_checkpoint_{i}.pkl"
    for i in range(0, 2001, 10)
]

GRIDSEARCH_OUTPUT_DIR = "./checkpoint_gridsearch_results"
SUMMARY_DIR = os.path.join(GRIDSEARCH_OUTPUT_DIR, "_gridsearch_summary")
os.makedirs(GRIDSEARCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

HUMAN_MEASURES = ilbv.HUMAN_MEASURES  # ["RT", "zRT", "ER"], reuse the same list
POPULATIONS = ["Kids_II", "Adults"]


def sanitize(name: str) -> str:
    """Turn a checkpoint filename into a filesystem-friendly folder name."""
    return re.sub(r"[^\w.-]", "_", name).replace(".pkl", "")


def checkpoint_subdir(checkpoint_name: str) -> str:
    return os.path.join(GRIDSEARCH_OUTPUT_DIR, sanitize(checkpoint_name))


# --------------------------------------------------------------------------
# 1. Run the full pipeline once per checkpoint
# --------------------------------------------------------------------------

def run_all_checkpoints(checkpoint_names):
    """Run item_level_behavioral_validation.main() once per checkpoint name,
    each time redirecting its CHECKPOINT_NAME and OUTPUT_DIR so every run's
    files land in their own subfolder. Returns the list of checkpoint names
    that completed without raising."""
    completed = []
    for checkpoint_name in checkpoint_names:
        sub_dir = checkpoint_subdir(checkpoint_name)
        os.makedirs(sub_dir, exist_ok=True)

        print(f"\n{'=' * 80}\nRunning full pipeline for CHECKPOINT_NAME = "
              f"{checkpoint_name!r}\nOutput folder: {sub_dir}\n{'=' * 80}")

        # Redirect the underlying module at this checkpoint / this folder.
        # This is exactly what you'd do by hand-editing CHECKPOINT_NAME and
        # OUTPUT_DIR at the top of item_level_behavioral_validation.py.
        ilbv.CHECKPOINT_NAME = checkpoint_name
        ilbv.OUTPUT_DIR = sub_dir

        try:
            ilbv.main()
            completed.append(checkpoint_name)
        except Exception as e:
            warnings.warn(f"[{checkpoint_name}] pipeline failed entirely: {e}")

    return completed


# --------------------------------------------------------------------------
# 2. Aggregate every checkpoint's saved CSVs into master grids
# --------------------------------------------------------------------------

def load_master_grids(checkpoint_names):
    per_epsilon_rows, pooled_rows = [], []

    for checkpoint_name in checkpoint_names:
        sub_dir = checkpoint_subdir(checkpoint_name)

        corr_path = os.path.join(sub_dir, "correlation_results.csv")
        if os.path.isfile(corr_path):
            df = pd.read_csv(corr_path)
            df.insert(0, "checkpoint_name", checkpoint_name)
            per_epsilon_rows.append(df)
        else:
            warnings.warn(f"[{checkpoint_name}] missing {corr_path} -- "
                           f"excluded from the per-epsilon master grid.")

        pooled_path = os.path.join(sub_dir, "pooled_correlation_results.csv")
        if os.path.isfile(pooled_path):
            df2 = pd.read_csv(pooled_path)
            df2.insert(0, "checkpoint_name", checkpoint_name)
            pooled_rows.append(df2)
        else:
            warnings.warn(f"[{checkpoint_name}] missing {pooled_path} -- "
                           f"excluded from the pooled master grid.")

    per_epsilon_master = (pd.concat(per_epsilon_rows, ignore_index=True)
                          if per_epsilon_rows else pd.DataFrame())
    pooled_master = (pd.concat(pooled_rows, ignore_index=True)
                     if pooled_rows else pd.DataFrame())
    return per_epsilon_master, pooled_master


# --------------------------------------------------------------------------
# 3. "both populations at once" combined score + best-of tables
# --------------------------------------------------------------------------

def add_both_population_score(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Given a long table with one row per (group_cols, population, measure),
    pivot Kids_II/Adults pearson_r onto the same row and add a `both_score`
    column = mean(|r_kids|, |r_adults|) -- i.e. how good the fit is for BOTH
    populations simultaneously, not just whichever is higher."""
    pivot = df.pivot_table(
        index=group_cols + ["measure"], columns="population",
        values="pearson_r", aggfunc="first",
    ).reset_index()
    for pop in POPULATIONS:
        if pop not in pivot.columns:
            pivot[pop] = np.nan
    pivot["abs_r_kids"] = pivot["Kids_II"].abs()
    pivot["abs_r_adults"] = pivot["Adults"].abs()
    pivot["both_score"] = pivot[["abs_r_kids", "abs_r_adults"]].mean(axis=1)
    return pivot


def best_rows_by_abs_value(df: pd.DataFrame, value_col: str, group_by: str = "measure") -> pd.DataFrame:
    """For each group in `group_by`, return the full row with the largest
    |value_col|. Rows with all-NaN value_col in a group are skipped."""
    best = []
    for key, sub in df.groupby(group_by):
        sub_valid = sub.dropna(subset=[value_col])
        if sub_valid.empty:
            continue
        idx = sub_valid[value_col].abs().idxmax()
        best.append(sub_valid.loc[idx])
    return pd.DataFrame(best) if best else pd.DataFrame(columns=df.columns)


def build_best_summary(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Build one tidy "who wins" table with columns:
    measure, criterion (Kids_II / Adults / both), <group_cols...>, score
    Populations use |pearson_r| directly; 'both' uses the combined both_score.
    """
    rows = []

    for pop in POPULATIONS:
        pop_df = df[df["population"] == pop].copy()
        best = best_rows_by_abs_value(pop_df, "pearson_r", group_by="measure")
        for _, r in best.iterrows():
            row = {gc: r[gc] for gc in group_cols}
            row.update({
                "measure": r["measure"],
                "criterion": pop,
                "pearson_r": r["pearson_r"],
                "pearson_p": r["pearson_p"],
                "score_used": abs(r["pearson_r"]),
            })
            rows.append(row)

    both_pivot = add_both_population_score(df, group_cols)
    best_both = best_rows_by_abs_value(both_pivot, "both_score", group_by="measure")
    for _, r in best_both.iterrows():
        row = {gc: r[gc] for gc in group_cols}
        row.update({
            "measure": r["measure"],
            "criterion": "both",
            "pearson_r_kids": r["Kids_II"],
            "pearson_r_adults": r["Adults"],
            "score_used": r["both_score"],
        })
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Plots
# --------------------------------------------------------------------------

def plot_checkpoint_epsilon_heatmaps(per_epsilon_master: pd.DataFrame, out_dir: str):
    """One heatmap per (population, measure): rows = checkpoint, columns =
    epsilon, cell value = pearson_r. This is the actual 'gridsearch' picture:
    which (checkpoint, epsilon) combinations correlate best with humans."""
    checkpoints_sorted = sorted(
        per_epsilon_master["checkpoint_name"].unique(),
        key=_checkpoint_sort_key,
    )
    for pop in POPULATIONS:
        for measure in HUMAN_MEASURES:
            sub = per_epsilon_master[
                (per_epsilon_master["population"] == pop) & (per_epsilon_master["measure"] == measure)
            ]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="checkpoint_name", columns="epsilon", values="pearson_r")
            pivot = pivot.reindex(checkpoints_sorted)

            fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivot.shape[1]), max(4, 0.5 * pivot.shape[0])))
            vmax = np.nanmax(np.abs(pivot.to_numpy())) if np.isfinite(pivot.to_numpy()).any() else 1.0
            im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([f"{e:.2f}" for e in pivot.columns], rotation=90, fontsize=7)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_xlabel("Epsilon")
            ax.set_ylabel("Checkpoint")
            ax.set_title(f"Pearson r: model error rate vs. {pop} {measure}\n(gridsearch over checkpoint x epsilon)")
            fig.colorbar(im, ax=ax, label="Pearson r")
            fig.tight_layout()
            out_path = os.path.join(out_dir, f"heatmap_{pop}_{measure}.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)


def _checkpoint_sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def plot_pooled_comparison_bars(pooled_master: pd.DataFrame, out_path: str):
    """Grouped bar chart: one subplot per measure, bars = |r| for Kids_II,
    Adults, and the combined 'both' score, grouped by checkpoint -- summarizes
    the pooled (all-epsilons-as-one-model) comparison across checkpoints."""
    both_pivot = add_both_population_score(pooled_master, ["checkpoint_name"])
    checkpoints_sorted = sorted(pooled_master["checkpoint_name"].unique(), key=_checkpoint_sort_key)

    fig, axes = plt.subplots(1, len(HUMAN_MEASURES), figsize=(6 * len(HUMAN_MEASURES), 5), sharey=True)
    if len(HUMAN_MEASURES) == 1:
        axes = [axes]

    width = 0.25
    x = np.arange(len(checkpoints_sorted))

    for ax, measure in zip(axes, HUMAN_MEASURES):
        sub = both_pivot[both_pivot["measure"] == measure].set_index("checkpoint_name").reindex(checkpoints_sorted)
        ax.bar(x - width, sub["abs_r_kids"], width, label="Kids_II |r|")
        ax.bar(x, sub["abs_r_adults"], width, label="Adults |r|")
        ax.bar(x + width, sub["both_score"], width, label="Both (mean |r|)")
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints_sorted, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Pooled |r| vs. {measure}")
        ax.set_ylabel("|Pearson r|")
        ax.axhline(0, color="grey", linewidth=0.8)

    axes[0].legend()
    fig.suptitle("Pooled (all-epsilons-as-one-model) checkpoint comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    completed = run_all_checkpoints(CHECKPOINT_NAMES)
    if not completed:
        print("No checkpoint finished the pipeline successfully -- nothing to aggregate.")
        return

    per_epsilon_master, pooled_master = load_master_grids(completed)

    if per_epsilon_master.empty and pooled_master.empty:
        print("No correlation results could be read back from any checkpoint's output "
              "folder -- nothing to aggregate.")
        return

    # --- Save the raw master grids ---
    if not per_epsilon_master.empty:
        per_epsilon_master_path = os.path.join(SUMMARY_DIR, "master_correlation_grid_per_epsilon.csv")
        per_epsilon_master.to_csv(per_epsilon_master_path, index=False)
    else:
        per_epsilon_master_path = None
        warnings.warn("Per-epsilon master grid is empty -- skipping per-epsilon best-of "
                       "analysis and heatmaps.")

    if not pooled_master.empty:
        pooled_master_path = os.path.join(SUMMARY_DIR, "master_correlation_grid_pooled.csv")
        pooled_master.to_csv(pooled_master_path, index=False)
    else:
        pooled_master_path = None
        warnings.warn("Pooled master grid is empty -- skipping pooled best-of analysis "
                       "and comparison bar chart.")

    # --- Best-of summaries ---
    if per_epsilon_master is not None and not per_epsilon_master.empty:
        best_per_epsilon = build_best_summary(per_epsilon_master, group_cols=["checkpoint_name", "epsilon"])
        best_per_epsilon_path = os.path.join(SUMMARY_DIR, "best_summary_per_epsilon.csv")
        best_per_epsilon.to_csv(best_per_epsilon_path, index=False)
    else:
        best_per_epsilon, best_per_epsilon_path = pd.DataFrame(), None

    if pooled_master is not None and not pooled_master.empty:
        best_pooled = build_best_summary(pooled_master, group_cols=["checkpoint_name"])
        best_pooled_path = os.path.join(SUMMARY_DIR, "best_summary_pooled.csv")
        best_pooled.to_csv(best_pooled_path, index=False)
    else:
        best_pooled, best_pooled_path = pd.DataFrame(), None

    # --- Plots ---
    heatmap_paths = []
    if per_epsilon_master is not None and not per_epsilon_master.empty:
        plot_checkpoint_epsilon_heatmaps(per_epsilon_master, SUMMARY_DIR)
        heatmap_paths = [
            os.path.join(SUMMARY_DIR, f"heatmap_{pop}_{measure}.png")
            for pop in POPULATIONS for measure in HUMAN_MEASURES
        ]

    bar_path = None
    if pooled_master is not None and not pooled_master.empty:
        bar_path = os.path.join(SUMMARY_DIR, "pooled_comparison_bars.png")
        plot_pooled_comparison_bars(pooled_master, bar_path)

    # --- Console summary ---
    print("\n\n" + "=" * 80)
    print("GRIDSEARCH SUMMARY")
    print("=" * 80)

    if not best_per_epsilon.empty:
        print("\n--- Best (checkpoint, epsilon) per measure and criterion "
              "(per-epsilon analysis) ---")
        print(best_per_epsilon.to_string(index=False))

    if not best_pooled.empty:
        print("\n--- Best checkpoint per measure and criterion "
              "(pooled all-epsilons-as-one-model analysis) ---")
        print(best_pooled.to_string(index=False))

    print("\nSaved:")
    for p in [per_epsilon_master_path, pooled_master_path, best_per_epsilon_path,
              best_pooled_path, bar_path, *heatmap_paths]:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
