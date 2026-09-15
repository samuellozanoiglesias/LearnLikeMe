"""
Generate the final "Item-Level Behavioral Validation" figure(s) for the
paper -- POOLED (children + adults averaged) version.

This is a derivative of paper_figure_item_level_validation.py, built to
fill in the four placeholders left in the manuscript:

    [PLACEHOLDER: pooled zRT correlation,        r = .XX, p < .001]
    [PLACEHOLDER: pooled raw-RT correlation,     r = .XX, p < .001]
    [PLACEHOLDER: pooled error-rate correlation, r = .XX, p < .001]
    [PLACEHOLDER: pooled epsilon-robustness range]

Differences from the two-population script:

  (a) The human item-level measures (zRT, RT, and error rate/ER, if an ER
      column can be found) are averaged across children and adults for
      each item -- one pooled human value per item -- instead of being
      kept as two separate series.
  (b) The Pearson correlation of pooled-model error rate against each
      pooled human measure is (re)computed from scratch and printed in a
      copy-pasteable form.
  (c) Figure~\\ref{fig:item_level_validation} is regenerated as a
      SINGLE-GROUP scatter + regression plot (no children/adults split),
      in black/grey ("bw"), written to
      Figures/item_level_validation_RT_pooled_bw.png (RT) and the
      equivalent zRT file -- matching the filename the manuscript
      currently points at as a placeholder.

On the epsilon-robustness range: that number needs the per-epsilon /
per-initialization correlation table (pooled_correlation_results.csv)
produced upstream by item_level_behavioral_validation.py. The exact
column names in that file weren't available when this script was
written, so the loader below is defensive: it reads the file if present,
prints the columns it actually finds, and only computes the range if it
can identify something that looks like an epsilon/omega column and an r
column. If it can't, it tells you what to check instead of guessing.

USE:

nohup python paper_figure_item_level_validation_pooled.py > paper_figure_item_level_validation_pooled.log 2>&1 &

Reads:
    OUTPUT_DIR/pooled_model_error_rates_mean_std.csv
    KIDS_XLS, ADULTS_XLS               (item-level zRT / RT, + ER if present)
    OUTPUT_DIR/pooled_correlation_results.csv   (optional, for the epsilon range)

Writes:
    Figures/item_level_validation_pooled_bw.png                 (zRT, no error bars)
    Figures/item_level_validation_with_errors_pooled_bw.png     (zRT, with error bars)
    Figures/item_level_validation_RT_pooled_bw.png               (RT, no error bars)
    Figures/item_level_validation_RT_with_errors_pooled_bw.png   (RT, with error bars)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr

# --------------------------------------------------------------------------
# CONFIG -- must match item_level_behavioral_validation.py
# --------------------------------------------------------------------------
OUTPUT_DIR = "../item_level_behavioral_validation"
KIDS_XLS = "../datasets/RT_Analysis_Kids_II_modelling.xls"
ADULTS_XLS = "../datasets/RT_Analyses_Adults_modelling.xls"
ITEM_SHEET = "Itemanalyse"

POOLED_MODEL_ERR_CSV = os.path.join(OUTPUT_DIR, "pooled_model_error_rates_mean_std.csv")
# Optional: only needed for the epsilon-robustness range. If your file has
# a different name, change this.
POOLED_CORR_RESULTS_CSV = os.path.join(OUTPUT_DIR, "pooled_correlation_results.csv")

CHECKPOINT_LABEL = "batch 600"
OMEGA_VALUE = 0.10

FIG_OUT_PATH = "./Figures/item_level_validation_pooled_bw.png"
FIG_OUT_PATH_WITH_ERRORS = "./Figures/item_level_validation_with_errors_pooled_bw.png"
FIG_OUT_PATH_RT = "./Figures/item_level_validation_RT_pooled_bw.png"
FIG_OUT_PATH_RT_WITH_ERRORS = "./Figures/item_level_validation_RT_with_errors_pooled_bw.png"

# Single pooled series -- black/grey ("bw"), no children/adults split.
POOLED_COLOR = "#808080"
POOLED_MARKER = "o"
POOLED_LABEL = "Pooled (children + adults)"

# Column-name guesses for a human error-rate measure in the Itemanalyse
# sheets. Adjust/extend this list if none of these match your files --
# the script will tell you if it can't find one.
ER_COL_CANDIDATES = ["ER", "error_rate", "ErrorRate", "PctError", "pct_error",
                      "Fehlerrate", "FehlerProzent", "ErrorPct"]

# Column-name guesses for the epsilon-robustness table.
EPSILON_COL_CANDIDATES = ["epsilon", "Epsilon", "omega", "Omega", "eps"]
R_COL_CANDIDATES = ["r", "pearson_r", "r_zrt", "zrt_r", "corr", "correlation"]


def load_itemanalyse(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=ITEM_SHEET)
    df["aufgabe"] = df["aufgabe"].astype(str).str.strip()
    return df


def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# --------------------------------------------------------------------------
# Publication style -- identical to paper_figure_effects.py
# --------------------------------------------------------------------------

def _configure_style():
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams.update({
        "axes.labelsize": 32,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 30,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


_configure_style()


def _fmt_p_latex(p):
    """Compact, paper-style p-value formatting for legend labels (math mode)."""
    if p is None or np.isnan(p):
        return "n/a"
    return "$<$ .001" if p < 0.001 else f"= {p:.3f}"


def _fmt_p_plain(p):
    """Plain-text p-value formatting, for copy-pasting into manuscript prose."""
    if p is None or np.isnan(p):
        return "n/a"
    return "< .001" if p < 0.001 else f"= {p:.3f}"


# --------------------------------------------------------------------------
# Pooled correlation helper (fills the first three placeholders)
# --------------------------------------------------------------------------

def _correlate(x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(y)
    n = int(valid.sum())
    if n < 3 or np.std(x[valid]) == 0:
        print(f"[WARN] {label}: not enough valid paired data to compute a "
              f"correlation (n = {n}).")
        return None, None, n
    r, p = pearsonr(x[valid], y[valid])
    print(f"{label}: r = {r:.2f}, p {_fmt_p_plain(p)}  (n = {n})")
    return r, p, n


# --------------------------------------------------------------------------
# Epsilon-robustness range (fills the fourth placeholder, if the file
# and its columns can be located/identified)
# --------------------------------------------------------------------------

def _epsilon_robustness_range(csv_path):
    print("-" * 70)
    print("POOLED EPSILON-ROBUSTNESS RANGE")
    print("-" * 70)
    if not os.path.exists(csv_path):
        print(f"[INFO] {csv_path} not found.")
        print("       This range has to come from the per-epsilon / "
              "per-initialization correlation table produced upstream by "
              "item_level_behavioral_validation.py. Point "
              "POOLED_CORR_RESULTS_CSV at that file and re-run.")
        return

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {csv_path} -- columns found: {list(df.columns)}")

    eps_col = _find_col(df, EPSILON_COL_CANDIDATES)
    r_col = _find_col(df, R_COL_CANDIDATES)

    if eps_col is None or r_col is None:
        print("[WARN] Could not automatically identify an epsilon column "
              "and/or an r column in that file.")
        print("       Add the real column names to EPSILON_COL_CANDIDATES / "
              "R_COL_CANDIDATES at the top of this script and re-run --  "
              "the printed column list above should tell you what to add.")
        return

    grouped = df.groupby(eps_col)[r_col].mean().sort_index()
    print(f"Pooled epsilon-robustness range: r in "
          f"[{grouped.min():.2f}, {grouped.max():.2f}] "
          f"across epsilon = {list(grouped.index)}")
    print(f"(per-epsilon mean r values: "
          f"{ {k: round(v, 2) for k, v in grouped.items()} })")


# --------------------------------------------------------------------------
# Single-group (pooled) figure
# --------------------------------------------------------------------------

def _draw_pooled_figure(x_all, xerr_all, y, y_label, show_error_bars, out_path):
    """
    Single-series scatter + regression line, no children/adults split.
    Same statistics logic as the two-population version: r/p and the
    regression line come purely from x_all and y; xerr is a purely visual
    +/-1 SD-across-initializations annotation and does not affect either.
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y) & ~np.isnan(x_all)

    errorbar_kwargs = dict(
        fmt=POOLED_MARKER, color=POOLED_COLOR, alpha=0.6, markersize=11,
        markeredgecolor="black", markeredgewidth=0.6,
        ecolor=POOLED_COLOR, elinewidth=1.0, capsize=0, zorder=3,
    )
    if show_error_bars:
        errorbar_kwargs["xerr"] = xerr_all[valid]
        ax.set_xlabel(r"Model Mean Error Rate (mean $\pm$ 1 SD)")
    else:
        ax.set_xlabel(r"Model Mean Error Rate ($\%$)")

    ax.errorbar(x_all[valid], y[valid], **errorbar_kwargs)

    r, p = (np.nan, np.nan)
    if valid.sum() >= 3 and np.std(x_all[valid]) > 0:
        r, p = pearsonr(x_all[valid], y[valid])
        z = np.polyfit(x_all[valid], y[valid], 1)
        xs = np.linspace(x_all[valid].min(), x_all[valid].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=POOLED_COLOR, linewidth=3.0, zorder=4)

    legend_label = f"{POOLED_LABEL}\n($r$ = {r:.2f}, $p$ {_fmt_p_latex(p)})"
    legend_handle = Line2D(
        [0], [0], marker=POOLED_MARKER, color=POOLED_COLOR, linestyle="-",
        linewidth=3.0, markersize=11, markeredgecolor="black",
        markeredgewidth=0.6, label=legend_label,
    )

    ax.set_ylabel(y_label)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100 * x:.0f}"))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35, color="gray")

    ax.legend(
        handles=[legend_handle],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
        handlelength=2.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure to {out_path}")


def main():
    pooled = pd.read_csv(POOLED_MODEL_ERR_CSV)

    kids_full = load_itemanalyse(KIDS_XLS)
    adults_full = load_itemanalyse(ADULTS_XLS)

    er_col_kids = _find_col(kids_full, ER_COL_CANDIDATES)
    er_col_adults = _find_col(adults_full, ER_COL_CANDIDATES)
    have_er = er_col_kids is not None and er_col_adults is not None
    if not have_er:
        print("[INFO] No matching human error-rate column found in one or "
              "both Itemanalyse sheets (looked for: "
              f"{ER_COL_CANDIDATES}). The pooled error-rate correlation "
              "will be skipped -- add the real column name to "
              "ER_COL_CANDIDATES if one exists.")

    kids_cols = ["aufgabe", "zRT", "RT"] + ([er_col_kids] if have_er else [])
    adults_cols = ["aufgabe", "zRT", "RT"] + ([er_col_adults] if have_er else [])

    kids_rename = {"zRT": "zRT_kids", "RT": "RT_kids"}
    adults_rename = {"zRT": "zRT_adults", "RT": "RT_adults"}
    if have_er:
        kids_rename[er_col_kids] = "ER_kids"
        adults_rename[er_col_adults] = "ER_adults"

    kids_raw = kids_full[kids_cols].rename(columns=kids_rename)
    adults_raw = adults_full[adults_cols].rename(columns=adults_rename)

    merged = (
        pooled.merge(kids_raw, on="aufgabe", how="left")
              .merge(adults_raw, on="aufgabe", how="left")
    )

    # (a) Average each human measure across children and adults, per item.
    merged["zRT_pooled"] = merged[["zRT_kids", "zRT_adults"]].mean(axis=1, skipna=True)
    merged["RT_pooled"] = merged[["RT_kids", "RT_adults"]].mean(axis=1, skipna=True)
    if have_er:
        merged["ER_pooled"] = merged[["ER_kids", "ER_adults"]].mean(axis=1, skipna=True)

    x_all = merged["model_error_mean"].to_numpy()
    xerr_all = merged["model_error_std"].to_numpy()

    # (b) Rerun the correlations.
    print("=" * 70)
    print("POOLED CORRELATIONS -- paste these into the manuscript placeholders")
    print("=" * 70)
    _correlate(x_all, merged["zRT_pooled"].to_numpy(),
               "Pooled zRT correlation      ")
    _correlate(x_all, merged["RT_pooled"].to_numpy(),
               "Pooled raw-RT correlation   ")
    if have_er:
        _correlate(x_all, merged["ER_pooled"].to_numpy(),
                   "Pooled error-rate correlation")
    else:
        print("Pooled error-rate correlation: [PLACEHOLDER left in place -- "
              "no ER column found, see [INFO] above]")

    _epsilon_robustness_range(POOLED_CORR_RESULTS_CSV)

    # (c) Regenerate the figure as a single-group (pooled) plot.
    zrt_label = r"Standardized reaction time ($zRT$)"
    rt_label = r"Human Reaction Time (ms)"

    _draw_pooled_figure(x_all, xerr_all, merged["zRT_pooled"].to_numpy(),
                         zrt_label, show_error_bars=False, out_path=FIG_OUT_PATH)
    _draw_pooled_figure(x_all, xerr_all, merged["zRT_pooled"].to_numpy(),
                         zrt_label, show_error_bars=True, out_path=FIG_OUT_PATH_WITH_ERRORS)

    _draw_pooled_figure(x_all, xerr_all, merged["RT_pooled"].to_numpy(),
                         rt_label, show_error_bars=False, out_path=FIG_OUT_PATH_RT)
    _draw_pooled_figure(x_all, xerr_all, merged["RT_pooled"].to_numpy(),
                         rt_label, show_error_bars=True, out_path=FIG_OUT_PATH_RT_WITH_ERRORS)


if __name__ == "__main__":
    main()
