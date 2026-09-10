"""
Generate the final "Item-Level Behavioral Validation" figure(s) for the paper.

This REPLACES the two-panel figure (pooled scatter + developmental profile
across epsilon) with a single, clean scatter panel:

    Pooled model error rate (mean +/- 1 SD across the 21 pooled
    initializations) plotted against a human reaction-time measure,
    separately for children and adults, with a best-fit regression line
    per population. The r / p statistics are folded into the legend labels
    rather than a separate annotation box, so the figure reads as a
    single, self-contained result.

The across-initialization robustness check (formerly Panel B) is no longer
part of the figure. If needed for supplementary material, it can still be
recomputed from correlation_results.csv, which this script no longer reads.

Two human RT measures are plotted, each on its own pair of figures:
  - zRT: standardized reaction time, the primary continuous, cross-
    population-comparable index of processing load used in the text.
  - RT:  the raw (non-standardized) reaction time.
Both RT and ER statistics continue to be reported in the surrounding text
and in correlation_results.csv / pooled_correlation_results.csv; this
script just also renders RT the same way it renders zRT.

NOTE ON THE X ERROR BARS: the horizontal error bars (+/- 1 SD across the 21
pooled initializations) are purely a visual annotation. Neither the
Pearson correlation (r, p) nor the least-squares regression line uses the
std at all -- both are computed from x_all (the per-item mean model error)
and y (the human RT/zRT) only. Because of that, this script produces, for
each of zRT and RT, TWO versions of the exact same figure from a single
run: one with the x error bars drawn, and one without. The underlying
statistics, regression lines, and legend labels are identical between the
with/without pair -- only the whiskers differ.

USE:

nohup python paper_figure_item_level_validation.py > paper_figure_item_level_validation.log 2>&1 &

Reads (already produced by item_level_behavioral_validation.py, unchanged):
    OUTPUT_DIR/pooled_model_error_rates_mean_std.csv
and the two raw Excel files, only for the item-level human zRT/RT columns
themselves (no model / JAX code needed here -- this script is deliberately
lightweight so it can be re-run to tweak the figure without touching the
heavy checkpoint-loading pipeline):
    KIDS_XLS, ADULTS_XLS

Writes:
    Figures/item_level_validation.png                    (zRT, no error bars)
    Figures/item_level_validation_with_errors.png         (zRT, with error bars)
    Figures/item_level_validation_RT.png                  (RT, no error bars)
    Figures/item_level_validation_RT_with_errors.png       (RT, with error bars)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

# --------------------------------------------------------------------------
# CONFIG -- must match item_level_behavioral_validation.py
# --------------------------------------------------------------------------
OUTPUT_DIR = "../item_level_behavioral_validation"
KIDS_XLS = "../datasets/RT_Analysis_Kids_II_modelling.xls"
ADULTS_XLS = "../datasets/RT_Analyses_Adults_modelling.xls"
ITEM_SHEET = "Itemanalyse"

CHECKPOINT_LABEL = "batch 600"
OMEGA_VALUE = 0.10

FIG_OUT_PATH = "./Figures/item_level_validation.png"
FIG_OUT_PATH_WITH_ERRORS = "./Figures/item_level_validation_with_errors.png"
FIG_OUT_PATH_RT = "./Figures/item_level_validation_RT.png"
FIG_OUT_PATH_RT_WITH_ERRORS = "./Figures/item_level_validation_RT_with_errors.png"

# Green / purple palette.
#COLORS = {
#    "kids":   "#66FF66",  # green
#    "adults": "#B266FF",  # purple
#}
COLORS = {
    "kids":   "#999999",  # light grey
    "adults": "#000000",  # dark grey
}
MARKERS = {
    "kids":   "o",
    "adults": "^",
}


def load_itemanalyse(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=ITEM_SHEET)
    df["aufgabe"] = df["aufgabe"].astype(str).str.strip()
    return df


# --------------------------------------------------------------------------
# Publication style -- identical to paper_figure_effects.py
# --------------------------------------------------------------------------

def _configure_style():
    """Same typography as paper_figure_effects.py: STIX mathtext with the
    STIXGeneral family (no LaTeX toolchain dependency), and the same font
    sizes used there (32pt labels, 28pt ticks, 30pt legend)."""
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


def _fmt_p(p):
    """Compact, paper-style p-value formatting."""
    if p is None or np.isnan(p):
        return "n/a"
    return "$<$ .001" if p < 0.001 else f"= {p:.3f}"


# --------------------------------------------------------------------------
# Main figure
# --------------------------------------------------------------------------

def _draw_figure(merged, x_all, xerr_all, kids_col, adults_col, y_label,
                  show_error_bars, out_path):
    """
    Draws the scatter + regression-line figure and saves it to out_path.

    kids_col / adults_col: which y-column in `merged` to plot for each
    population (e.g. "zRT_kids"/"zRT_adults" or "RT_kids"/"RT_adults").
    y_label: axis label matching whichever measure is being plotted.

    show_error_bars: if True, draws the xerr whiskers (+/- 1 SD across the
    21 pooled initializations) on each point. If False, the same points and
    regression lines are drawn with no whiskers. In both cases the
    statistics (r, p) and the regression line itself are computed purely
    from x_all and y, so they are identical between the two versions --
    only the visual whiskers change.
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    groups = [
        (kids_col, COLORS["kids"], MARKERS["kids"], "Children"),
        (adults_col, COLORS["adults"], MARKERS["adults"], "Adults"),
    ]

    legend_handles = []

    for col, color, marker, label in groups:
        y = merged[col].to_numpy()
        valid = ~np.isnan(y) & ~np.isnan(x_all)

        errorbar_kwargs = dict(
            fmt=marker, color=color, alpha=0.6, markersize=11,
            markeredgecolor="black", markeredgewidth=0.6,
            ecolor=color, elinewidth=1.0, capsize=0, zorder=3,
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
            ax.plot(xs, np.polyval(z, xs), color=color, linewidth=3.0, zorder=4)

        legend_label = f"{label}  ($r$ = {r:.2f}, $p$ {_fmt_p(p)})"
        legend_handles.append(
            Line2D([0], [0], marker=marker, color=color, linestyle="-", linewidth=3.0,
                   markersize=11, markeredgecolor="black", markeredgewidth=0.6,
                   label=legend_label)
        )

    ax.set_ylabel(y_label)
    # Make x axis to be percentage, not fraction.
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100*x:.0f}"))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35, color="gray")

    #ax.legend(handles=legend_handles, loc="upper left", frameon=True,
    #          edgecolor="black", framealpha=0.9, handlelength=2.2)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,                 # una fila con las 4 entradas
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
        handlelength=2.2,
        columnspacing=1.5,
    )

    #fig.text(0.5, -0.03,
    #          f"Training checkpoint = {CHECKPOINT_LABEL}, "
    #          rf"$\Omega$ = {OMEGA_VALUE:.2f}",
    #          ha="center", fontsize=20, color="dimgray")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure to {out_path}")


def main():
    pooled = pd.read_csv(os.path.join(OUTPUT_DIR, "pooled_model_error_rates_mean_std.csv"))

    kids_raw = load_itemanalyse(KIDS_XLS)[["aufgabe", "zRT", "RT"]].rename(
        columns={"zRT": "zRT_kids", "RT": "RT_kids"}
    )
    adults_raw = load_itemanalyse(ADULTS_XLS)[["aufgabe", "zRT", "RT"]].rename(
        columns={"zRT": "zRT_adults", "RT": "RT_adults"}
    )

    merged = (
        pooled.merge(kids_raw, on="aufgabe", how="left")
              .merge(adults_raw, on="aufgabe", how="left")
    )

    x_all = merged["model_error_mean"].to_numpy()
    xerr_all = merged["model_error_std"].to_numpy()

    zrt_label = r"Standardized reaction time ($zRT$)"
    rt_label = r"Human Reaction Time (ms)"

    # zRT, without / with x error bars.
    _draw_figure(merged, x_all, xerr_all, "zRT_kids", "zRT_adults", zrt_label,
                 show_error_bars=False, out_path=FIG_OUT_PATH)
    _draw_figure(merged, x_all, xerr_all, "zRT_kids", "zRT_adults", zrt_label,
                 show_error_bars=True, out_path=FIG_OUT_PATH_WITH_ERRORS)

    # RT, without / with x error bars.
    _draw_figure(merged, x_all, xerr_all, "RT_kids", "RT_adults", rt_label,
                 show_error_bars=False, out_path=FIG_OUT_PATH_RT)
    _draw_figure(merged, x_all, xerr_all, "RT_kids", "RT_adults", rt_label,
                 show_error_bars=True, out_path=FIG_OUT_PATH_RT_WITH_ERRORS)


if __name__ == "__main__":
    main()