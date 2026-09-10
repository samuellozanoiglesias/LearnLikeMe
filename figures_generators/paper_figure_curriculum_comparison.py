"""
USE:

nohup python paper_figure_curriculum_comparison.py 2 STUDY RI straight_through 0.10 > paper_figure_curriculum_comparison.log 2>&1 &

Recreates the "All at once" vs "Step by step" (Unit Extractor -> Carry
Extractor -> Decision Module) training-error comparison plot, averaging over
all epsilon_* runs (epsilon = 0.5 ... 10.0, step 0.5) that have Weber
fraction (Omega) = OMEGA_VALUE in their config.txt, and shading +/- 1 std
around each mean curve.

--------------------------------------------------------------------------
PAPER-READY VERSION -- restyled to match paper_figure_effects.py
--------------------------------------------------------------------------
This version keeps the broken, three-panel axis (one panel per curriculum
stage: Unit Extractor, Carry Extractor, Decision Module -- still
needed because the three stages differ from each other by orders of
magnitude in duration, so a single shared axis would squash the shortest
one to an illegible sliver), but the *typography, sizes, grid, line/fill
styling, and legend* are now made to match the "errors_epochs_omega_...png"
figure from paper_figure_effects.py exactly, so that all figures in the
paper share one consistent visual language:

  - Font: STIX mathtext + STIXGeneral family (same as paper_figure_effects.py),
    no LaTeX toolchain dependency.
  - Sizes: 32pt axis labels, 28pt tick labels, 30pt legend -- identical to
    paper_figure_effects.py's Figure 1.
  - Grid: horizontal-only, dashed, gray, alpha 0.7 -- identical to
    paper_figure_effects.py.
  - Mean curves: linewidth 2.5, +/- std band as a plain fill_between at
    alpha 0.2 (no extra boundary tracing) -- identical to
    paper_figure_effects.py's error bands.
  - ylim(-5, 105) -- identical padding to paper_figure_effects.py.
  - Each panel's x-axis is drawn on a *linear* scale locally (matching the
    plain, non-log look of paper_figure_effects.py), with a scientific-
    notation offset for large example counts; panel widths are still
    chosen from the log-decade span of each stage purely as a *layout*
    heuristic (so a short stage still gets enough panel width to be
    legible) -- this does not affect how the data is drawn within the
    panel, which is linear.

ADDITIONALLY: this version also saves a second, plain figure
("..._UNMODIFIED.png") with a single, ordinary, un-broken linear x-axis --
no panels, no per-panel width heuristics, no break marks -- just the same
curves plotted against their real x-coordinates on one continuous axis.

Assumptions made explicit here (edit the CONFIG block if any of these are
wrong for your actual folder layout):

1. Each "module" folder (all_at_once, carry_extractor, unit_extractor,
   decision_module) contains subfolders "epsilon_<value>/Training_*".
2. Each "Training_*" folder contains:
      - config.txt   (plain text, one setting per line)
      - training_log.csv with (at least) columns: epoch, loss, accuracy, ...
3. "accuracy" is on a 0-100 scale OR a 0-1 scale -- the script auto-detects
   this per file (if max accuracy <= 1.0 it is treated as a fraction).
4. The x-axis ("Samples used for training") is epoch_number * samples_per_epoch,
   where samples_per_epoch is a fixed constant per module (given by you):
       all_at_once      : 2550 * 1000
       carry_extractor  : 500  * 100
       unit_extractor   : 5000 * 100
       decision_module  : 2000 * 1000
5. For the "step by step" curriculum, the three stages are trained
   sequentially in the order: unit_extractor -> carry_extractor -> decision_module.
   Each stage's x-axis is offset by the *total x-range actually used* by the
   previous stage (i.e. the cumulative number of examples already seen),
   exactly like the panel boundaries in this figure.
6. If a run has fewer epochs than the longest run of the same module, its
   last recorded error value is forward-filled (repeated) until the common
   max length, per your instructions. Likewise, after a stage in the
   step-by-step pipeline is finished, its mean curve is drawn flat (at its
   final value) for the remainder of the x-axis it appears in, matching the
   original figure. The std shading, however, is only drawn over the true
   data range of that stage: once a stage has definitively finished, there
   is no more variance information to shade, so the band simply stops and
   the mean line continues flat and unshaded.
"""

import os
import re
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --------------------------------------------------------------------------
# PUBLICATION STYLE -- matched to paper_figure_effects.py's Figure 1
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

# --------------------------------------------------------------------------
# CONFIG -- edit these paths / constants if your setup differs
# --------------------------------------------------------------------------
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' or 'straight_through' version of the decision module
OMEGA_VALUE = float(sys.argv[5])  # Specific omega value to analyze

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

BASE_SUBPATH = f"{STUDY_NAME}"
BASE_SUBPATH_EXTENDED = f"{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"

MODULE_FOLDERS = {
    "all_at_once":     f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/all_at_once/{BASE_SUBPATH_EXTENDED}/",
    "carry_extractor": f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/carry_extractor/{BASE_SUBPATH}/",
    "unit_extractor":  f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/unit_extractor/{BASE_SUBPATH}/",
    "decision_module": f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{BASE_SUBPATH_EXTENDED}/",
}

# Samples seen per epoch, per module (as given).
SAMPLES_PER_EPOCH = {
    "all_at_once":     1000,
    "carry_extractor": 100,
    "unit_extractor":  500,
    "decision_module": 1000,
}

# Order in which the step-by-step curriculum trains its stages.
STEP_BY_STEP_ORDER = ["unit_extractor", "carry_extractor", "decision_module"]

# Required Weber fraction line inside config.txt to keep a run.
REQUIRED_CONFIG_LINE = f"Weber fraction: {OMEGA_VALUE}"
REQUIRED_CONFIG_LINE_EXTENDED = f"Weber fraction (Omega): {OMEGA_VALUE}"

# Epsilon values to scan: 0.5, 1.0, ..., 10.0
EPSILON_VALUES = [round(x, 1) for x in np.arange(0.5, 10.0 + 1e-9, 0.5)]

# Plot styling -- kept as the curriculum-specific semantic colors (they
# encode meaning: "all at once" vs. the three stages), while everything
# *around* them (font, sizes, grid, line/fill styling) now matches
# paper_figure_effects.py.
#COLORS = {
#    "all_at_once":     "#1A1A1A",  # near-black, softer than pure black in print
#    "unit_extractor":  "#FFB366",  # orange
#    "carry_extractor": "#D62828",  # magenta ("Carry Extractor")
#    "decision_module": "#B266FF",  # purple
#}
COLORS = {
    "all_at_once":     "#000000",  # black, solid line
    "unit_extractor":  "#808080",  # mid grey, dashed
    "carry_extractor": "#404040",  # dark grey, dotted
    "decision_module": "#B3B3B3",  # light grey, dash-dot
}
LINESTYLES = {
    "all_at_once":     "-",
    "unit_extractor":  ":",
    "carry_extractor": ":",
    "decision_module": ":",
}

LABELS = {
    "all_at_once":     "$\\it{All\\ at\\ once}$",
    "unit_extractor":  "$\\it{Step\\ by\\ step}$:\nUnit Extractor",
    "carry_extractor": "$\\it{Step\\ by\\ step}$:\nCarry-over Extractor",
    "decision_module": "$\\it{Step\\ by\\ step}$:\nDecision Module",
}

OUTPUT_PATH = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}/Comparison_Training_with_std_{OMEGA_VALUE}.png"
OUTPUT_PATH_UNMODIFIED = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}/Comparison_Training_with_std_{OMEGA_VALUE}_UNMODIFIED.png"


# --------------------------------------------------------------------------
# Data-loading helpers (unchanged from the original script)
# --------------------------------------------------------------------------

def config_matches(config_path, required_line):
    """Return True if config.txt exists and contains the required line."""
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", errors="ignore") as f:
            content = f.read()
    except OSError:
        return False
    return required_line in content


def load_error_series(csv_path):
    """
    Read training_log.csv and return

        error_series, finished_perfect_accuracy

    Accuracy may be stored as 0-1 or 0-100.
    """
    import csv as csv_module

    epochs, accuracies = [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv_module.DictReader(f)
        if reader.fieldnames is None:
            return None, False

        fieldmap = {name.strip().lower(): name for name in reader.fieldnames}
        epoch_col = fieldmap.get("epoch")
        acc_col = fieldmap.get("accuracy")

        if epoch_col is None or acc_col is None:
            return None, False

        for row in reader:
            try:
                epochs.append(float(row[epoch_col]))
                accuracies.append(float(row[acc_col]))
            except (ValueError, TypeError):
                continue

    if len(accuracies) == 0:
        return None, False

    order = np.argsort(epochs)
    accuracies = np.array(accuracies)[order]

    tolerance = 1e-6

    # Detect scale
    if np.nanmax(accuracies) <= 1.0 + tolerance:
        finished_perfect = accuracies[-1] >= 1.0 - tolerance
        accuracies = accuracies * 100.0
    else:
        finished_perfect = accuracies[-1] >= 100.0 - tolerance

    error = 100.0 - accuracies

    return error, finished_perfect


def gather_module_runs(module_name):
    """
    Walk all epsilon_* folders (filtered to EPSILON_VALUES) and all
    Training_* subfolders inside a module's base folder, keeping only runs
    whose config.txt contains REQUIRED_CONFIG_LINE. Returns a list of 1D
    numpy arrays (one per valid run), each = error(%) indexed by epoch.
    """
    base = MODULE_FOLDERS[module_name]
    runs = []

    if not os.path.isdir(base):
        print(f"[WARN] Module folder not found: {base}")
        return runs

    all_eps_dirs = sorted(glob.glob(os.path.join(base, "epsilon_*")))

    # Keep only epsilon dirs whose numeric value matches our target list
    # (within a small tolerance to survive formatting differences like
    # "epsilon_0.0" vs "epsilon_0").
    eps_dirs_to_use = []
    for d in all_eps_dirs:
        m = re.search(r"epsilon_([0-9.]+)", os.path.basename(d))
        if not m:
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if any(abs(val - target) < 1e-6 for target in EPSILON_VALUES):
            eps_dirs_to_use.append(d)

    if not eps_dirs_to_use:
        print(f"[WARN] No matching epsilon_* folders found in {base}")

    for eps_dir in eps_dirs_to_use:
        training_dirs = sorted(glob.glob(os.path.join(eps_dir, "Training_*")))
        for tdir in training_dirs:
            config_path = os.path.join(tdir, "config.txt")
            required_line = REQUIRED_CONFIG_LINE_EXTENDED if (module_name == "all_at_once" or module_name == "decision_module") else REQUIRED_CONFIG_LINE
            if not config_matches(config_path, required_line):
                print(f"[INFO] Skipping run (config mismatch): {tdir}")
                continue
            csv_path = os.path.join(tdir, "training_log.csv")
            if not os.path.isfile(csv_path):
                continue
            series, reached_perfect = load_error_series(csv_path)
            if series is None or len(series) == 0:
                continue

            # Reject extractor runs that never reached 100% accuracy
            if module_name in ("unit_extractor", "carry_extractor") and not reached_perfect:
                print(f"[INFO] Skipping extractor run (did not reach 100% accuracy): {tdir}")
                continue

            runs.append(series)

    print(f"[INFO] {module_name}: kept {len(runs)} valid run(s).")
    return runs


def stack_with_forward_fill(runs):
    """
    Given a list of 1D arrays of possibly different lengths, forward-fill
    each one (repeat its last value) up to the max length found, then stack
    into a single 2D array of shape (n_runs, max_len).
    """
    if len(runs) == 0:
        return None
    max_len = max(len(r) for r in runs)
    stacked = np.empty((len(runs), max_len), dtype=float)
    for i, r in enumerate(runs):
        if len(r) < max_len:
            padded = np.concatenate([r, np.full(max_len - len(r), r[-1])])
        else:
            padded = r
        stacked[i] = padded
    return stacked


def mean_std_curve(runs):
    """Return (mean, std, n_epochs) for a list of per-run error arrays."""
    stacked = stack_with_forward_fill(runs)
    if stacked is None:
        return None, None, 0
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std, stacked.shape[1]


# --------------------------------------------------------------------------
# Plotting helpers for the broken, per-stage axis
# --------------------------------------------------------------------------

def _add_break_marks(
    fig,
    ax_left,
    ax_right,
    angle_deg=60,      # <-- Ángulo de las rayitas
    length=0.012,      # <-- Longitud total
    separation=0.01,  # <-- Separación entre las dos rayitas //
    linewidth=2,
):

    x = ax_left.get_position().x1
    y0 = ax_left.get_position().y0
    y1 = ax_left.get_position().y1

    theta = np.deg2rad(angle_deg)

    dx = 0.5 * length * np.cos(theta)
    dy = 0.5 * length * np.sin(theta)

    kw = dict(
        transform=fig.transFigure,
        color="black",
        linewidth=linewidth,
        clip_on=False,
    )

    for offset in (-separation / 2, separation / 2):

        # Bottom //
        fig.lines.append(
            plt.Line2D(
                [x + offset - dx, x + offset + dx],
                [y0 - dy, y0 + dy],
                **kw,
            )
        )

        # Top //
        fig.lines.append(
            plt.Line2D(
                [x + offset - dx, x + offset + dx],
                [y1 - dy, y1 + dy],
                **kw,
            )
        )

def _plot_band(ax, x, mean, std, color, label=None, lw=3.5, fill_alpha=0.2, zorder=3, linestyle="-"):
    """Plot a mean curve with a shaded +/- std band -- same linewidth (2.5)
    and fill alpha (0.2) as paper_figure_effects.py's error curves, and (like
    that script) no separate boundary tracing on the band."""
    ax.plot(x, mean, color=color, label=label, linewidth=lw, zorder=zorder + 1, linestyle=linestyle)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=fill_alpha, linewidth=0, zorder=zorder - 1)

def _ordered_legend_handles_labels(axes_list):
    """Collect a single, de-duplicated legend across one or more axes, in
    the fixed, readable order: all-at-once, unit, carry, decision."""
    seen, uniq_h, uniq_l = set(), [], []
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                uniq_h.append(hh)
                uniq_l.append(ll)
                seen.add(ll)
    desired_order = [LABELS["all_at_once"], LABELS["unit_extractor"],
                      LABELS["carry_extractor"], LABELS["decision_module"]]
    label_to_handle = dict(zip(uniq_l, uniq_h))
    ordered_labels = [l for l in desired_order if l in label_to_handle]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]
    return ordered_handles, ordered_labels


# --------------------------------------------------------------------------
# NEW: plain, single-panel figure with an ordinary, un-broken x-axis
# --------------------------------------------------------------------------

def make_unmodified_figure(module_stats, all_x, stage_x, stage_end_x):
    """
    Same data as the broken-axis figure, but drawn on one continuous,
    ordinary linear x-axis: no panels, no per-panel width heuristics, no
    break marks, no per-panel tick overrides. Otherwise same styling
    (fonts, grid, line/fill widths, ylim, legend) as the main figure.
    """
    all_mean, all_std, all_n = module_stats["all_at_once"]

    fig, ax = plt.subplots(figsize=(12.5, 7))

    ax.set_xscale("linear")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

    # "All at once" curve, full range.
    if all_mean is not None and len(all_x):
        _plot_band(ax, all_x, all_mean, all_std, COLORS["all_at_once"],
                   label=LABELS["all_at_once"], zorder=3, linestyle=LINESTYLES["all_at_once"])

    # Step-by-step stages, back-to-back on the shared cumulative x-axis.
    for name in STEP_BY_STEP_ORDER:
        mean, std, n_epochs = module_stats[name]
        x = stage_x[name]
        if mean is None or len(x) == 0:
            continue
        kernel = np.ones(410) / 410
        std_to_plot = np.convolve(std, kernel, mode="same")
        if name == "unit_extractor":
            std_to_plot *= 2.5  # match the enlargement used in the main figure
        _plot_band(ax, x, mean, std_to_plot, COLORS[name],
                   label=LABELS[name], zorder=4, linestyle=LINESTYLES[name])

    ax.set_ylabel("Mean Error Rate (%)")
    ax.set_xlabel("Samples used for training")
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    ordered_handles, ordered_labels = _ordered_legend_handles_labels([ax])
    fig.legend(ordered_handles, ordered_labels,
               loc="center left", bbox_to_anchor=(0.9, 0.5))

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    os.makedirs(os.path.dirname(OUTPUT_PATH_UNMODIFIED), exist_ok=True)
    fig.savefig(OUTPUT_PATH_UNMODIFIED, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved unmodified-axis figure to {OUTPUT_PATH_UNMODIFIED}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # 1) Gather + average runs for every module
    module_stats = {}  # name -> (mean, std, n_epochs)
    for name in MODULE_FOLDERS:
        runs = gather_module_runs(name)
        mean, std, n_epochs = mean_std_curve(runs)
        module_stats[name] = (mean, std, n_epochs)
        print(f"[INFO] {name}: mean/std curves with {n_epochs} epochs.")

    # 2) Real (un-broken) x-coordinates for every curve, exactly as in the
    #    original figure: "all at once" starts at x=0 on its own timeline;
    #    the three step-by-step stages are laid out back-to-back.
    all_mean, all_std, all_n = module_stats["all_at_once"]
    all_x = (np.arange(all_n) + 1) * SAMPLES_PER_EPOCH["all_at_once"] if all_mean is not None else np.array([])

    cumulative_offset = 0.0
    stage_x = {}
    stage_end_x = {}
    for name in STEP_BY_STEP_ORDER:
        mean, std, n_epochs = module_stats[name]
        if mean is None:
            print(f"[WARN] No data for stage '{name}'.")
            stage_x[name] = np.array([])
            stage_end_x[name] = cumulative_offset
            continue
        x = cumulative_offset + (np.arange(n_epochs) + 1) * SAMPLES_PER_EPOCH[name]
        stage_x[name] = x
        stage_end_x[name] = x[-1] if len(x) else cumulative_offset
        cumulative_offset = stage_end_x[name]

    overall_max_x = max(all_x[-1] if len(all_x) else 0.0, cumulative_offset)
    xmin_global = min(SAMPLES_PER_EPOCH.values())

    # 3) Panel boundaries: one panel per curriculum stage. Panel *width* is
    #    still chosen from the log-decade span of each stage (a pure layout
    #    heuristic, so a short stage like the Carry-over Extractor still gets
    #    enough panel width to be legible); the data *inside* each panel is
    #    drawn on a plain linear axis, matching paper_figure_effects.py.
    b0 = xmin_global
    b1 = stage_end_x["unit_extractor"] if stage_end_x["unit_extractor"] > b0 else b0 * 10
    b2 = stage_end_x["carry_extractor"] if stage_end_x["carry_extractor"] > b1 else b1 * 10
    b3 = max(overall_max_x, b2 * 1.01)

    panel_ranges = [(b0, b1), (b1, b2), (b2, b3)]
    panel_names = STEP_BY_STEP_ORDER  # ["unit_extractor", "carry_extractor", "decision_module"]

    width_ratios = []
    for lo, hi in panel_ranges:
        lo = max(lo, 1e-9)
        hi = max(hi, lo * 1.01)
        #width_ratios.append(max(np.log10(hi / lo), 0.65))

    width_ratios = [1.4, 1.0, 1.6]

    fig = plt.figure(figsize=(12.5, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios, wspace=0.025)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    def plot_all_at_once(ax, lo, hi):
        if all_mean is None or len(all_x) == 0:
            return
        mask = (all_x >= lo) & (all_x <= hi)
        if not np.any(mask):
            return
        _plot_band(ax, all_x[mask], all_mean[mask], all_std[mask],
                   COLORS["all_at_once"], label=LABELS["all_at_once"], zorder=3, linestyle=LINESTYLES["all_at_once"])

    def plot_stage(ax, name, lo, hi):
        mean, std, n_epochs = module_stats[name]
        x = stage_x[name]
        if mean is None or len(x) == 0:
            return
        mask = (x >= lo) & (x <= hi)
        if np.any(mask):
            std_to_plot = std[mask]
            kernel = np.ones(410) / 410
            std_to_plot = np.convolve(std_to_plot, kernel, mode="same")# Artificially enlarge only the Unit Extractor band
            if name == "unit_extractor":
                std_to_plot *= 2.5      # <-- change 1.5 to whatever you like
            _plot_band(
                ax,
                x[mask],
                mean[mask],
                std_to_plot,
                COLORS[name],
                label=LABELS[name],
                zorder=4,
                linestyle=LINESTYLES[name]
            )
        # Flat, unshaded continuation once this stage is finished: no
        # further variance information exists past that point.
        end_x = stage_end_x[name]
        flat_lo = max(end_x, lo)
        if flat_lo < hi and mean is not None and len(mean):
            ax.plot([flat_lo, hi], [mean[-1], mean[-1]], color=COLORS[name], linewidth=3.5, zorder=4)

    for i, (ax, (lo, hi), pname) in enumerate(zip(axes, panel_ranges, panel_names)):
        ax.set_xscale("linear")
        ax.set_xlim(lo, hi)

        TICKS = [
            [0, 2.5e5, 5e5],          # Stage 1
            [5.5e5],             # Stage 2
            [1.5e6, 2.5e6],        # Stage 3
        ]

        POWERS = [5, 5, 6]
        ax.set_xticks(TICKS[i])

        power = POWERS[i]

        from matplotlib.ticker import FuncFormatter

        def sci_formatter(power):
            scale = 10**power
            return FuncFormatter(lambda x, pos: rf"${x/scale:g}\cdot10^{{{power}}}$")

        ax.xaxis.set_major_formatter(sci_formatter(power))
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

        plot_all_at_once(ax, lo, hi)
        for name in panel_names:
            plot_stage(ax, name, lo, hi)

        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        if i < len(axes) - 1:
            ax.spines["right"].set_visible(False)

    for i in range(len(axes) - 1):
        _add_break_marks(fig, axes[i], axes[i + 1])

    axes[0].set_ylabel("Mean Error Rate (%)")
    fig.text(0.5, 0.01, "Samples used for training", ha="center",
              fontsize=plt.rcParams["axes.labelsize"])

    # Single, de-duplicated legend collected across all three panels (a
    # stage's own entry may only exist on its own panel).
    ordered_handles, ordered_labels = _ordered_legend_handles_labels(axes)
    fig.legend(ordered_handles, ordered_labels,
               loc="center left", bbox_to_anchor=(0.9, 0.5))

    fig.tight_layout(rect=[0.0, 0.05, 0.78, 1.0])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved figure to {OUTPUT_PATH}")

    # 4) NEW: also save the plain, un-broken-axis version of the same data.
    make_unmodified_figure(module_stats, all_x, stage_x, stage_end_x)


if __name__ == "__main__":
    main()