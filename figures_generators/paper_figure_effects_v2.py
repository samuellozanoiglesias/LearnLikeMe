# USE: nohup python paper_figure_effects_v2.py 2 STUDY WI argmax decision_module 0.15 500 > logs_paper_effects.out 2>&1 &
#
# ============================================================================
# WHAT'S NEW vs paper_figure_effects.py
# ============================================================================
# The "barplot WITH experimental RT" figure (fname3) now also plots a set of
# four white circles: the empirical HUMAN ERROR RATE for each of the four
# carry/size categories (No Carry/Carry x Small/Large), computed directly
# from the raw item-level Excel files (the same two files used by
# paper_figure_item_level_validation.py), aggregated to the PARTICIPANT
# level, and then averaged (equal-weight mean) across children and adults.
#
# "Aggregated to participant level" means: for each participant, first
# compute their own mean error rate within each category (across whatever
# items of that category they attempted), THEN average those per-participant
# category means across participants. This gives every participant equal
# weight regardless of how many trials of a given category they happened to
# contribute -- as opposed to pooling all trials from all participants
# together, which would over-weight participants who did more trials.
#
# Categorization reuses the classify_pairs() logic you supplied (kept
# verbatim below), applied to each item's parsed addends.
#
# ============================================================================
# CONFIG YOU MUST VERIFY -- new in this version (human error-rate circles)
# ============================================================================
# I do not have access to your actual Excel files, so the raw-sheet layout
# below is an ASSUMPTION, not a fact. If it's wrong, this script will fail
# loudly (KeyError naming the missing column/sheet and listing what it did
# find) rather than silently compute the wrong numbers -- fix the constants
# below to match your real files and re-run.
#
# Paths to the two raw Excel files (the same files
# paper_figure_item_level_validation.py reads for the "Itemanalyse" sheet).
HUMAN_KIDS_XLS = "../datasets/RT_Analysis_Kids_II_modelling.xls"
HUMAN_ADULTS_XLS = "../datasets/RT_Analyses_Adults_modelling.xls"

# ASSUMPTION: raw per-participant, per-item trial data (one row per trial:
# one participant x one item, with a correctness flag) lives in a sheet
# called RAW_SHEET in each of the files above -- separate from the
# already-aggregated "Itemanalyse" sheet used by the other script. If your
# raw trial-level sheet has a different name or your files don't have one
# at all (e.g. only the aggregated Itemanalyse sheet exists), update
# RAW_SHEET, or set HUMAN_ERROR_CIRCLES_ENABLED = False below and supply the
# four category means directly via MANUAL_HUMAN_ERROR_BY_CATEGORY instead.
RAW_SHEET = "RawData"
PARTICIPANT_COL = "subject"
ITEM_COL = "aufgabe"
CORRECT_COL = "correct"

# ASSUMPTION: 'aufgabe' items are encoded as two addends separated by '+'
# (e.g. "23+45", "23 + 45"). See parse_aufgabe() below if that's wrong --
# it's the one place addend parsing happens.

# Master switch: set to False to skip the new circles entirely (e.g. while
# you're still fixing up the CONFIG above) without touching the rest of the
# figure.
HUMAN_ERROR_CIRCLES_ENABLED = True

# Fallback: if you already have the four category means computed some other
# way (e.g. from a stats package) and don't want this script to touch the
# raw Excel files at all, fill these in (percent error, 0-100) and the
# script will use them instead of recomputing from RAW_SHEET.
MANUAL_HUMAN_ERROR_BY_CATEGORY = {
    # "no_carry_small": None,
    # "carry_small": None,
    # "no_carry_large": None,
    # "carry_large": None,
}

import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' or 'straight_through' version of the decision module
TRAINING_TYPE = str(sys.argv[5]).lower()  # 'decision_module' or 'all_at_once' for the analysis
OMEGA_VALUE = float(sys.argv[6])  # Specific omega value to analyze
EPOCH = int(sys.argv[7]) if len(sys.argv) > 7 else "last"  # Specific epoch for barplot analysis

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}/{TRAINING_TYPE}"
RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{TRAINING_TYPE}/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"


# ============================================================================
# Human error-rate-by-category pipeline (new in this version)
# ============================================================================

def classify_pairs(pairs, number_size):
    """Verbatim from your snippet -- the canonical carry/size categorizer."""
    categories = {
        "carry_small": [],
        "carry_large": [],
        "no_carry_small": [],
        "no_carry_large": []
    }
    max_val = 10 ** number_size
    small_thresh = 0.4 * max_val
    large_thresh = 0.6 * max_val
    for a, b, total, carries in pairs:
        has_any_carry = any(carries)
        if total < small_thresh:
            if has_any_carry:
                categories["carry_small"].append((a, b, carries))
            else:
                categories["no_carry_small"].append((a, b, carries))
        elif total > large_thresh:
            if has_any_carry:
                categories["carry_large"].append((a, b, carries))
            else:
                categories["no_carry_large"].append((a, b, carries))
    return categories


def parse_aufgabe(aufgabe):
    """
    Parse an 'aufgabe' item string into its two addends (a, b).

    ASSUMPTION: items are encoded as "a+b" (e.g. "23+45", "23 + 45"). If
    your files use a different encoding (zero-padded concatenation, a
    different separator, or a numeric item-ID needing a lookup table),
    update this function -- it's the one place addend parsing happens.
    Returns None (and the caller counts/reports it) if parsing fails.
    """
    s = str(aufgabe).strip()
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def compute_carries(a, b, number_size):
    """
    Digit-wise addition of a and b (each zero-padded to number_size
    digits), returning a list of booleans -- one per digit position from
    the ones place upward -- indicating whether that position produces a
    carry-out. Used to determine has_any_carry for classify_pairs().
    """
    a_digits = [int(d) for d in str(a).zfill(number_size)][::-1]
    b_digits = [int(d) for d in str(b).zfill(number_size)][::-1]
    carries = []
    carry_in = 0
    for da, db in zip(a_digits, b_digits):
        s = da + db + carry_in
        carry_out = s >= 10
        carries.append(carry_out)
        carry_in = 1 if carry_out else 0
    return carries


def build_category_lookup(items, number_size):
    """
    Parses every unique 'aufgabe' item into its two addends and runs
    classify_pairs() (verbatim, as given) to build an
    {(a, b): category_name} lookup. Items that fall in the excluded
    mid-range total (neither "small" nor "large") are simply absent from
    the lookup, matching classify_pairs()'s own behavior.

    ASSUMPTION: (a, b) addend pairs are unique per item, which should hold
    for a two-digit-addition stimulus set. If two different items share
    the same (a, b), this prints a warning -- check parse_aufgabe() and
    your item set if you see it.
    """
    pairs = []
    parsed_by_item = {}
    unparsed = []
    for item in items:
        parsed = parse_aufgabe(item)
        if parsed is None:
            unparsed.append(item)
            continue
        a, b = parsed
        total = a + b
        carries = compute_carries(a, b, number_size)
        pairs.append((a, b, total, carries))
        parsed_by_item[item] = (a, b)

    if unparsed:
        print(f"[WARN] Could not parse {len(unparsed)} item(s) as 'a+b': "
              f"{unparsed[:10]}{'...' if len(unparsed) > 10 else ''}. "
              f"Update parse_aufgabe() if your item encoding differs.")

    if len(set(parsed_by_item.values())) != len(parsed_by_item):
        print("[WARN] Some items share the same (a, b) addend pair -- the "
              "category lookup may not map back to items uniquely. Check "
              "parse_aufgabe() and your item set.")

    categories = classify_pairs(pairs, number_size)
    lookup = {}
    for cat_name, entries in categories.items():
        for (a, b, carries) in entries:
            lookup[(a, b)] = cat_name
    return lookup, parsed_by_item


def compute_participant_category_error(raw_xls_path, number_size):
    """
    Returns {category_name: mean_error_rate_percent} for ONE population
    (one Excel file), aggregated at the PARTICIPANT level: for each
    participant, compute their mean error rate within each category, then
    average those per-participant means across participants (equal weight
    per participant).

    Raises KeyError with the actual columns found if RAW_SHEET /
    PARTICIPANT_COL / ITEM_COL / CORRECT_COL don't match your file, rather
    than silently producing wrong numbers.
    """
    try:
        df = pd.read_excel(raw_xls_path, sheet_name=RAW_SHEET)
    except ValueError as e:
        raise ValueError(
            f"Could not find sheet '{RAW_SHEET}' in {raw_xls_path}. Update "
            f"RAW_SHEET at the top of this script to match your file. "
            f"Original error: {e}"
        )

    missing = [c for c in (PARTICIPANT_COL, ITEM_COL, CORRECT_COL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} in sheet '{RAW_SHEET}' of {raw_xls_path}, "
            f"found: {list(df.columns)}. Update PARTICIPANT_COL / ITEM_COL / "
            f"CORRECT_COL at the top of this script."
        )

    df = df.copy()
    df[ITEM_COL] = df[ITEM_COL].astype(str).str.strip()

    lookup, parsed_by_item = build_category_lookup(df[ITEM_COL].unique(), number_size)
    df["category"] = df[ITEM_COL].map(lambda it: lookup.get(parsed_by_item.get(it)))

    n_before = len(df)
    df = df.dropna(subset=["category"])
    print(f"[INFO] {raw_xls_path}: kept {len(df)}/{n_before} trials after "
          f"categorization (mid-range-total items and unparseable 'aufgabe' "
          f"strings are excluded, same as classify_pairs()).")

    df[CORRECT_COL] = pd.to_numeric(df[CORRECT_COL], errors="coerce")
    df["error_pct"] = 100.0 * (1.0 - df[CORRECT_COL])

    participant_cat_means = (
        df.groupby([PARTICIPANT_COL, "category"])["error_pct"].mean().reset_index()
    )
    category_means = participant_cat_means.groupby("category")["error_pct"].mean()
    return category_means.to_dict()


def compute_pooled_human_error_by_category(number_size):
    """
    Computes participant-level category error rates separately for kids and
    adults, then averages the two populations with equal weight (simple
    mean of the two population means -- NOT weighted by n) to get one
    number per category.
    """
    if MANUAL_HUMAN_ERROR_BY_CATEGORY and all(
        v is not None for v in MANUAL_HUMAN_ERROR_BY_CATEGORY.values()
    ) and len(MANUAL_HUMAN_ERROR_BY_CATEGORY) == 4:
        print("[INFO] Using MANUAL_HUMAN_ERROR_BY_CATEGORY instead of recomputing from raw Excel files.")
        return dict(MANUAL_HUMAN_ERROR_BY_CATEGORY)

    kids_cat = compute_participant_category_error(HUMAN_KIDS_XLS, number_size)
    adults_cat = compute_participant_category_error(HUMAN_ADULTS_XLS, number_size)

    pooled = {}
    for cat in ("no_carry_small", "carry_small", "no_carry_large", "carry_large"):
        vals = [d[cat] for d in (kids_cat, adults_cat) if cat in d and not np.isnan(d[cat])]
        if not vals:
            print(f"[WARN] No data for category '{cat}' in either population.")
            pooled[cat] = np.nan
        else:
            pooled[cat] = float(np.mean(vals))
    return pooled


# ============================================================================
# Main analysis / figures (unchanged aside from the new circles in fname3)
# ============================================================================

def analyze_multidigit_module(raw_dir, figures_dir, omega_value, param_type):
    os.makedirs(figures_dir, exist_ok=True)

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        return

    # Read CSV with low_memory=False to avoid dtype warning
    combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
    print(f"Combined logs loaded from: {combined_logs_path}")
    print(f"Initial number of rows in combined logs: {len(combined_logs)}")

    # Clean and convert data types
    combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
    combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')

    # Convert accuracy columns to numeric
    acc_cols = [
        "test_pairs_no_carry_small_accuracy",
        "test_pairs_carry_small_accuracy",
        "test_pairs_no_carry_large_accuracy",
        "test_pairs_carry_large_accuracy"
    ]
    for col in acc_cols:
        if col in combined_logs.columns:
            combined_logs[col] = pd.to_numeric(combined_logs[col], errors='coerce')

    print(f"After cleaning: {len(combined_logs)} rows remaining")

    # Filter for the specific omega and param_init_type
    subset_param = combined_logs[
        (combined_logs['param_init_type'] == param_type) &
        (combined_logs['omega'] == omega_value)
    ]

    if subset_param.empty:
        print(f"No data found for param_type={param_type} and omega={omega_value}")
        return

    print(f"Found {len(subset_param)} rows for omega={omega_value}, param={param_type}")

    # --- Figure 1: Errors over epochs (aggregated across all epsilons) ---
    acc_labels = ["Small - No Carry", "Small - Carry", "Large - No Carry", "Large - Carry"]
    colors = ["#999999", "#4D4D4D", "#999999", "#4D4D4D"]  # light grey, dark grey, light grey, dark grey
    linestyles = ["-", "-", ":", ":"]  # solid = blue family, dashed = red family

    pw = {}
    for col in acc_cols:
        if col in subset_param.columns:
            pw[col] = subset_param.pivot_table(index='epoch', columns='run', values=col)

    if pw:
        plt.figure(figsize=(12, 7))
        for col, lbl, colcol, linestyle in zip(acc_cols, acc_labels, colors, linestyles):
            if col not in pw:
                continue
            df_runs = pw[col]
            if df_runs is None or df_runs.empty:
                continue
            mean = df_runs.mean(axis=1)
            std = df_runs.std(axis=1).fillna(0)
            mean = pd.to_numeric(mean, errors='coerce').astype(float).to_numpy()
            std = pd.to_numeric(std, errors='coerce').astype(float).to_numpy()
            idx = np.array(df_runs.index, dtype=float)
            mask = np.isfinite(mean) & np.isfinite(std) & np.isfinite(idx)
            mean = mean[mask]
            std = std[mask]
            idx = idx[mask]
            if len(mean) == 0:
                continue
            error_mean = 100 - mean
            error_std = np.sqrt(std)
            plt.plot(idx, error_mean, label=lbl, color=colcol, linewidth=2.5, linestyle=linestyle)
            plt.fill_between(idx, error_mean - error_std, error_mean + error_std, color=colcol, alpha=0.2)

        plt.xlabel('Batch', fontsize=32)
        plt.ylabel('Model Mean Error Rate (%)', fontsize=32)
        plt.ylim(-5, 105)
        plt.xlim(left=0, right=2000)
        plt.tick_params(axis='both', labelsize=28)
        plt.legend(loc='best', fontsize=30)
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

        safe_om = str(omega_value).replace('.', '_')
        fname = os.path.join(figures_dir, f"errors_epochs_omega_{safe_om}_all_eps_{MODEL_TYPE}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Figure 1 saved to: {fname}")

    # --- Figure 2: Barplot for a specific epoch (averaged over all epsilons) ---
    if not subset_param.empty:
        if EPOCH == "last":
            selected_epoch = subset_param['epoch'].max()
        else:
            selected_epoch = EPOCH
        epoch_data = subset_param[subset_param['epoch'] == selected_epoch]

        if not epoch_data.empty:
            cols_ordered = [
                "test_pairs_no_carry_small_accuracy",
                "test_pairs_carry_small_accuracy",
                "test_pairs_no_carry_large_accuracy",
                "test_pairs_carry_large_accuracy"
            ]

            means = []
            stds = []
            for col in cols_ordered:
                if col in epoch_data.columns:
                    values = pd.to_numeric(epoch_data[col], errors='coerce').dropna()
                    error_values = 100 - values
                    means.append(error_values.mean())
                    stds.append(error_values.std())
                else:
                    means.append(0)
                    stds.append(0)

            x_positions = [0, 0.8, 2.2, 3.0]
            safe_om = str(omega_value).replace('.', '_')

            # --- Barplot WITHOUT experimental RT (unchanged) ---
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(x_positions, means, yerr=np.sqrt(stds), capsize=5, color=colors,
                         alpha=0.8, edgecolor='black', linewidth=1.5, width=0.7)

            ax.set_ylabel('Model Mean Error Rate (%)', fontsize=32)
            ax.set_xticks([0.4, 2.6])
            ax.set_xticklabels(['Small', 'Large'], fontsize=28)
            ax.set_xlabel('Problem Size', fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

            legend_elements = [
                Patch(facecolor=colors[0], edgecolor='black', label='Small - No Carry', alpha=0.8),
                Patch(facecolor=colors[1], edgecolor='black', label='Small - Carry', alpha=0.8),
                Patch(facecolor=colors[2], edgecolor='black', label='Large - No Carry', alpha=0.8),
                Patch(facecolor=colors[3], edgecolor='black', label='Large - Carry', alpha=0.8)
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=30, framealpha=0.95)

            fname2 = os.path.join(figures_dir, f"barplot_errors_omega_{safe_om}_epoch_{int(selected_epoch)}_{MODEL_TYPE}.png")
            plt.savefig(fname2, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Figure 2 (without RT) saved to: {fname2}")

            # --- Barplot WITH experimental RT (+ NEW human error-rate circles) ---
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(x_positions, means, yerr=np.sqrt(stds), capsize=5, color=colors,
                         alpha=0.8, edgecolor='black', linewidth=1.5, width=0.7)

            # Experimental reaction times data
            min_RT = 1250
            max_RT = 4137.5
            rt_values = [1400, 1800, 2700, 3200]  # RT(SNC), RT(SC), RT(LNC), RT(LC)

            ax2 = ax.twinx()
            ax2.set_ylabel('Human Reaction Time (ms)', fontsize=32)
            ax2.set_ylim(min_RT, max_RT)
            ax2.tick_params(axis='y', labelsize=28)

            rt_normalized = [(rt - min_RT) / (max_RT - min_RT) * 105 for rt in rt_values]

            ax.plot(x_positions[0:2], rt_normalized[0:2], color='black', linewidth=2.5,
                    linestyle='-', zorder=4)
            ax.plot(x_positions[2:4], rt_normalized[2:4], color='black', linewidth=2.5,
                    linestyle='-', zorder=4)

            ax.scatter(x_positions, rt_normalized, marker='*', s=1000, color='black', edgecolors='dimgray', linewidth=1,
                      zorder=5, label='Experimental RTs')

            # --- NEW: human error rate (circles), from raw per-participant data,
            # aggregated participant-level and averaged over kids + adults.
            # Plotted directly on the primary (error-rate) axis -- no
            # normalization needed since it's already on the same 0-100% scale
            # as the model bars.
            human_error_plotted = False
            if HUMAN_ERROR_CIRCLES_ENABLED:
                try:
                    human_error_by_cat = compute_pooled_human_error_by_category(NUMBER_SIZE)
                    human_error_ordered = [
                        human_error_by_cat.get("no_carry_small", np.nan),
                        human_error_by_cat.get("carry_small", np.nan),
                        human_error_by_cat.get("no_carry_large", np.nan),
                        human_error_by_cat.get("carry_large", np.nan),
                    ]
                    ax.plot(x_positions[0:2], human_error_ordered[0:2], color='dimgray',
                            linewidth=2.5, linestyle='--', zorder=4)
                    ax.plot(x_positions[2:4], human_error_ordered[2:4], color='dimgray',
                            linewidth=2.5, linestyle='--', zorder=4)
                    ax.scatter(x_positions, human_error_ordered, marker='o', s=350,
                               color='white', edgecolors='black', linewidth=2,
                               zorder=6, label='Human Error Rate (pooled)')
                    human_error_plotted = True
                except (KeyError, ValueError, FileNotFoundError) as e:
                    print(f"[WARN] Skipping human error-rate circles -- {e}")

            ax.set_ylabel('Model Mean Error Rate (%)', fontsize=32)
            ax.set_xticks([0.4, 2.6])
            ax.set_xticklabels(['Small', 'Large'], fontsize=28)
            ax.set_xlabel('Problem Size', fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

            legend_elements_with_rt = [
                Patch(facecolor=colors[0], edgecolor='black', label='No Carry', alpha=0.8),
                Patch(facecolor=colors[1], edgecolor='black', label='Carry', alpha=0.8),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markeredgecolor='black',
                       markersize=25, label='Experimental RTs'),
            ]
            if human_error_plotted:
                legend_elements_with_rt.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                           markeredgecolor='black', markersize=20,
                           label='Human Error Rate (pooled)')
                )
            ax.legend(handles=legend_elements_with_rt, loc='upper left', fontsize=32, framealpha=0.95)

            fname3 = os.path.join(figures_dir, f"barplot_errors_omega_{safe_om}_epoch_{int(selected_epoch)}_with_RT_{MODEL_TYPE}.png")
            plt.savefig(fname3, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Figure 2 (with RT) saved to: {fname3}")
        else:
            print(f"No data found for last epoch")
    else:
        print("No data available for barplot")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)