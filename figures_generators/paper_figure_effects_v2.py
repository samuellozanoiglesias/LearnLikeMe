# USE: nohup python paper_figure_effects_v2.py 2 STUDY RI straight_through decision_module 0.10 600 > logs_paper_effects.out 2>&1 &
#
# ============================================================================
# WHAT'S NEW vs the previous version
# ============================================================================
# This version produces the NEW main-text Figure 4 (fig:Results_with_RT):
# it plots ONLY the pooled Moeller et al. (2011) reaction times (children
# and adults averaged with equal weight) against the model's error-rate
# bars, on a real millisecond scale. All other human RT markers previously
# shown on this figure (the Kids-only / Adults-only item-level markers, and
# the literature-reported Klein et al. "Experimental RTs" stars) have been
# REMOVED from this script -- Klein et al. and the Moeller et al. age-group
# breakdown now live exclusively in the companion SM figure produced by
# paper_figure_effects_v2_local_scales.py.
#
# In addition, this version runs the participant-based 2x2 repeated-measures
# ANOVA (Problem Size x Carry) that treats each of the model's N=20 weight
# initializations as a "participant" (analogous to the human repeated-
# measures ANOVA in Moeller et al., 2011), and writes the full descriptive +
# inferential results to a plain-text file (Moeller_et_al_ANOVA_pooled.txt)
# formatted for direct transcription into the manuscript's
# "Participant-Based Analysis (F1-Analog)" section.
#
# Pipeline for the pooled human RT value (unchanged from the previous
# version): the real data lives in the "Itemanalyse" sheet of the two raw
# Excel files, one row per ITEM (already aggregated across participants
# within each file), with:
#     col "aufgabe" -- the item, e.g. "4 + 3 ="  ->  interpreted as (4, 3)
#     col "RT"      -- the reaction time for that item (the value to compare)
# Each item is classified into one of the four carry/size categories using
# has_carry() + a threshold check (max_number=100), RT is averaged across
# items within each category SEPARATELY for the kids file and the adults
# file, and the two population means are then averaged together with equal
# weight (NOT weighted by n) to get one pooled value per category.
#
# ============================================================================
# CONFIG YOU MUST VERIFY
# ============================================================================
# Paths to the two raw Excel files (the same files
# paper_figure_item_level_validation.py reads for the "Itemanalyse" sheet).
HUMAN_KIDS_XLS = "../datasets/RT_Analysis_Kids_II_modelling.xls"
HUMAN_ADULTS_XLS = "../datasets/RT_Analyses_Adults_modelling.xls"

# Item-level sheet/columns, per your description.
ITEM_SHEET = "Itemanalyse"
ITEM_COL = "aufgabe"   # e.g. "4 + 3 =" -> (4, 3)
RT_COL = "RT"          # reaction time -- the value we average per category

# max_number for the carry/size categorization, per your instructions.
MAX_NUMBER_FOR_CATEGORIES = 100

CATEGORY_ORDER = ["no_carry_small", "carry_small", "no_carry_large", "carry_large"]

# Real millisecond range for the Moeller et al. pooled RT axis (unchanged
# from the "Moeller et al. Reaction Time (ms)" axis in the previous
# combined Klein+Moeller figure).
MIN_RT_POOLED, MAX_RT_POOLED = 2000, 6000

# Master switch: set to False to skip the pooled human RT markers entirely
# without touching the rest of the figure.
HUMAN_RT_MARKERS_ENABLED = True

import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from statsmodels.stats.anova import AnovaRM

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
# Human item-level RT-by-category pipeline (used only to build the pooled
# Moeller et al. value -- kids/adults are computed separately here purely as
# an intermediate step, they are NOT plotted individually on this figure)
# ============================================================================

def parse_aufgabe(aufgabe):
    """
    Parse an 'aufgabe' item string into its two addends (a, b).

    Handles "4 + 3 =" (trailing '=' and surrounding whitespace optional),
    as well as plain "4+3". Returns None (and the caller counts/reports
    it) if parsing fails.
    """
    s = str(aufgabe).strip()
    s = re.sub(r"=+\s*$", "", s).strip()  # drop a trailing '='
    m = re.match(r"^(\d+)\s*\+\s*(\d+)$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def has_carry(a, b, number_size):
    """
    Digit-wise carry check, adapted verbatim from generate_carry_operations():
    True if ANY corresponding digit pair of a and b sums to >= 10. This is
    a per-digit check (no carry propagation across positions), matching
    the snippet you supplied.
    """
    for d in range(number_size):
        digit_a = (a // (10 ** d)) % 10
        digit_b = (b // (10 ** d)) % 10
        if digit_a + digit_b >= 10:
            return True
    return False


def classify_item(a, b, max_number, number_size):
    """
    Classifies a single (a, b) addend pair into one of the four
    carry/size categories, adapted from generate_problem_size_datasets():
    small_threshold = int(0.4 * max_number), large_min = int(0.6 * max_number),
    large_max = max_number. Items whose total falls in the excluded
    mid-range (neither small nor large) return None, matching the
    original datasets' behavior of simply not including them.
    """
    small_threshold = int(0.4 * max_number)
    large_min = int(0.6 * max_number)
    large_max = max_number
    total = a + b
    if total < small_threshold:
        return "carry_small" if has_carry(a, b, number_size) else "no_carry_small"
    elif large_min < total < large_max:
        return "carry_large" if has_carry(a, b, number_size) else "no_carry_large"
    return None


def compute_item_category_rt(xls_path, sheet_name, item_col, rt_col,
                              max_number, number_size):
    """
    Returns {category_name: mean_RT} for ONE population (one Excel file),
    computed directly at the ITEM level: each row of `sheet_name` is one
    item (already aggregated across participants in this sheet), so the
    category mean is just the mean of `rt_col` over the items that fall
    in that category -- no participant-level step needed.

    Raises KeyError with the actual columns found if item_col/rt_col
    don't match your file, rather than silently computing the wrong
    numbers.
    """
    df = pd.read_excel(xls_path, sheet_name=sheet_name)
    missing = [c for c in (item_col, rt_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} in sheet '{sheet_name}' of {xls_path}, "
            f"found: {list(df.columns)}."
        )

    df = df.copy()
    parsed = df[item_col].apply(parse_aufgabe)
    n_before = len(df)

    unparsed_mask = parsed.isna()
    if unparsed_mask.any():
        examples = df.loc[unparsed_mask, item_col].astype(str).unique().tolist()
        print(f"[WARN] {xls_path}: could not parse {int(unparsed_mask.sum())} "
              f"item(s) as 'a+b': {examples[:10]}"
              f"{'...' if len(examples) > 10 else ''}. Update parse_aufgabe() "
              f"if your item encoding differs.")

    df = df.loc[~unparsed_mask].copy()
    parsed = parsed.loc[~unparsed_mask]
    df["_a"] = parsed.apply(lambda t: t[0])
    df["_b"] = parsed.apply(lambda t: t[1])
    df["category"] = df.apply(
        lambda row: classify_item(row["_a"], row["_b"], max_number, number_size),
        axis=1,
    )

    n_no_cat = int(df["category"].isna().sum())
    df = df.dropna(subset=["category"])
    print(f"[INFO] {xls_path}: kept {len(df)}/{n_before} items after parsing + "
          f"categorization ({n_no_cat} fell in the excluded mid-range total, "
          f"same as generate_problem_size_datasets()).")

    df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")
    cat_means = df.groupby("category")[rt_col].mean()
    cat_counts = df.groupby("category")[rt_col].count()

    for cat in CATEGORY_ORDER:
        if cat in cat_means.index:
            print(f"[INFO] {xls_path}: category '{cat}' -> mean RT = "
                  f"{cat_means[cat]:.1f} (n items = {int(cat_counts[cat])})")
        else:
            print(f"[WARN] {xls_path}: no items found for category '{cat}'.")

    return cat_means.to_dict()


def compute_pooled_rt_by_category(kids_dict, adults_dict):
    """
    Equal-weight mean of the kids and adults per-category RT means (NOT
    weighted by n) -- one pooled number per category. This is the value
    plotted in the new main-text Figure 4.
    """
    pooled = {}
    for cat in CATEGORY_ORDER:
        vals = [d[cat] for d in (kids_dict, adults_dict)
                if cat in d and d[cat] is not None and not np.isnan(d[cat])]
        if not vals:
            print(f"[WARN] No RT data for category '{cat}' in either "
                  f"population -- pooled mean set to NaN.")
            pooled[cat] = np.nan
        else:
            pooled[cat] = float(np.mean(vals))
    return pooled


# ============================================================================
# Participant-based ANOVA (F1-Analog): the model's N weight initializations
# stand in for human "participants" in a 2x2 (Problem Size x Carry)
# repeated-measures ANOVA, analogous to the human ANOVA reported in
# Moeller et al. (2011). This is entirely a property of the MODEL's own
# error rates (it does not use the human RT data at all) -- it is the
# statistic that gets qualitatively compared against Moeller et al.'s
# reported pattern in the manuscript text.
# ============================================================================

def _partial_eta_sq(F, df_num, df_den):
    """Partial eta^2 from F and its degrees of freedom: (F*df1)/(F*df1+df2)."""
    return (F * df_num) / (F * df_num + df_den)


def run_participant_based_anova(epoch_data, cols_ordered, output_path,
                                 omega_value, selected_epoch):
    """
    Builds a long-format (subject x ProblemSize x Carry) table from the
    model's per-initialization error rates at `selected_epoch` (one row of
    `epoch_data` per weight initialization, identified by the 'run'
    column), runs the 2x2 repeated-measures ANOVA, and writes descriptive
    statistics plus the full ANOVA table (F, df, p, partial eta^2) for
    both main effects and their interaction to `output_path` as plain
    text, formatted for direct transcription into the manuscript's
    "Participant-Based Analysis (F1-Analog)" section.
    """
    if "run" not in epoch_data.columns:
        print("[WARN] 'run' column not found in epoch_data -- cannot identify "
              "individual initializations as 'subjects'. Skipping ANOVA.")
        return

    # category column -> (ProblemSize, Carry, human-readable label), in the
    # canonical small/no-carry, small/carry, large/no-carry, large/carry order.
    factor_map = {
        cols_ordered[0]: ("Small", "NoCarry", "small/no-carry"),
        cols_ordered[1]: ("Small", "Carry", "small/carry"),
        cols_ordered[2]: ("Large", "NoCarry", "large/no-carry"),
        cols_ordered[3]: ("Large", "Carry", "large/carry"),
    }
    label_order = [lbl for _, _, lbl in factor_map.values()]

    long_rows = []
    for _, row in epoch_data.iterrows():
        subject = row["run"]
        for col, (size, carry, label) in factor_map.items():
            if col in epoch_data.columns and pd.notna(row[col]):
                error_rate = 100.0 - float(row[col])
                long_rows.append({
                    "subject": subject, "ProblemSize": size, "Carry": carry,
                    "category_label": label, "ErrorRate": error_rate,
                })
    long_df = pd.DataFrame(long_rows)

    if long_df.empty:
        print("[WARN] No data available to run the participant-based ANOVA.")
        return

    # AnovaRM requires a complete, balanced design: keep only subjects
    # (initializations) with all four categories present.
    complete_subjects = (
        long_df.groupby("subject")["category_label"].nunique()
        .loc[lambda s: s == 4].index
    )
    n_dropped = long_df["subject"].nunique() - len(complete_subjects)
    if n_dropped > 0:
        print(f"[WARN] Dropping {n_dropped} initialization(s) with incomplete "
              f"category data before running the repeated-measures ANOVA.")
    long_df = long_df[long_df["subject"].isin(complete_subjects)].copy()
    n_used = long_df["subject"].nunique()

    # --- Descriptive statistics per category ---
    desc = (
        long_df.groupby("category_label")["ErrorRate"]
        .agg(["mean", "std", "count"])
        .reindex(label_order)
    )

    # --- 2x2 repeated-measures ANOVA ---
    aovrm = AnovaRM(long_df, depvar="ErrorRate", subject="subject",
                     within=["ProblemSize", "Carry"])
    fit = aovrm.fit()
    table = fit.anova_table.copy()
    table["partial_eta_sq"] = [
        _partial_eta_sq(r["F Value"], r["Num DF"], r["Den DF"])
        for _, r in table.iterrows()
    ]

    # --- Write everything to a plain-text file ---
    with open(output_path, "w") as f:
        f.write("Participant-Based Analysis (F1-Analog) -- Moeller et al. pooled benchmark\n")
        f.write("(Figure 4 of the main text: model bars vs. pooled Moeller et al. RTs)\n")
        f.write("=" * 78 + "\n")
        f.write(f"Omega = {omega_value}, Epoch = {int(selected_epoch)}, "
                f"N initializations (subjects) used = {n_used}\n\n")

        f.write("Descriptive statistics (model mean error rate %, across initializations):\n")
        for label in label_order:
            row = desc.loc[label]
            f.write(f"  {label:<15s} M = {row['mean']:.2f}%  "
                    f"(SD = {row['std']:.2f}%, n = {int(row['count'])})\n")
        f.write("\n")

        f.write("2x2 repeated-measures ANOVA (Problem Size x Carry) on model error rate:\n")
        effect_names = {"ProblemSize": "Problem Size", "Carry": "Carry-over",
                         "ProblemSize:Carry": "Problem Size x Carry"}
        for effect, row in table.iterrows():
            F = row["F Value"]
            df1 = row["Num DF"]
            df2 = row["Den DF"]
            p = row["Pr > F"]
            eta2 = row["partial_eta_sq"]
            p_str = "< .001" if p < .001 else f"= {p:.3f}"
            eta2_str = "< .001" if eta2 < .001 else f"= {eta2:.2f}"
            name = effect_names.get(effect, effect)
            f.write(f"  {name:<22s} F({df1:.0f}, {df2:.0f}) = {F:.2f}, "
                    f"p {p_str}, partial eta^2 {eta2_str}\n")
        f.write("\n")

        f.write("Ready-to-paste sentence skeletons (verify numbers before use):\n")
        f.write(
            "  Mean error rates (across the {n} initializations, epoch {ep}, "
            "$\\omega$={om}) were: small/no-carry $M$={m0:.2f}\\% ($SD$={s0:.2f}\\%); "
            "small/carry $M$={m1:.2f}\\% ($SD$={s1:.2f}\\%); large/no-carry "
            "$M$={m2:.2f}\\% ($SD$={s2:.2f}\\%); large/carry $M$={m3:.2f}\\% "
            "($SD$={s3:.2f}\\%).\n".format(
                n=n_used, ep=int(selected_epoch), om=omega_value,
                m0=desc.loc["small/no-carry", "mean"], s0=desc.loc["small/no-carry", "std"],
                m1=desc.loc["small/carry", "mean"], s1=desc.loc["small/carry", "std"],
                m2=desc.loc["large/no-carry", "mean"], s2=desc.loc["large/no-carry", "std"],
                m3=desc.loc["large/carry", "mean"], s3=desc.loc["large/carry", "std"],
            )
        )
        for effect, row in table.iterrows():
            F = row["F Value"]; df1 = row["Num DF"]; df2 = row["Den DF"]
            p = row["Pr > F"]; eta2 = row["partial_eta_sq"]
            p_str = "p<.001" if p < .001 else f"p={p:.3f}"
            eta2_str = "<.001" if eta2 < .001 else f"={eta2:.2f}"
            name = effect_names.get(effect, effect)
            f.write(f"  {name}: $F({df1:.0f},{df2:.0f})={F:.2f}$, {p_str}, "
                    f"partial $\\eta^2${eta2_str}\n")

    print(f"[INFO] Participant-based ANOVA (Moeller-pooled benchmark) written to: {output_path}")


# ============================================================================
# Main analysis / figures
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
    linestyles = ["-", "-", (0, (1, 3)), (0, (1, 3))]  # solid = blue family, dashed = red family

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
            ax.set_xticklabels(['Small', 'Large'], fontsize=30)
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

            # --- Participant-based ANOVA (model's N initializations as "participants") ---
            anova_output_path = os.path.join(figures_dir, "Moeller_et_al_ANOVA_pooled.txt")
            try:
                run_participant_based_anova(epoch_data, cols_ordered, anova_output_path,
                                             omega_value, selected_epoch)
            except Exception as e:
                print(f"[WARN] Participant-based ANOVA failed -- {e}")

            # --- Barplot WITH Moeller et al. pooled human RT only (main-text Figure 4) ---
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(x_positions, means, yerr=np.sqrt(stds), capsize=5, color=colors,
                         alpha=0.8, edgecolor='black', linewidth=1.5, width=0.7)

            ax2 = ax.twinx()
            ax2.set_ylabel('Human Reaction Time (ms)', fontsize=32)
            ax2.set_ylim(MIN_RT_POOLED, MAX_RT_POOLED)
            ax2.tick_params(axis='y', labelsize=28)

            human_rt_plotted = False
            if HUMAN_RT_MARKERS_ENABLED:
                try:
                    kids_rt_by_cat = compute_item_category_rt(
                        HUMAN_KIDS_XLS, ITEM_SHEET, ITEM_COL, RT_COL,
                        MAX_NUMBER_FOR_CATEGORIES, NUMBER_SIZE,
                    )
                    adults_rt_by_cat = compute_item_category_rt(
                        HUMAN_ADULTS_XLS, ITEM_SHEET, ITEM_COL, RT_COL,
                        MAX_NUMBER_FOR_CATEGORIES, NUMBER_SIZE,
                    )
                    pooled_rt_by_cat = compute_pooled_rt_by_category(kids_rt_by_cat, adults_rt_by_cat)

                    print(f"[INFO] Kids RT by category (intermediate only):   {kids_rt_by_cat}")
                    print(f"[INFO] Adults RT by category (intermediate only): {adults_rt_by_cat}")
                    print(f"[INFO] Pooled (Moeller et al.) RT by category:    {pooled_rt_by_cat}")

                    pooled_rt = [pooled_rt_by_cat.get(cat, np.nan) for cat in CATEGORY_ORDER]

                    ax2.plot(x_positions[0:2], pooled_rt[0:2], color='dimgray', linewidth=2,
                             linestyle='--', zorder=4)
                    ax2.plot(x_positions[2:4], pooled_rt[2:4], color='dimgray', linewidth=2,
                             linestyle='--', zorder=4)
                    ax2.scatter(x_positions, pooled_rt, marker='*', s=30 ** 2, facecolor='white',
                                edgecolors='black', linewidth=2, zorder=6, label='Experimental RTs')
                    human_rt_plotted = True
                except (KeyError, ValueError, FileNotFoundError) as e:
                    print(f"[WARN] Skipping Moeller et al. pooled RT markers -- {e}")

            ax.set_ylabel('Model Mean Error Rate (%)', fontsize=32)
            ax.set_xticks([0.4, 2.6])
            ax.set_xticklabels(['Small', 'Large'], fontsize=30)
            ax.set_xlabel('Problem Size', fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

            legend_elements_with_rt = [
                Patch(facecolor=colors[0], edgecolor='black', label='Without carry-over', alpha=0.8),
                Patch(facecolor=colors[1], edgecolor='black', label='With carry-over', alpha=0.8),
            ]
            if human_rt_plotted:
                legend_elements_with_rt.append(
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
                           markeredgecolor='black', markersize=24, label='Human data')
                )
            ax.legend(handles=legend_elements_with_rt, loc='upper left', fontsize=30, framealpha=0.95)

            fname3 = os.path.join(
                figures_dir,
                f"barplot_errors_omega_{safe_om}_epoch_{int(selected_epoch)}_with_RT_pooled_Moeller_local_scale_{MODEL_TYPE}.png"
            )
            plt.savefig(fname3, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Figure 3 (Moeller pooled RT, main-text Figure 4) saved to: {fname3}")
        else:
            print(f"No data found for last epoch")
    else:
        print("No data available for barplot")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)
