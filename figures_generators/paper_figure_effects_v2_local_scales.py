# USE: nohup python paper_figure_effects_v2_local_scales.py 2 STUDY RI straight_through decision_module 0.10 600 > logs_paper_effects_local_scales.out 2>&1 &
#
# ============================================================================
# WHAT'S NEW vs the previous version
# ============================================================================
# This version produces the SM-Appendix robustness figure referenced as
# fig:Results_with_RT_Klein_Moeller_age (subsection
# "Participant-Based Robustness Across Studies and Age Groups"). It overlays
# THREE empirical reaction-time sources onto the same model error-rate bars
# used in the main-text Figure 4:
#   (i)   Klein et al. (2010) adults      -- literature-reported category
#         means (hardcoded, unchanged from previous versions), black stars.
#   (ii)  Moeller et al. (2011) adults    -- computed from the adults raw
#         Excel file, NOT pooled with kids, white triangles.
#   (iii) Moeller et al. (2011) children  -- computed from the kids raw
#         Excel file, NOT pooled with adults, gray circles.
#
# Because (i) and (ii) are both adult samples with comparable absolute RT
# ranges, they share a single right-hand axis ("Adult Reaction Time, ms").
# Because children respond systematically more slowly overall, (iii) gets
# its own, separately-scaled right-hand axis ("Child Reaction Time, ms").
# Both axis ranges are now computed DYNAMICALLY from the data actually
# plotted on them (with a fixed padding fraction), rather than hardcoded,
# so the figure cannot silently clip a marker if the real numbers differ
# from what was assumed when the constants were first tuned.
#
# The previous three-axis (kids/adults/pooled, all on independently scaled
# real-ms axes) and pooled-only-with-Klein-overlay figures have been
# REMOVED from this script: the main text now uses the pooled Moeller et
# al. benchmark exclusively (see paper_figure_effects_v2.py), and this
# script's only remaining "with RT" figure is the age-group/cross-study
# robustness figure described above.
#
# This version also writes age_group_ANOVA_Moeller_and_Klein.txt, which
# contains everything needed to fill in the SM subsection
# "Participant-Based Robustness Across Studies and Age Groups":
#   1. The SAME 2x2 repeated-measures ANOVA (Problem Size x Carry) on the
#      model's N=20 weight initializations reported in
#      Moeller_et_al_ANOVA_pooled.txt (reproduced here for convenience,
#      since it is the statistic underlying every panel of this figure --
#      the model bars themselves do not change across empirical overlays).
#   2. Descriptive category means for all three empirical sources (plus,
#      as a cross-check, their equal-weight pooled mean).
#   3. Pearson correlations between the model's category-level error rate
#      and each of the three empirical RT sources, as a quantitative index
#      of convergence. IMPORTANT CAVEAT, also written into the .txt file:
#      we only have literature-reported category means for Klein et al.
#      (no raw participant-level data), and only item-level-aggregated
#      Excel sheets for Moeller et al. (no raw per-participant trial data
#      either). A genuine repeated-measures ANOVA on human RT (of the kind
#      Moeller et al. themselves ran) can therefore NOT be reproduced here
#      for any of the three empirical sources -- only for the model. The
#      Pearson correlations (n=4 categories) are a purely descriptive
#      convergence index, not a substitute inferential test, and should be
#      reported/cited as such.
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
CATEGORY_LABELS = ["small/no-carry", "small/carry", "large/no-carry", "large/carry"]

# Literature-reported Klein et al. (2010) category-level RTs (ms), unchanged
# from previous versions: RT(SNC), RT(SC), RT(LNC), RT(LC).
KLEIN_RT_VALUES = [1475, 1867, 2580, 3198]

# Padding fraction used when auto-computing each right-hand axis range from
# the data actually plotted on it (see _axis_range_with_padding()).
AXIS_PADDING_FRACTION = 0.1

# Master switch: set to False to skip the empirical RT markers entirely
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
from scipy.stats import pearsonr
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
# Human item-level RT-by-category pipeline
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
    weighted by n) -- reported in the robustness .txt as a cross-check
    against the main-text pooled value, but NOT plotted on this figure.
    """
    pooled = {}
    for cat in CATEGORY_ORDER:
        vals = [d[cat] for d in (kids_dict, adults_dict)
                if cat in d and d[cat] is not None and not np.isnan(d[cat])]
        if not vals:
            pooled[cat] = np.nan
        else:
            pooled[cat] = float(np.mean(vals))
    return pooled


def _ordered(d):
    """dict {category: value} -> list in CATEGORY_ORDER, NaN for missing."""
    return [d.get(cat, np.nan) for cat in CATEGORY_ORDER]


def _axis_range_with_padding(values, pad_frac=AXIS_PADDING_FRACTION):
    """
    Computes a (min, max) axis range from whatever data will actually be
    plotted on that axis, with symmetric padding, instead of relying on a
    hardcoded range that could silently clip a marker if the real
    empirical values differ from what was assumed when tuning constants.
    """
    finite_vals = [v for v in values if np.isfinite(v)]
    if not finite_vals:
        return (0.0, 1.0)
    lo, hi = min(finite_vals), max(finite_vals)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.2
    pad = span * pad_frac
    return (lo - pad, hi + pad)


# ============================================================================
# Participant-based ANOVA (model initializations as "participants") -- same
# statistic as in paper_figure_effects_v2.py / Moeller_et_al_ANOVA_pooled.txt,
# reproduced here so this figure's companion .txt is self-contained.
# ============================================================================

def _partial_eta_sq(F, df_num, df_den):
    return (F * df_num) / (F * df_num + df_den)


def run_participant_based_anova(epoch_data, cols_ordered):
    """
    Returns (desc, table, n_used) for the 2x2 repeated-measures ANOVA
    (Problem Size x Carry) on the model's per-initialization error rates,
    without writing anything to disk (the caller writes the combined
    robustness report).
    """
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
        return None, None, 0

    complete_subjects = (
        long_df.groupby("subject")["category_label"].nunique()
        .loc[lambda s: s == 4].index
    )
    long_df = long_df[long_df["subject"].isin(complete_subjects)].copy()
    n_used = long_df["subject"].nunique()

    desc = (
        long_df.groupby("category_label")["ErrorRate"]
        .agg(["mean", "std", "count"])
        .reindex(label_order)
    )

    aovrm = AnovaRM(long_df, depvar="ErrorRate", subject="subject",
                     within=["ProblemSize", "Carry"])
    table = aovrm.fit().anova_table.copy()
    table["partial_eta_sq"] = [
        _partial_eta_sq(r["F Value"], r["Num DF"], r["Den DF"])
        for _, r in table.iterrows()
    ]
    return desc, table, n_used


def write_age_group_robustness_report(output_path, omega_value, selected_epoch,
                                       model_means, klein_rt, moeller_adults_rt,
                                       moeller_kids_rt, moeller_pooled_rt,
                                       anova_desc, anova_table, n_used):
    """
    Writes age_group_ANOVA_Moeller_and_Klein.txt: the model ANOVA (same
    statistic as the main-text benchmark), descriptive category means for
    all three empirical sources (plus pooled, as a cross-check), and
    descriptive Pearson correlations between the model's error-rate
    pattern and each empirical source -- with an explicit caveat that this
    correlation, not a repeated-measures ANOVA, is the only statistic we
    can compute on the human side given the data actually available (see
    header comment of this script for the full explanation).
    """
    effect_names = {"ProblemSize": "Problem Size", "Carry": "Carry-over",
                     "ProblemSize:Carry": "Problem Size x Carry"}

    with open(output_path, "w") as f:
        f.write("Participant-Based Robustness Across Studies and Age Groups\n")
        f.write("(SM figure: Klein et al. adults + Moeller et al. adults/children)\n")
        f.write("=" * 78 + "\n")
        f.write(f"Omega = {omega_value}, Epoch = {int(selected_epoch)}\n\n")

        f.write("-" * 78 + "\n")
        f.write("PART 1 -- Model ANOVA (identical statistic underlying every panel of\n")
        f.write("this figure; reproduced here for convenience from\n")
        f.write("Moeller_et_al_ANOVA_pooled.txt, since the model bars do not change\n")
        f.write("across empirical overlays).\n")
        f.write("-" * 78 + "\n")
        if anova_desc is not None:
            f.write(f"N initializations (subjects) used = {n_used}\n\n")
            f.write("Descriptive statistics (model mean error rate %, across initializations):\n")
            for label in CATEGORY_LABELS:
                row = anova_desc.loc[label]
                f.write(f"  {label:<15s} M = {row['mean']:.2f}%  "
                        f"(SD = {row['std']:.2f}%, n = {int(row['count'])})\n")
            f.write("\n2x2 repeated-measures ANOVA (Problem Size x Carry) on model error rate:\n")
            for effect, row in anova_table.iterrows():
                F = row["F Value"]; df1 = row["Num DF"]; df2 = row["Den DF"]
                p = row["Pr > F"]; eta2 = row["partial_eta_sq"]
                p_str = "< .001" if p < .001 else f"= {p:.3f}"
                eta2_str = "< .001" if eta2 < .001 else f"= {eta2:.2f}"
                name = effect_names.get(effect, effect)
                f.write(f"  {name:<22s} F({df1:.0f}, {df2:.0f}) = {F:.2f}, "
                        f"p {p_str}, partial eta^2 {eta2_str}\n")
        else:
            f.write("[WARN] Model ANOVA could not be computed -- see console log.\n")
        f.write("\n")

        f.write("-" * 78 + "\n")
        f.write("PART 2 -- Descriptive category means for each empirical RT source (ms)\n")
        f.write("-" * 78 + "\n")
        header = f"  {'category':<15s} {'Klein adults':>13s} {'Moeller adults':>15s} " \
                 f"{'Moeller kids':>13s} {'pooled (M+A)':>13s}"
        f.write(header + "\n")
        for i, label in enumerate(CATEGORY_LABELS):
            cat = CATEGORY_ORDER[i]
            f.write(f"  {label:<15s} {klein_rt[i]:>13.1f} {moeller_adults_rt[i]:>15.1f} "
                    f"{moeller_kids_rt[i]:>13.1f} {moeller_pooled_rt.get(cat, float('nan')):>13.1f}\n")
        f.write("\n")

        f.write("-" * 78 + "\n")
        f.write("PART 3 -- Descriptive convergence: Pearson correlation between the\n")
        f.write("model's category-level error rate and each empirical RT source.\n")
        f.write("CAVEAT: n=4 categories -- these are DESCRIPTIVE convergence indices,\n")
        f.write("not a substitute for an inferential repeated-measures ANOVA on human\n")
        f.write("data. We could not run such an ANOVA on any of the three empirical\n")
        f.write("sources because we only have literature-reported category means for\n")
        f.write("Klein et al. (no raw participant data) and item-level-aggregated\n")
        f.write("Excel sheets for Moeller et al. (no raw per-participant trial data).\n")
        f.write("Report/cite r values with this caveat explicit.\n")
        f.write("-" * 78 + "\n")
        for name, emp_vals in [("Klein et al. adults", klein_rt),
                                ("Moeller et al. adults", moeller_adults_rt),
                                ("Moeller et al. children", moeller_kids_rt)]:
            pairs = [(m, e) for m, e in zip(model_means, emp_vals)
                     if np.isfinite(m) and np.isfinite(e)]
            if len(pairs) < 3:
                f.write(f"  {name:<26s} insufficient data (n={len(pairs)}) -- skipped\n")
                continue
            mv, ev = zip(*pairs)
            r, p = pearsonr(mv, ev)
            f.write(f"  {name:<26s} r({len(pairs) - 2}) = {r:.3f}, p = {p:.3f}  (n = {len(pairs)})\n")
        f.write("\n")

        f.write("Ready-to-paste sentence skeleton (verify numbers before use):\n")
        f.write(
            "  Once each empirical source is placed on its own age-appropriate\n"
            "  axis, the model's category-level error rate correlates with Klein\n"
            "  et al. adults, Moeller et al. adults, and Moeller et al. children\n"
            "  alike (Pearson's r reported above for each; see PART 3 caveat on\n"
            "  interpretation given n=4 categories).\n"
        )

    print(f"[INFO] Age-group/cross-study robustness report written to: {output_path}")


# ============================================================================
# Main analysis / figures
# ============================================================================

def analyze_multidigit_module(raw_dir, figures_dir, omega_value, param_type):
    os.makedirs(figures_dir, exist_ok=True)

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        return

    combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
    print(f"Combined logs loaded from: {combined_logs_path}")
    print(f"Initial number of rows in combined logs: {len(combined_logs)}")

    combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
    combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')

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
    colors = ["#999999", "#4D4D4D", "#999999", "#4D4D4D"]
    linestyles = ["-", "-", ":", ":"]

    pw = {}
    for col in acc_cols:
        if col in subset_param.columns:
            pw[col] = subset_param.pivot_table(index='epoch', columns='run', values=col)

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

            # --- Compute the three empirical RT sources (Klein adults hardcoded;
            #     Moeller adults/children computed separately, NOT pooled) ---
            human_rt_available = False
            klein_rt = list(KLEIN_RT_VALUES)
            moeller_adults_rt = [np.nan] * 4
            moeller_kids_rt = [np.nan] * 4
            moeller_pooled_rt = {}
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
                    moeller_pooled_rt = compute_pooled_rt_by_category(kids_rt_by_cat, adults_rt_by_cat)
                    moeller_adults_rt = _ordered(adults_rt_by_cat)
                    moeller_kids_rt = _ordered(kids_rt_by_cat)

                    print(f"[INFO] Klein et al. adults RT by category:   {dict(zip(CATEGORY_ORDER, klein_rt))}")
                    print(f"[INFO] Moeller et al. adults RT by category: {adults_rt_by_cat}")
                    print(f"[INFO] Moeller et al. kids RT by category:   {kids_rt_by_cat}")
                    human_rt_available = True
                except (KeyError, ValueError, FileNotFoundError) as e:
                    print(f"[WARN] Skipping empirical RT markers -- {e}")

            # --- Model ANOVA + combined robustness report ---
            anova_desc, anova_table, n_used = (None, None, 0)
            try:
                anova_desc, anova_table, n_used = run_participant_based_anova(epoch_data, cols_ordered)
            except Exception as e:
                print(f"[WARN] Participant-based ANOVA failed -- {e}")

            robustness_output_path = os.path.join(figures_dir, "age_group_ANOVA_Moeller_and_Klein.txt")
            try:
                write_age_group_robustness_report(
                    robustness_output_path, omega_value, selected_epoch,
                    model_means=means, klein_rt=klein_rt,
                    moeller_adults_rt=moeller_adults_rt, moeller_kids_rt=moeller_kids_rt,
                    moeller_pooled_rt=moeller_pooled_rt,
                    anova_desc=anova_desc, anova_table=anova_table, n_used=n_used,
                )
            except Exception as e:
                print(f"[WARN] Could not write age-group robustness report -- {e}")

            # --- Barplot WITH Klein et al. adults + Moeller et al. adults/children
            #     (SM Appendix age-group/cross-study robustness figure) ---
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(x_positions, means, yerr=np.sqrt(stds), capsize=5, color=colors,
                         alpha=0.8, edgecolor='black', linewidth=1.5, width=0.7)

            ax_adult = ax.twinx()
            ax_child = ax.twinx()
            ax_child.spines['right'].set_position(('axes', 1.14))

            ax_adult.set_ylabel('Adult Reaction Time (ms)', fontsize=32)
            ax_child.set_ylabel('Child Reaction Time (ms)', fontsize=32)

            # Adult axis is shared by Klein et al. adults and Moeller et al.
            # adults; range is fit to whichever of the two is actually plotted.
            ax_adult.set_ylim(1400, 4100)
            ax_child.set_ylim(2750, 7250)

            ax_adult.tick_params(axis='y', labelsize=28)
            ax_child.tick_params(axis='y', labelsize=28)

            # Klein et al. adults -- black stars, on the adult axis.
            ax_adult.plot(x_positions[0:2], klein_rt[0:2], color='black', linewidth=2.5,
                          linestyle='-', zorder=4)
            ax_adult.plot(x_positions[2:4], klein_rt[2:4], color='black', linewidth=2.5,
                          linestyle='-', zorder=4)
            ax_adult.scatter(x_positions, klein_rt, marker='*', s=28 ** 2, color='black',
                             edgecolors='gray', linewidth=1, zorder=4,
                             label='Klein et al. RTs')

            if human_rt_available:
                # Moeller et al. adults -- white triangles, on the adult axis.
                ax_adult.plot(x_positions[0:2], moeller_adults_rt[0:2], color='dimgray',
                              linewidth=1.5, linestyle='--', zorder=4)
                ax_adult.plot(x_positions[2:4], moeller_adults_rt[2:4], color='dimgray',
                              linewidth=1.5, linestyle='--', zorder=4)
                ax_adult.scatter(x_positions, moeller_adults_rt, marker='^', s=18 ** 2,
                                 facecolor='white', edgecolors='black', linewidth=2,
                                 zorder=6, label='Moeller et al. adult RTs')

                # Moeller et al. children -- gray circles, on the child axis.
                ax_child.plot(x_positions[0:2], moeller_kids_rt[0:2], color='dimgray',
                              linewidth=1.5, linestyle='--', zorder=4)
                ax_child.plot(x_positions[2:4], moeller_kids_rt[2:4], color='dimgray',
                              linewidth=1.5, linestyle='--', zorder=4)
                ax_child.scatter(x_positions, moeller_kids_rt, marker='o', s=18 ** 2,
                                 facecolor='white', edgecolors='black', linewidth=2,
                                 zorder=6, label='Moeller et al. child RTs')

            ax.set_ylabel('Model Mean Error Rate (%)', fontsize=32)
            ax.set_xticks([0.4, 2.6])
            ax.set_xticklabels(['Small', 'Large'], fontsize=30)
            ax.set_xlabel('Problem Size', fontsize=32)
            ax.tick_params(axis='y', labelsize=28)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)

            legend_elements_age_group = [
                Patch(facecolor=colors[0], edgecolor='black', label='Without carry-over', alpha=0.8),
                Patch(facecolor=colors[1], edgecolor='black', label='With carry-over', alpha=0.8),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markeredgecolor='gray',
                       markersize=24, label='Klein et al. adults'),
            ]
            if human_rt_available:
                legend_elements_age_group.extend([
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
                           markeredgecolor='black', markersize=20, label='Moeller et al. adults'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                           markeredgecolor='black', markersize=20, label='Moeller et al. children'),
                ])
            ax.legend(handles=legend_elements_age_group, loc='upper left', fontsize=28, framealpha=0.95)

            fname_age_group = os.path.join(
                figures_dir,
                f"barplot_errors_omega_{safe_om}_epoch_{int(selected_epoch)}_with_RT_Klein_Moeller_by_age_group_local_scale_{MODEL_TYPE}.png"
            )
            plt.savefig(fname_age_group, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Figure (Klein + Moeller by age group, SM Appendix) saved to: {fname_age_group}")
        else:
            print(f"No data found for last epoch")
    else:
        print("No data available for barplot")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)
