# USE: nohup python paper_anova_effects.py 2 STUDY RI argmax 0.15 500 > log_anova_effects.out 2>&1 &
#
# CHANGES vs. the original script:
#   1. Removed the artificial np.repeat() expansion that was pseudo-replicating
#      each averaged accuracy value N_REPETITIONS_PER_SAMPLE times before
#      handing it to f_oneway(). That trick inflated the between-subjects df
#      without adding any real information, which is what produced the
#      unexplained F(1,164) Moeller flagged.
#   2. Added a proper "F1-analog" analysis: a within-subject (repeated-measures)
#      test across model initializations, matching the original Klein et al.
#      design (N subjects = N initializations, df = (1, n-1)). For a 2-level
#      factor, RM-ANOVA with 1 df is algebraically identical to a paired
#      t-test (F = t^2), so we use scipy.stats.ttest_rel directly -- simple,
#      exact, and easy to sanity-check.
#   3. Added a proper "F2-analog" analysis: a one-way, BETWEEN-items ANOVA
#      across the 96 unique stimulus items (as suggested by Moeller), using
#      the model's per-item error rate pooled over initializations and over
#      the two presentation orders (a,b)/(b,a) of each item.
#   4. Added Excel export: merges the model's per-item simulated error rate
#      into copies of the Kids/Adults Excel files (same "aufgabe" key,
#      "Itemanalyse" sheet) so they can be sent back to Korbinian directly.
#   5. All results (F1-analog AND F2-analog) are written to a single,
#      clearly-labeled txt file so there is no more ambiguity about which
#      analysis produced which numbers.
#   6. [NEW] FIXED a units/scale bug: pooled_model_error_rates_mean_std.csv
#      stores "model_error_mean" as a PROPORTION in [0, 1] (e.g. 0.4285714
#      for an item with error rate 42.86%), but the F2-analog printouts
#      appended a literal "%" sign to that raw value without multiplying by
#      100 first -- so a true 43.25% error rate was being printed/exported
#      as "0.43%". This did NOT affect the F-statistics, p-values, or eta^2
#      reported for the F2-analog ANOVA (those are scale-invariant), but it
#      DID make every printed/exported mean and SD 100x too small. Fixed by
#      converting model_error to a 0-100 percentage immediately on load, in
#      load_item_level_model_data(). The Excel export to Korbinian now also
#      carries the corrected percentage-scale values.

import os
import re
import sys
import pandas as pd
import numpy as np
from scipy import stats

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study
PARAM_TYPE = str(sys.argv[3]).upper()  # 'WI' or 'RI'
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector'
OMEGA_VALUE = float(sys.argv[5])  # Specific omega value to analyze
EPOCH = int(sys.argv[6]) if len(sys.argv) > 6 else "last"  # Specific epoch for analysis

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"
FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"

# ---------------------------------------------------------------------------
# ITEM-LEVEL DATA (for the F2-analog, by-item ANOVA and the Korbinian Excel).
#
# ASSUMPTION: this CSV already contains ONE ROW PER UNIQUE STIMULUS ITEM
# (i.e. the two presentation orders (a,b)/(b,a) already collapsed/averaged),
# with the model's error rate pooled over all initializations, for the
# specific (omega, epoch, param_type, model_type) configuration being
# analyzed. This mirrors "pooled_model_error_rates_mean_std.csv" used in
# paper_figure_item_level_validation.py.
#
# Required columns:
#   "aufgabe"          - item string, e.g. "4+3" (must match the Kids/Adults
#                         Excel "aufgabe" column exactly)
#   "model_error_mean"  - model error rate for that item, pooled over inits
#                         and over both presentation orders. STORED AS A
#                         PROPORTION IN [0, 1] -- converted to a 0-100
#                         percentage on load (see FIX #6 above).
#
# If your pipeline names things differently, just edit ITEM_LEVEL_CSV and
# the two column names below (ITEM_COL / ERROR_COL). If a future version of
# the CSV already stores percentages directly, set ERROR_COL_IS_PROPORTION
# to False to disable the x100 conversion.
# ---------------------------------------------------------------------------
ITEM_LEVEL_DIR = "../item_level_behavioral_validation"
ITEM_LEVEL_CSV = os.path.join(ITEM_LEVEL_DIR, "pooled_model_error_rates_mean_std.csv")
ITEM_COL = "aufgabe"
ERROR_COL = "model_error_mean"
ERROR_COL_IS_PROPORTION = True  # set False if the CSV already stores 0-100 percentages

KIDS_XLS = "../datasets/RT_Analysis_Kids_II_modelling.xls"
ADULTS_XLS = "../datasets/RT_Analyses_Adults_modelling.xls"
ITEM_SHEET = "Itemanalyse"


def classify_item(aufgabe: str):
    """
    Derive (size, carry) category for a stimulus item string, robust to
    formatting variants such as '4+3', '4 + 3', or '4 + 3 =' (with a
    trailing equals sign and/or extra whitespace, as in the pooled model
    CSV). Extracts the first two integers found in the string, following
    exactly the category definitions used in Klein et al. (2010):
      - carry: True if the sum of the unit digits of the two addends >= 10
      - size: 'Small' if a+b < 40, 'Large' if a+b > 60
        (teen-range problems with 40 <= sum <= 60 do not occur in the
        96-item critical stimulus set, so any such item is flagged and
        excluded rather than silently mis-binned)
    """
    numbers = re.findall(r'\d+', str(aufgabe))
    if len(numbers) < 2:
        raise ValueError(f"Could not parse two operands from item string: {aufgabe!r}")
    a, b = int(numbers[0]), int(numbers[1])
    unit_sum = (a % 10) + (b % 10)
    carry = unit_sum >= 10
    total = a + b
    if total < 40:
        size = "Small"
    elif total > 60:
        size = "Large"
    else:
        size = None  # ambiguous / not part of the critical 96-item set
    return size, carry


def canonical_key(aufgabe: str) -> str:
    """
    Normalizes any item-string formatting variant ('4+3', '4 + 3', '4 + 3 =',
    with arbitrary whitespace) to a single canonical 'a+b' key, so that the
    model's item CSV and Korbinian's Excel files can be merged reliably even
    if their exact string formatting differs.
    """
    numbers = re.findall(r'\d+', str(aufgabe))
    if len(numbers) < 2:
        raise ValueError(f"Could not parse two operands from item string: {aufgabe!r}")
    return f"{int(numbers[0])}+{int(numbers[1])}"


def load_item_level_model_data(item_level_csv, item_col, error_col):
    """
    Load per-item model error rates and attach (size, carry) category labels.
    Returns a DataFrame with columns: aufgabe (original string), item_key
    (canonical 'a+b' merge key), model_error (0-100 PERCENTAGE SCALE), Size,
    Carry.
    """
    if not os.path.exists(item_level_csv):
        return None

    df = pd.read_csv(item_level_csv)
    df = df.rename(columns={item_col: "aufgabe", error_col: "model_error"})
    df["aufgabe"] = df["aufgabe"].astype(str).str.strip()
    df["item_key"] = df["aufgabe"].apply(canonical_key)

    # --- FIX #6: convert proportion [0,1] -> percentage [0,100] -------------
    # The source CSV (pooled_model_error_rates_mean_std.csv) stores error
    # rate as a proportion (e.g. 0.4286), not a percentage. Every downstream
    # print/export in this script formats "model_error" with a literal "%"
    # suffix, so failing to rescale here silently produces values 100x too
    # small (e.g. "0.43%" printed instead of "43.25%"). This rescaling does
    # NOT change the ANOVA F-statistics, p-values, or eta^2 below, since
    # those are invariant to a constant multiplicative rescaling of the
    # dependent variable -- only the reported means/SDs (and the Excel
    # export to Korbinian) were affected.
    if ERROR_COL_IS_PROPORTION:
        df["model_error"] = df["model_error"] * 100.0

    sizes, carries = [], []
    for a in df["aufgabe"]:
        size, carry = classify_item(a)
        sizes.append(size)
        carries.append(carry)
    df["Size"] = sizes
    df["Carry"] = carries

    n_before = len(df)
    df = df.dropna(subset=["Size"])
    n_after = len(df)
    if n_after < n_before:
        print(f"[WARNING] Dropped {n_before - n_after} items that did not fall into "
              f"the Small(<40)/Large(>60) bins used by Klein et al. (2010).")

    return df


def paired_effect(df_wide, col_a, col_b, label):
    """
    Within-subject (repeated-measures) test for a 2-level factor, computed
    as a paired t-test across model initializations. For a single 1-df
    factor this is exactly equivalent to a repeated-measures ANOVA main
    effect: F(1, n-1) = t(n-1)^2, with the identical p-value.

    df_wide: one row per initialization ("subject"), with columns col_a and
             col_b holding that subject's error rate in each level.

    NOTE: the F1-analog inputs (ER_small_no_carry, etc.) are already
    computed as 100 - accuracy in perform_anova_analysis(), i.e. already on
    the correct 0-100 percentage scale. Only the F2-analog (item-level) path
    had the proportion/percentage scale bug fixed above.
    """
    a = df_wide[col_a].to_numpy()
    b = df_wide[col_b].to_numpy()
    n = len(a)
    t_stat, p_value = stats.ttest_rel(a, b)
    f_stat = t_stat ** 2
    df_error = n - 1

    diff = a - b
    # Partial eta-squared for a paired design: SS_effect / (SS_effect + SS_error)
    # equivalently derivable from t^2: eta2_p = t^2 / (t^2 + df_error)
    eta2_p = f_stat / (f_stat + df_error) if (f_stat + df_error) > 0 else np.nan

    print(f"\n{label}:")
    print(f"  N (initializations) = {n}")
    print(f"  Mean {col_a} = {a.mean():.3f}%, Mean {col_b} = {b.mean():.3f}%")
    print(f"  F(1, {df_error}) = {f_stat:.4f}, p = {p_value:.6f}, partial eta^2 = {eta2_p:.4f}")

    return {
        "label": label,
        "n": n,
        "df_error": df_error,
        "f_stat": f_stat,
        "t_stat": t_stat,
        "p_value": p_value,
        "eta2_p": eta2_p,
        "mean_a": a.mean(),
        "mean_b": b.mean(),
    }


def perform_anova_analysis(raw_dir, figures_dir, omega_value, param_type, epoch):
    """
    Runs BOTH the F1-analog (by-initialization, within-subject) analysis and
    the F2-analog (by-item, between-items) analysis for a single epoch, and
    writes a single results file plus Excel files for Korbinian.
    """
    os.makedirs(figures_dir, exist_ok=True)

    safe_om = str(omega_value).replace('.', '_')
    epoch_tag = "last_epoch" if epoch == "last" else f"epoch_{epoch}"
    output_path = os.path.join(figures_dir, f"ANOVA_results_{STUDY_NAME}_omega_{safe_om}_{epoch_tag}.txt")
    output_file = open(output_path, 'w')

    def print_both(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_file)

    # =========================================================================
    # PART 1: F1-ANALOG -- BY INITIALIZATION (within-subject, matches Klein
    # et al.'s original participant-level design; "subject" = one model
    # initialization).
    # =========================================================================
    print_both("=" * 80)
    print_both("PART 1: F1-ANALOG ANOVA -- BY MODEL INITIALIZATION")
    print_both("=" * 80)
    print_both("Each model initialization is treated as a 'subject', exactly as each")
    print_both("human participant was a subject in Klein et al. (2010). Main effects")
    print_both("of Carry and Problem Size, and their interaction, are tested WITHIN")
    print_both("subject (paired t-tests, equivalent to a 1-df repeated-measures ANOVA:")
    print_both("F(1, n-1) = t(n-1)^2). No artificial repetition/expansion is applied.")

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print_both(f"\n[SKIPPED] No combined_logs.csv found in {raw_dir}")
    else:
        combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
        combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
        combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')

        acc_cols = [
            "test_pairs_no_carry_small_accuracy",
            "test_pairs_carry_small_accuracy",
            "test_pairs_no_carry_large_accuracy",
            "test_pairs_carry_large_accuracy",
        ]
        for col in acc_cols:
            if col in combined_logs.columns:
                combined_logs[col] = pd.to_numeric(combined_logs[col], errors='coerce')

        subset_param = combined_logs[
            (combined_logs['param_init_type'] == param_type) &
            (combined_logs['omega'] == omega_value)
        ]

        if subset_param.empty:
            print_both(f"\n[SKIPPED] No data found for param_type={param_type} and omega={omega_value}")
        else:
            selected_epoch = subset_param['epoch'].max() if epoch == "last" else epoch
            epoch_data = subset_param[subset_param['epoch'] == selected_epoch].copy()

            if epoch_data.empty:
                print_both(f"\n[SKIPPED] No data found for epoch {selected_epoch}")
            else:
                # Convert accuracy -> error rate. Each row here is ALREADY one
                # model initialization's own accuracy for that category -- no
                # additional averaging/expansion needed or applied. These
                # test_pairs_*_accuracy columns are already on a 0-100 scale,
                # so "100 - accuracy" is already a correct 0-100 percentage
                # (unlike the F2-analog item CSV, which stores a 0-1
                # proportion -- see FIX #6 in load_item_level_model_data()).
                epoch_data["ER_small_no_carry"] = 100 - epoch_data["test_pairs_no_carry_small_accuracy"]
                epoch_data["ER_small_carry"] = 100 - epoch_data["test_pairs_carry_small_accuracy"]
                epoch_data["ER_large_no_carry"] = 100 - epoch_data["test_pairs_no_carry_large_accuracy"]
                epoch_data["ER_large_carry"] = 100 - epoch_data["test_pairs_carry_large_accuracy"]
                epoch_data = epoch_data.dropna(
                    subset=["ER_small_no_carry", "ER_small_carry", "ER_large_no_carry", "ER_large_carry"]
                ).reset_index(drop=True)

                n_inits = len(epoch_data)
                print_both(f"\nEpoch analyzed: {selected_epoch}")
                print_both(f"Number of model initializations ('subjects'): {n_inits}")

                print_both("\nDESCRIPTIVE STATISTICS (mean error rate across initializations)")
                for col, name in [
                    ("ER_small_no_carry", "Small - No Carry"),
                    ("ER_small_carry", "Small - Carry"),
                    ("ER_large_no_carry", "Large - No Carry"),
                    ("ER_large_carry", "Large - Carry"),
                ]:
                    print_both(f"  {name}: Mean = {epoch_data[col].mean():.2f}%, SD = {epoch_data[col].std():.2f}%")

                # --- Main effect of Carry: average across size, paired across subjects ---
                epoch_data["ER_no_carry_avg"] = epoch_data[["ER_small_no_carry", "ER_large_no_carry"]].mean(axis=1)
                epoch_data["ER_carry_avg"] = epoch_data[["ER_small_carry", "ER_large_carry"]].mean(axis=1)
                res_carry = paired_effect(epoch_data, "ER_carry_avg", "ER_no_carry_avg", "MAIN EFFECT: Carry-over")

                # --- Main effect of Size: average across carry, paired across subjects ---
                epoch_data["ER_large_avg"] = epoch_data[["ER_large_no_carry", "ER_large_carry"]].mean(axis=1)
                epoch_data["ER_small_avg"] = epoch_data[["ER_small_no_carry", "ER_small_carry"]].mean(axis=1)
                res_size = paired_effect(epoch_data, "ER_large_avg", "ER_small_avg", "MAIN EFFECT: Problem Size")

                # --- Interaction: (Small_Carry - Small_NoCarry) vs (Large_Carry - Large_NoCarry) ---
                epoch_data["carry_effect_small"] = epoch_data["ER_small_carry"] - epoch_data["ER_small_no_carry"]
                epoch_data["carry_effect_large"] = epoch_data["ER_large_carry"] - epoch_data["ER_large_no_carry"]
                res_interaction = paired_effect(
                    epoch_data, "carry_effect_large", "carry_effect_small",
                    "INTERACTION: Size x Carry (is the carry effect larger for Large problems?)"
                )

                print_both("\n--- F1-analog summary ---")
                print_both(f"Carry-over effect:  F(1,{res_carry['df_error']}) = {res_carry['f_stat']:.4f}, "
                           f"p = {res_carry['p_value']:.6f}, partial eta^2 = {res_carry['eta2_p']:.4f}")
                print_both(f"Problem size effect: F(1,{res_size['df_error']}) = {res_size['f_stat']:.4f}, "
                           f"p = {res_size['p_value']:.6f}, partial eta^2 = {res_size['eta2_p']:.4f}")
                print_both(f"Interaction effect:  F(1,{res_interaction['df_error']}) = {res_interaction['f_stat']:.4f}, "
                           f"p = {res_interaction['p_value']:.6f}, partial eta^2 = {res_interaction['eta2_p']:.4f}")

    # =========================================================================
    # PART 2: F2-ANALOG -- BY ITEM (between-items, as suggested by Moeller;
    # matches the item-level analysis Klein/Moeller's item data supports).
    # =========================================================================
    print_both("\n\n" + "=" * 80)
    print_both("PART 2: F2-ANALOG ANOVA -- BY STIMULUS ITEM")
    print_both("=" * 80)
    print_both("Each of the 96 unique stimulus items is treated as the unit of analysis.")
    print_both("Model error rate per item is pooled over all initializations AND over")
    print_both("both presentation orders (a,b)/(b,a) of that item. Carry and Size are")
    print_both("tested as BETWEEN-items one-way ANOVAs (independent groups of items),")
    print_both("as requested by Moeller, so this is directly comparable to a one-way")
    print_both("ANOVA run on the empirical item-level RT/ER data.")

    item_df = load_item_level_model_data(ITEM_LEVEL_CSV, ITEM_COL, ERROR_COL)

    if item_df is None:
        print_both(f"\n[SKIPPED] No item-level model data found at {ITEM_LEVEL_CSV}.")
        print_both("Update ITEM_LEVEL_CSV / ITEM_COL / ERROR_COL at the top of this script")
        print_both("to point at your pooled per-item model error rate file.")
    else:
        print_both(f"\nLoaded {len(item_df)} unique items from {ITEM_LEVEL_CSV}")

        for factor_col, factor_name in [("Carry", "Carry-over"), ("Size", "Problem Size")]:
            levels = sorted(item_df[factor_col].unique(), key=str)
            groups = [item_df.loc[item_df[factor_col] == lvl, "model_error"].to_numpy() for lvl in levels]
            f_stat, p_value = stats.f_oneway(*groups)

            grand_mean = item_df["model_error"].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((item_df["model_error"] - grand_mean) ** 2)
            eta2 = ss_between / ss_total if ss_total > 0 else np.nan

            print_both(f"\nMAIN EFFECT (by-item): {factor_name}")
            for lvl, g in zip(levels, groups):
                print_both(f"  {lvl}: Mean = {g.mean():.2f}%, SD = {g.std():.2f}%, N items = {len(g)}")
            df_between = len(groups) - 1
            df_within = len(item_df) - len(groups)
            print_both(f"  F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_value:.6f}, eta^2 = {eta2:.4f}")

        # 2x2 between-items ANOVA (Size x Carry), for completeness / interaction term
        print_both("\nTWO-WAY BETWEEN-ITEMS ANOVA (Size x Carry)")
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            model = ols("model_error ~ C(Size) * C(Carry)", data=item_df).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            print_both(aov_table.to_string())

            # Report partial eta^2 for each term alongside the SS table, so
            # this matches the F1-analog reporting format and the eta^2
            # printed above is not the only effect-size number in the file.
            ss_resid = aov_table.loc["Residual", "sum_sq"]
            print_both("\nPartial eta^2 per term (SS_effect / (SS_effect + SS_residual)):")
            for term in aov_table.index:
                if term == "Residual":
                    continue
                ss_term = aov_table.loc[term, "sum_sq"]
                eta2_p = ss_term / (ss_term + ss_resid)
                print_both(f"  {term}: partial eta^2 = {eta2_p:.4f}")
        except ImportError:
            print_both("[NOTE] statsmodels not available -- skipping the 2-way interaction term.")
            print_both("       Install statsmodels to get the Size x Carry interaction F-test.")

    output_file.close()
    print(f"\nResults saved to: {output_path}")

    # =========================================================================
    # PART 3: EXCEL EXPORT FOR KORBINIAN
    # Adds a "Simulated_ER_<config>" column to copies of his Kids/Adults Excel
    # files, matched on the "aufgabe" item string. NOTE: this now exports the
    # corrected 0-100 percentage-scale values (see FIX #6), not the raw
    # 0-1 proportions that the original script would have exported.
    # =========================================================================
    if item_df is not None:
        config_tag = f"{STUDY_NAME}_{param_type}_omega{safe_om}_{epoch_tag}"
        sim_col_name = f"Simulated_ER_pct_{config_tag}"

        for xls_path, label in [(KIDS_XLS, "Kids"), (ADULTS_XLS, "Adults")]:
            if not os.path.exists(xls_path):
                print(f"[SKIPPED] {label} Excel not found at {xls_path}")
                continue

            human_df = pd.read_excel(xls_path, sheet_name=ITEM_SHEET)
            human_df["aufgabe"] = human_df["aufgabe"].astype(str).str.strip()
            human_df["item_key"] = human_df["aufgabe"].apply(canonical_key)

            merged = human_df.merge(
                item_df[["item_key", "model_error"]].rename(columns={"model_error": sim_col_name}),
                on="item_key", how="left"
            ).drop(columns=["item_key"])

            n_missing = merged[sim_col_name].isna().sum()
            if n_missing > 0:
                print(f"[WARNING] {n_missing} items in {label} Excel had no matching model item "
                      f"(check 'aufgabe' formatting consistency).")

            out_xls_path = os.path.join(
                figures_dir, f"{os.path.splitext(os.path.basename(xls_path))[0]}_with_{config_tag}.xlsx"
            )
            merged.to_excel(out_xls_path, sheet_name=ITEM_SHEET, index=False)
            print(f"Excel for Korbinian ({label}) saved to: {out_xls_path}")


# Run analysis
perform_anova_analysis(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE, EPOCH)