# USE: nohup python paper_anova_effects_sweep.py 2 STUDY RI argmax 0.15 100 500 50 > log_anova_sweep.out 2>&1 &
#
# CHANGES vs. the original script:
#   1. Removed the artificial np.repeat() expansion (pseudo-replication) that
#      inflated degrees of freedom without adding real information -- this
#      was the source of the unexplained df in the single-epoch script too.
#   2. F1-analog effect sizes (by-initialization, within-subject) are now
#      computed with paired t-tests across initializations at each epoch
#      (F(1, n-1) = t(n-1)^2), matching the design of Klein et al. (2010).
#   3. Added an F2-analog, BY-ITEM sweep: at each epoch, loads the model's
#      per-item error rate (pooled over initializations, over both
#      presentation orders), and correlates it against children's and
#      adults' item-level RT/zRT from Moeller's data. This produces the
#      developmental "crossing" analysis Moeller asked for: correlation
#      with children's data should be higher early in training, correlation
#      with adults' data higher later in training, if the model captures
#      developmental change.
#   4. Writes a combined Excel file for Korbinian with one Simulated_ER
#      column per swept epoch, plus a CSV/PNG of the correlation sweep.

import os
import re
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])
STUDY_NAME = str(sys.argv[2]).upper()
PARAM_TYPE = str(sys.argv[3]).upper()
MODEL_TYPE = str(sys.argv[4]).lower()
OMEGA_VALUE = float(sys.argv[5])
EPOCH_START = int(sys.argv[6])
EPOCH_END = int(sys.argv[7])
EPOCH_STEP = int(sys.argv[8]) if len(sys.argv) > 8 else 50

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
# ITEM-LEVEL DATA (per epoch), for the F2-analog sweep and the developmental
# correlation-crossing analysis.
#
# ASSUMPTION: for each epoch in the sweep, there exists a pooled per-item
# model error-rate CSV, one row per unique stimulus item (both presentation
# orders already averaged, all initializations already pooled). If your
# pipeline names these files differently per epoch, edit
# item_level_csv_for_epoch() below -- that is the ONLY place the naming
# convention is assumed.
# ---------------------------------------------------------------------------
ITEM_LEVEL_DIR = "../item_level_behavioral_validation"
ITEM_COL = "aufgabe"
ERROR_COL = "model_error_mean"

KIDS_XLS = "../datasets/RT_Analysis_Kids_II_modelling.xls"
ADULTS_XLS = "../datasets/RT_Analyses_Adults_modelling.xls"
ITEM_SHEET = "Itemanalyse"


def item_level_csv_for_epoch(epoch):
    """
    EDIT THIS if your per-epoch item-level files follow a different naming
    convention. Expected default:
        pooled_model_error_rates_mean_std_epoch_<epoch>.csv
    """
    return os.path.join(ITEM_LEVEL_DIR, f"pooled_model_error_rates_mean_std_epoch_{epoch}.csv")


def classify_item(aufgabe: str):
    """
    Same category logic as in paper_anova_effects.py (Klein et al., 2010).
    Robust to formatting variants such as '4+3', '4 + 3', or '4 + 3 =':
    extracts the first two integers found in the string.
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
        size = None
    return size, carry


def canonical_key(aufgabe: str) -> str:
    """Normalizes item-string formatting variants to a canonical 'a+b' merge key."""
    numbers = re.findall(r'\d+', str(aufgabe))
    if len(numbers) < 2:
        raise ValueError(f"Could not parse two operands from item string: {aufgabe!r}")
    return f"{int(numbers[0])}+{int(numbers[1])}"


def load_item_level_model_data(item_level_csv, item_col=ITEM_COL, error_col=ERROR_COL):
    if not os.path.exists(item_level_csv):
        return None
    df = pd.read_csv(item_level_csv)
    df = df.rename(columns={item_col: "aufgabe", error_col: "model_error"})
    df["aufgabe"] = df["aufgabe"].astype(str).str.strip()
    df["item_key"] = df["aufgabe"].apply(canonical_key)
    sizes, carries = [], []
    for a in df["aufgabe"]:
        size, carry = classify_item(a)
        sizes.append(size)
        carries.append(carry)
    df["Size"] = sizes
    df["Carry"] = carries
    df = df.dropna(subset=["Size"])
    return df


def paired_effect_stats(a, b):
    """Paired t-test -> (F(1,n-1), p, partial eta^2), a and b same length arrays."""
    n = len(a)
    t_stat, p_value = stats.ttest_rel(a, b)
    f_stat = t_stat ** 2
    df_error = n - 1
    eta2_p = f_stat / (f_stat + df_error) if (f_stat + df_error) > 0 else np.nan
    return f_stat, p_value, eta2_p, df_error


def analyze_single_epoch_f1(combined_logs, param_type, omega_value, epoch):
    """
    F1-analog (by-initialization) effect sizes for a single epoch, using
    paired t-tests across initializations -- NO artificial repetition.
    """
    subset_param = combined_logs[
        (combined_logs['param_init_type'] == param_type) &
        (combined_logs['omega'] == omega_value)
    ]
    if subset_param.empty:
        return None

    epoch_data = subset_param[subset_param['epoch'] == epoch].copy()
    if epoch_data.empty:
        return None

    epoch_data["ER_small_no_carry"] = 100 - epoch_data["test_pairs_no_carry_small_accuracy"]
    epoch_data["ER_small_carry"] = 100 - epoch_data["test_pairs_carry_small_accuracy"]
    epoch_data["ER_large_no_carry"] = 100 - epoch_data["test_pairs_no_carry_large_accuracy"]
    epoch_data["ER_large_carry"] = 100 - epoch_data["test_pairs_carry_large_accuracy"]
    epoch_data = epoch_data.dropna(
        subset=["ER_small_no_carry", "ER_small_carry", "ER_large_no_carry", "ER_large_carry"]
    ).reset_index(drop=True)

    if len(epoch_data) < 2:
        return None

    n_inits = len(epoch_data)

    # Carry main effect
    er_carry = epoch_data[["ER_small_carry", "ER_large_carry"]].mean(axis=1).to_numpy()
    er_no_carry = epoch_data[["ER_small_no_carry", "ER_large_no_carry"]].mean(axis=1).to_numpy()
    carry_f, carry_p, carry_eta2, df_error = paired_effect_stats(er_carry, er_no_carry)

    # Size main effect
    er_large = epoch_data[["ER_large_no_carry", "ER_large_carry"]].mean(axis=1).to_numpy()
    er_small = epoch_data[["ER_small_no_carry", "ER_small_carry"]].mean(axis=1).to_numpy()
    size_f, size_p, size_eta2, _ = paired_effect_stats(er_large, er_small)

    # Interaction
    carry_effect_small = (epoch_data["ER_small_carry"] - epoch_data["ER_small_no_carry"]).to_numpy()
    carry_effect_large = (epoch_data["ER_large_carry"] - epoch_data["ER_large_no_carry"]).to_numpy()
    interaction_f, interaction_p, interaction_eta2, _ = paired_effect_stats(carry_effect_large, carry_effect_small)

    return {
        'epoch': epoch,
        'n_inits': n_inits,
        'df_error': df_error,
        'carry_f': carry_f, 'carry_p': carry_p, 'carry_eta2': carry_eta2,
        'size_f': size_f, 'size_p': size_p, 'size_eta2': size_eta2,
        'interaction_f': interaction_f, 'interaction_p': interaction_p, 'interaction_eta2': interaction_eta2,
        'mean_small_no_carry': epoch_data["ER_small_no_carry"].mean(),
        'mean_small_carry': epoch_data["ER_small_carry"].mean(),
        'mean_large_no_carry': epoch_data["ER_large_no_carry"].mean(),
        'mean_large_carry': epoch_data["ER_large_carry"].mean(),
    }


def analyze_single_epoch_f2(epoch, kids_item_df, adults_item_df):
    """
    F2-analog (by-item) analysis for a single epoch:
      - one-way between-items ANOVA for Carry and for Size
      - Pearson correlation of model per-item error against children's and
        adults' item-level RT (and zRT)
    Returns None if the per-epoch item-level model file is not found.
    """
    item_csv = item_level_csv_for_epoch(epoch)
    item_df = load_item_level_model_data(item_csv)
    if item_df is None:
        return None

    result = {'epoch': epoch, 'n_items': len(item_df)}

    for factor_col, prefix in [("Carry", "item_carry"), ("Size", "item_size")]:
        levels = sorted(item_df[factor_col].unique(), key=str)
        groups = [item_df.loc[item_df[factor_col] == lvl, "model_error"].to_numpy() for lvl in levels]
        f_stat, p_value = stats.f_oneway(*groups)
        grand_mean = item_df["model_error"].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((item_df["model_error"] - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
        result[f"{prefix}_f"] = f_stat
        result[f"{prefix}_p"] = p_value
        result[f"{prefix}_eta2"] = eta2

    # Correlations with children's / adults' item-level data, if provided.
    for label, human_df, rt_col, zrt_col in [
        ("children", kids_item_df, "RT_kids", "zRT_kids"),
        ("adults", adults_item_df, "RT_adults", "zRT_adults"),
    ]:
        if human_df is None:
            continue
        merged = item_df.merge(human_df, on="item_key", how="inner")
        for col, tag in [(rt_col, "RT"), (zrt_col, "zRT")]:
            valid = merged[["model_error", col]].dropna()
            if len(valid) >= 3 and valid["model_error"].std() > 0:
                r, p = pearsonr(valid["model_error"], valid[col])
            else:
                r, p = np.nan, np.nan
            result[f"corr_{label}_{tag}_r"] = r
            result[f"corr_{label}_{tag}_p"] = p

    result["_item_df"] = item_df  # kept temporarily for the Excel export step
    return result


def load_human_item_data():
    """Loads Kids/Adults item-level RT and zRT, keyed on canonical 'item_key'."""
    kids_df, adults_df = None, None
    if os.path.exists(KIDS_XLS):
        kids_raw = pd.read_excel(KIDS_XLS, sheet_name=ITEM_SHEET)
        kids_raw["aufgabe"] = kids_raw["aufgabe"].astype(str).str.strip()
        kids_raw["item_key"] = kids_raw["aufgabe"].apply(canonical_key)
        kids_df = kids_raw[["item_key", "RT", "zRT"]].rename(columns={"RT": "RT_kids", "zRT": "zRT_kids"})
    if os.path.exists(ADULTS_XLS):
        adults_raw = pd.read_excel(ADULTS_XLS, sheet_name=ITEM_SHEET)
        adults_raw["aufgabe"] = adults_raw["aufgabe"].astype(str).str.strip()
        adults_raw["item_key"] = adults_raw["aufgabe"].apply(canonical_key)
        adults_df = adults_raw[["item_key", "RT", "zRT"]].rename(columns={"RT": "RT_adults", "zRT": "zRT_adults"})
    return kids_df, adults_df


def run_epoch_sweep(raw_dir, figures_dir, omega_value, param_type, epoch_start, epoch_end, epoch_step):
    os.makedirs(figures_dir, exist_ok=True)
    safe_om = str(omega_value).replace('.', '_')

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    combined_logs = None
    if os.path.exists(combined_logs_path):
        combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
        combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
        combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')
        acc_cols = [
            "test_pairs_no_carry_small_accuracy", "test_pairs_carry_small_accuracy",
            "test_pairs_no_carry_large_accuracy", "test_pairs_carry_large_accuracy",
        ]
        for col in acc_cols:
            if col in combined_logs.columns:
                combined_logs[col] = pd.to_numeric(combined_logs[col], errors='coerce')
    else:
        print(f"[WARNING] No combined_logs.csv found in {raw_dir} -- F1-analog sweep will be skipped.")

    kids_item_df, adults_item_df = load_human_item_data()
    if kids_item_df is None:
        print(f"[WARNING] Kids Excel not found at {KIDS_XLS} -- children correlations will be skipped.")
    if adults_item_df is None:
        print(f"[WARNING] Adults Excel not found at {ADULTS_XLS} -- adult correlations will be skipped.")

    epochs = list(range(epoch_start, epoch_end + 1, epoch_step))
    f1_results, f2_results = [], []
    excel_columns = {}  # epoch -> per-item Series, for the combined Korbinian Excel

    for epoch in epochs:
        print(f"Analyzing epoch {epoch}...")
        if combined_logs is not None:
            r1 = analyze_single_epoch_f1(combined_logs, param_type, omega_value, epoch)
            if r1 is not None:
                f1_results.append(r1)

        r2 = analyze_single_epoch_f2(epoch, kids_item_df, adults_item_df)
        if r2 is not None:
            item_df = r2.pop("_item_df")
            excel_columns[epoch] = item_df.set_index("item_key")["model_error"]
            f2_results.append(r2)

    df_f1 = pd.DataFrame(f1_results)
    df_f2 = pd.DataFrame(f2_results)

    # ------------------------------------------------------------------
    # TXT summary
    # ------------------------------------------------------------------
    output_filename = f"ANOVA_sweep_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.txt"
    output_path = os.path.join(figures_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EPOCH SWEEP -- F1-ANALOG (BY INITIALIZATION) AND F2-ANALOG (BY ITEM)\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nConfiguration: {NUMBER_SIZE}-digit, {STUDY_NAME}, {param_type}, {MODEL_TYPE}\n")
        f.write(f"Omega: {omega_value}\n")
        f.write(f"Epoch range: {epoch_start} to {epoch_end} (step: {epoch_step})\n")
        f.write("\nMethod notes:\n")
        f.write(" - F1-analog: paired t-tests across model initializations at each epoch\n")
        f.write("   (F(1,n-1) = t(n-1)^2). No artificial repetition/expansion is applied.\n")
        f.write(" - F2-analog: one-way between-items ANOVA across the 96 unique stimulus\n")
        f.write("   items (model error pooled over initializations and presentation order),\n")
        f.write("   plus Pearson correlations against children's/adults' item-level RT/zRT.\n")

        if not df_f1.empty:
            f.write("\n" + "-" * 80 + "\n")
            f.write("F1-ANALOG RESULTS BY EPOCH\n")
            f.write("-" * 80 + "\n")
            for _, row in df_f1.iterrows():
                f.write(f"\nEpoch {int(row['epoch'])} (n_inits={int(row['n_inits'])}, df_error={int(row['df_error'])}):\n")
                f.write(f"  Carry:       F(1,{int(row['df_error'])})={row['carry_f']:.4f}, p={row['carry_p']:.6f}, "
                        f"partial eta^2={row['carry_eta2']:.4f}\n")
                f.write(f"  Size:        F(1,{int(row['df_error'])})={row['size_f']:.4f}, p={row['size_p']:.6f}, "
                        f"partial eta^2={row['size_eta2']:.4f}\n")
                f.write(f"  Interaction: F(1,{int(row['df_error'])})={row['interaction_f']:.4f}, "
                        f"p={row['interaction_p']:.6f}, partial eta^2={row['interaction_eta2']:.4f}\n")
        else:
            f.write("\n[No F1-analog results -- combined_logs.csv not found or empty]\n")

        if not df_f2.empty:
            f.write("\n" + "-" * 80 + "\n")
            f.write("F2-ANALOG RESULTS BY EPOCH (BY-ITEM)\n")
            f.write("-" * 80 + "\n")
            for _, row in df_f2.iterrows():
                f.write(f"\nEpoch {int(row['epoch'])} (n_items={int(row['n_items'])}):\n")
                if "item_carry_f" in row:
                    f.write(f"  Carry (by-item):  F={row['item_carry_f']:.4f}, p={row['item_carry_p']:.6f}, "
                            f"eta^2={row['item_carry_eta2']:.4f}\n")
                    f.write(f"  Size  (by-item):  F={row['item_size_f']:.4f}, p={row['item_size_p']:.6f}, "
                            f"eta^2={row['item_size_eta2']:.4f}\n")
                for label in ["children", "adults"]:
                    for tag in ["RT", "zRT"]:
                        rkey, pkey = f"corr_{label}_{tag}_r", f"corr_{label}_{tag}_p"
                        if rkey in row and pd.notna(row[rkey]):
                            f.write(f"  Corr. with {label} {tag}: r={row[rkey]:.3f}, p={row[pkey]:.6f}\n")
        else:
            f.write("\n[No F2-analog results -- check ITEM_LEVEL_DIR / item_level_csv_for_epoch()]\n")

    print(f"\nTXT results saved to: {output_path}")

    # ------------------------------------------------------------------
    # CSVs
    # ------------------------------------------------------------------
    if not df_f1.empty:
        csv_f1 = os.path.join(figures_dir, f"ANOVA_sweep_F1_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.csv")
        df_f1.to_csv(csv_f1, index=False)
        print(f"F1-analog sweep CSV saved to: {csv_f1}")

    if not df_f2.empty:
        csv_f2 = os.path.join(figures_dir, f"ANOVA_sweep_F2_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.csv")
        df_f2.to_csv(csv_f2, index=False)
        print(f"F2-analog sweep CSV saved to: {csv_f2}")

    # ------------------------------------------------------------------
    # Developmental crossing plot: correlation with children vs. adults
    # over training (this is the analysis Moeller specifically asked for).
    # ------------------------------------------------------------------
    if not df_f2.empty and any(c.startswith("corr_") for c in df_f2.columns):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, tag in zip(axes, ["RT", "zRT"]):
            ck, ca = f"corr_children_{tag}_r", f"corr_adults_{tag}_r"
            if ck in df_f2.columns:
                ax.plot(df_f2["epoch"], df_f2[ck], 'o-', color="#66CC66", linewidth=2.5,
                        markersize=7, label="Children")
            if ca in df_f2.columns:
                ax.plot(df_f2["epoch"], df_f2[ca], 's-', color="#9966CC", linewidth=2.5,
                        markersize=7, label="Adults")
            ax.set_xlabel("Training batch (epoch)", fontsize=16)
            ax.set_ylabel(f"Pearson $r$ (model error vs. human {tag})", fontsize=16)
            ax.set_title(f"Model-Human Correlation Over Training ({tag})", fontsize=17)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.legend(fontsize=13)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"developmental_correlation_sweep_{STUDY_NAME}_omega_{safe_om}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Developmental correlation figure saved to: {fig_path}")

    # Effect-size sweep plot (F1-analog), analogous to the original figure.
    if not df_f1.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(df_f1['epoch'], df_f1['carry_eta2'], 'o-', label='Carry Effect (partial eta^2)',
                color='#1C6CE5', linewidth=2.5, markersize=7)
        ax.plot(df_f1['epoch'], df_f1['size_eta2'], 's-', label='Problem Size Effect (partial eta^2)',
                color='#D62828', linewidth=2.5, markersize=7)
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Partial eta^2 (by-initialization)', fontsize=16)
        ax.set_title('F1-analog Effect Sizes Across Epochs', fontsize=17)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"ANOVA_sweep_F1_{STUDY_NAME}_omega_{safe_om}_epochs_{epoch_start}-{epoch_end}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"F1-analog effect-size figure saved to: {fig_path}")

    # ------------------------------------------------------------------
    # Combined Excel for Korbinian: one Simulated_ER_epoch_<N> column per
    # swept epoch, appended to his Kids/Adults files.
    # ------------------------------------------------------------------
    if excel_columns:
        sim_wide = pd.DataFrame(excel_columns)
        sim_wide.columns = [f"Simulated_ER_epoch_{e}" for e in sim_wide.columns]
        sim_wide = sim_wide.reset_index().rename(columns={"index": "item_key"})

        for xls_path, human_df, label in [
            (KIDS_XLS, kids_item_df, "Kids"),
            (ADULTS_XLS, adults_item_df, "Adults"),
        ]:
            if human_df is None or not os.path.exists(xls_path):
                continue
            full_human_df = pd.read_excel(xls_path, sheet_name=ITEM_SHEET)
            full_human_df["aufgabe"] = full_human_df["aufgabe"].astype(str).str.strip()
            full_human_df["item_key"] = full_human_df["aufgabe"].apply(canonical_key)
            merged = full_human_df.merge(sim_wide, on="item_key", how="left").drop(columns=["item_key"])

            out_xls_path = os.path.join(
                figures_dir,
                f"{os.path.splitext(os.path.basename(xls_path))[0]}_with_sweep_{STUDY_NAME}_omega_{safe_om}.xlsx"
            )
            merged.to_excel(out_xls_path, sheet_name=ITEM_SHEET, index=False)
            print(f"Combined sweep Excel for Korbinian ({label}) saved to: {out_xls_path}")
    else:
        print("[WARNING] No per-epoch item-level model files were found -- "
              "no Excel export generated. Check item_level_csv_for_epoch().")


run_epoch_sweep(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE, EPOCH_START, EPOCH_END, EPOCH_STEP)