"""
Item-Level Behavioral Validation across Development
=====================================================

USE:

nohup python item_level_behavioral_validation.py > item_level_behavioral_validation.log 2>&1 &


Correlates the trained 2-digit addition decision-module's item-level error
rates against human behavioral data (RT, zRT, ER) collected from the
`Itemanalyse` sheets of the Kids_II and Adults RT-analysis Excel files, for
every trained epsilon value (0.00 -> 10.00, step 0.50) at Weber fraction
Omega = 0.10, using the batch-600 checkpoint.

Pipeline
--------
1.  Parse the 96 addition items ("aufgabe" column, e.g. "4 + 3 =") from both
    Excel files and build the corresponding model input tensor
    x = [tens1, units1, tens2, units2] for each item.
2.  For each epsilon in {0.00, 0.50, ..., 10.00}:
        - scan every Training_<timestamp> folder under
          .../RI/argmax_version/epsilon_<eps>/
        - keep only the folders whose config.txt reports
          "Weber fraction (Omega): 0.10"
        - load trained_model_checkpoint_600.pkl from each kept folder
          (one "initialization" / seed)
        - run decision_model_argmax on the 96 items
        - compare the rounded predicted digits to the true sum digits
          -> per-item binary error (0/1) for that initialization
        - average the binary error across all kept initializations
          -> one 96-length "model error rate" vector for this epsilon
3.  Merge the model error-rate vector with the human RT / zRT / ER columns
    (matched by the "aufgabe" string) for both Kids_II and Adults.
4.  Compute Pearson r and Spearman rho (+ p-values) between the model error
    rate and each human measure, for each population, at each epsilon.
5.  Save:
        - item-level model error rates per epsilon      -> csv
        - full correlation table (epsilon x pop x measure) -> csv
        - a "developmental profile" figure (r vs epsilon, kids vs adults)
        - scatter plots at the best-fitting epsilon
        - a ready-to-paste paper paragraph with the key numbers

Everything that depends on your cluster layout / package (paths, the
checkpoint's internal structure, `little_learner.modules...utils`) is
isolated in the CONFIG block and the `load_epsilon_models` /
`load_checkpoint_params` functions below -- these are the only places you
should need to touch if your actual file layout differs slightly from what
is assumed here.
"""

import os
import re
import sys
import glob
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONFIG -- edit these paths / names for your setup
# --------------------------------------------------------------------------

CLUSTER_DIR = os.environ.get("CLUSTER_DIR", "")  # e.g. "" if already on the cluster, or a mount prefix
STUDY_NAME = "16_STUDY-FIXED_EXP_DECAY_0.05"

MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
DECISION_BASE_DIR = (
    f"{MODULES_DIR}/LearnLikeMe/decision_module/2-digit/"
    f"{STUDY_NAME}/RI/argmax_version"
)

TARGET_OMEGA = 0.10          # Weber fraction to filter on, per config.txt
EPSILON_VALUES = [round(v, 2) for v in np.arange(0.00, 10.001, 0.50)]
CHECKPOINT_NAME = "trained_model_checkpoint_600.pkl"

UNIT_STRUCTURE_DEFAULT = (256, 128)
CARRY_STRUCTURE_DEFAULT = (16,)

KIDS_XLS = "./datasets/RT_Analysis_Kids_II_modelling.xls"
ADULTS_XLS = "./datasets/RT_Analyses_Adults_modelling.xls"
ITEM_SHEET = "Itemanalyse"

OUTPUT_DIR = "./item_level_behavioral_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import from the user's package. This mirrors the loading snippet the user
# already uses in their own training scripts.
try:
    sys.path.insert(0, CLUSTER_DIR if CLUSTER_DIR else ".")
    from little_learner.modules.decision_module.utils import (
        _make_hashable, _parse_structure, load_extractor_module,
    )
    HAVE_LITTLE_LEARNER = True
except Exception as e:  # pragma: no cover - depends on the user's cluster env
    warnings.warn(
        f"[WARN] Could not import little_learner.modules.decision_module.utils "
        f"({e}). Falling back to loading unit/carry params directly from the "
        f"checkpoint pickle if they are bundled there. If they are not "
        f"bundled, fix the import path above (sys.path) and re-run."
    )
    HAVE_LITTLE_LEARNER = False

sys.path.insert(0, "./datasets")
from little_learner.modules.extractor_modules.models import ExtractorModel  # noqa: E402


# --------------------------------------------------------------------------
# 1. Parse the 96 items from the Excel files
# --------------------------------------------------------------------------

AUFGABE_RE = re.compile(r"\s*(\d+)\s*\+\s*(\d+)\s*=?\s*")


def parse_aufgabe(aufgabe: str):
    """'12 + 7 =' -> (12, 7)"""
    m = AUFGABE_RE.match(str(aufgabe).strip())
    if not m:
        raise ValueError(f"Could not parse aufgabe string: {aufgabe!r}")
    return int(m.group(1)), int(m.group(2))


def to_two_digits(n: int):
    """37 -> (tens=3, units=7); 7 -> (tens=0, units=7)"""
    return n // 10, n % 10


def to_three_digits(n: int):
    """Zero-padded (hundreds, tens, units) representation of a sum <= 198."""
    h = n // 100
    t = (n // 10) % 10
    u = n % 10
    return h, t, u


def load_itemanalyse(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=ITEM_SHEET)
    df["aufgabe"] = df["aufgabe"].astype(str).str.strip()
    a_b = df["aufgabe"].apply(parse_aufgabe)
    df["a"] = a_b.apply(lambda t: t[0])
    df["b"] = a_b.apply(lambda t: t[1])
    df["true_sum"] = df["a"] + df["b"]
    return df


def build_model_inputs(aufgabe_list):
    """Build the (96, 4) input array [tens1, units1, tens2, units2] and the
    (96, 3) true-digit target array [hundreds, tens, units] of the sum, in
    the exact order of `aufgabe_list`."""
    x_rows, target_rows = [], []
    for aufgabe in aufgabe_list:
        a, b = parse_aufgabe(aufgabe)
        t1, u1 = to_two_digits(a)
        t2, u2 = to_two_digits(b)
        x_rows.append([t1, u1, t2, u2])
        target_rows.append(list(to_three_digits(a + b)))
    return np.array(x_rows, dtype=np.float32), np.array(target_rows, dtype=np.int32)


# --------------------------------------------------------------------------
# 2. Locating and loading the trained models for a given epsilon
# --------------------------------------------------------------------------

WEBER_RE = re.compile(r"Weber fraction \(Omega\)\s*:\s*([0-9.]+)")


def config_matches_omega(config_path: str, target_omega: float, tol: float = 1e-6) -> bool:
    try:
        text = Path(config_path).read_text(errors="ignore")
    except OSError:
        return False
    m = WEBER_RE.search(text)
    if not m:
        return False
    return abs(float(m.group(1)) - target_omega) < tol


def find_training_dirs_for_epsilon(epsilon: float, target_omega: float = TARGET_OMEGA):
    """Return the list of Training_<timestamp> directories under
    epsilon_<epsilon>/ whose config.txt reports the target Weber fraction."""
    eps_dir = os.path.join(DECISION_BASE_DIR, f"epsilon_{epsilon:.2f}")
    matches = []
    for training_dir in sorted(glob.glob(os.path.join(eps_dir, "Training_*"))):
        config_path = os.path.join(training_dir, "config.txt")
        if os.path.isfile(config_path) and config_matches_omega(config_path, target_omega):
            matches.append(training_dir)
    return matches


def load_checkpoint_params(checkpoint_path: str):
    """Load a trained_model_checkpoint_600.pkl and return
    (decision_params, unit_module, carry_module, unit_structure, carry_structure)
    where the last four are None if not bundled in the checkpoint (in which
    case they must be supplied separately, e.g. via load_extractor_module)."""
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    # Be liberal about the checkpoint's shape -- adjust here if your actual
    # pickles use different key names.
    if isinstance(ckpt, dict) and "decision_params" in ckpt:
        decision_params = ckpt["decision_params"]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        decision_params = ckpt["params"]
    elif isinstance(ckpt, dict) and all(k.startswith("dense_") for k in ckpt.keys()):
        decision_params = ckpt
    else:
        raise ValueError(
            f"Unrecognized checkpoint structure in {checkpoint_path}: "
            f"top-level keys = {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}. "
            f"Edit load_checkpoint_params() to match your actual pickle layout."
        )

    unit_module = ckpt.get("unit_module") if isinstance(ckpt, dict) else None
    carry_module = ckpt.get("carry_module") if isinstance(ckpt, dict) else None
    unit_structure = ckpt.get("unit_structure") if isinstance(ckpt, dict) else None
    carry_structure = ckpt.get("carry_structure") if isinstance(ckpt, dict) else None

    return decision_params, unit_module, carry_module, unit_structure, carry_structure


def get_extractor_modules(omega: float, unit_module, carry_module,
                           unit_structure, carry_structure):
    """Fill in unit_module/carry_module (+structures) via the user's own
    load_extractor_module() if they weren't already bundled in the checkpoint."""
    if unit_module is not None and carry_module is not None:
        u_struct = _parse_structure(unit_structure) if HAVE_LITTLE_LEARNER and unit_structure else tuple(UNIT_STRUCTURE_DEFAULT)
        c_struct = _parse_structure(carry_structure) if HAVE_LITTLE_LEARNER and carry_structure else tuple(CARRY_STRUCTURE_DEFAULT)
        return unit_module, carry_module, tuple(u_struct), tuple(c_struct)

    if not HAVE_LITTLE_LEARNER:
        raise RuntimeError(
            "unit_module/carry_module are not bundled in the checkpoint and "
            "little_learner.modules.decision_module.utils could not be "
            "imported, so they cannot be loaded. Fix the import at the top "
            "of this script (sys.path / package location)."
        )

    carry_module, _, carry_structure = load_extractor_module(
        omega, MODULES_DIR, model_type="carry_extractor", study_name=STUDY_NAME
    )
    unit_module, _, unit_structure = load_extractor_module(
        omega, MODULES_DIR, model_type="unit_extractor", study_name=STUDY_NAME
    )
    carry_structure = _make_hashable(_parse_structure(carry_structure))
    unit_structure = _make_hashable(_parse_structure(unit_structure))
    return unit_module, carry_module, unit_structure, carry_structure


# --------------------------------------------------------------------------
# 3. Forward pass (copied logic from model.py's decision_model_argmax so
#    this script has no hard dependency on the parent package for the math)
# --------------------------------------------------------------------------

def decision_model_argmax(params, x, unit_module, carry_module,
                           unit_structure, carry_structure):
    number_size = x.shape[1] // 2
    idx_i = jnp.arange(number_size)
    idx_j = jnp.arange(number_size, 2 * number_size)
    pairs = jnp.array([(i, j) for i in idx_i for j in idx_j])
    single_digit_inputs = x[:, pairs].reshape(x.shape[0], -1, 2)

    carry_outputs = jnp.stack([
        jnp.argmax(
            ExtractorModel(structure=list(carry_structure), output_dim=2)
            .apply({"params": carry_module}, single_digit_inputs[:, k]),
            axis=-1,
        )
        for k in range(single_digit_inputs.shape[1])
    ], axis=1)
    unit_outputs = jnp.stack([
        jnp.argmax(
            ExtractorModel(structure=list(unit_structure), output_dim=10)
            .apply({"params": unit_module}, single_digit_inputs[:, k]),
            axis=-1,
        )
        for k in range(single_digit_inputs.shape[1])
    ], axis=1)

    concat_features = jnp.concatenate([carry_outputs, unit_outputs], axis=1).astype(jnp.float32)
    outputs = jnp.stack([
        jnp.dot(concat_features, params[f"dense_{i}"]) for i in range(number_size + 1)
    ], axis=1)
    return outputs  # (batch, number_size + 1) raw / near-integer digit predictions


def per_item_binary_error(params, x, targets, unit_module, carry_module,
                           unit_structure, carry_structure):
    """1 if ANY predicted digit (rounded) differs from the true digit, else 0."""
    outputs = decision_model_argmax(params, x, unit_module, carry_module,
                                     unit_structure, carry_structure)
    predicted_digits = np.array(jnp.round(outputs)).astype(int)
    mismatches = (predicted_digits != targets).any(axis=1)
    return mismatches.astype(np.float64)  # (96,)


# --------------------------------------------------------------------------
# 4. Run everything for one epsilon
# --------------------------------------------------------------------------

@dataclass
class EpsilonResult:
    epsilon: float
    n_inits: int
    item_error_rate: np.ndarray  # (96,)


def run_epsilon(epsilon: float, x_all: np.ndarray, targets_all: np.ndarray) -> EpsilonResult:
    training_dirs = find_training_dirs_for_epsilon(epsilon)
    if not training_dirs:
        warnings.warn(f"[epsilon={epsilon:.2f}] No Training_* folder with "
                       f"Weber fraction {TARGET_OMEGA} found -- skipping.")
        return EpsilonResult(epsilon, 0, np.full(x_all.shape[0], np.nan))

    per_init_errors = []
    for training_dir in training_dirs:
        checkpoint_path = os.path.join(training_dir, CHECKPOINT_NAME)
        if not os.path.isfile(checkpoint_path):
            warnings.warn(f"[epsilon={epsilon:.2f}] missing {CHECKPOINT_NAME} "
                           f"in {training_dir} -- skipping this init.")
            continue
        try:
            (decision_params, unit_module, carry_module,
             unit_structure, carry_structure) = load_checkpoint_params(checkpoint_path)
            unit_module, carry_module, unit_structure, carry_structure = get_extractor_modules(
                TARGET_OMEGA, unit_module, carry_module, unit_structure, carry_structure
            )
            errs = per_item_binary_error(
                decision_params, jnp.array(x_all), targets_all,
                unit_module, carry_module, unit_structure, carry_structure,
            )
            per_init_errors.append(errs)
        except Exception as e:
            warnings.warn(f"[epsilon={epsilon:.2f}] failed on {training_dir}: {e}")

    if not per_init_errors:
        return EpsilonResult(epsilon, 0, np.full(x_all.shape[0], np.nan))

    stacked = np.stack(per_init_errors, axis=0)  # (n_inits, 96)
    mean_error_rate = stacked.mean(axis=0)
    return EpsilonResult(epsilon, stacked.shape[0], mean_error_rate)


# --------------------------------------------------------------------------
# 5. Correlations
# --------------------------------------------------------------------------

HUMAN_MEASURES = ["RT", "zRT", "ER"]


def compute_correlations(epsilon_results, aufgabe_order, kids_df, adults_df):
    kids_by_item = kids_df.set_index("aufgabe")
    adults_by_item = adults_df.set_index("aufgabe")

    rows = []
    for res in epsilon_results:
        if res.n_inits == 0:
            continue
        for pop_name, human_df in (("Kids_II", kids_by_item), ("Adults", adults_by_item)):
            human_aligned = human_df.loc[aufgabe_order]
            model_err = res.item_error_rate
            valid = ~np.isnan(model_err)
            for measure in HUMAN_MEASURES:
                y = human_aligned[measure].to_numpy()
                if valid.sum() < 3 or np.nanstd(model_err[valid]) == 0:
                    r_p, p_p, r_s, p_s = np.nan, np.nan, np.nan, np.nan
                else:
                    r_p, p_p = pearsonr(model_err[valid], y[valid])
                    r_s, p_s = spearmanr(model_err[valid], y[valid])
                rows.append({
                    "epsilon": res.epsilon,
                    "n_inits": res.n_inits,
                    "population": pop_name,
                    "measure": measure,
                    "pearson_r": r_p,
                    "pearson_p": p_p,
                    "spearman_rho": r_s,
                    "spearman_p": p_s,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. Plots
# --------------------------------------------------------------------------

def plot_developmental_profile(corr_df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, measure in zip(axes, HUMAN_MEASURES):
        sub = corr_df[corr_df["measure"] == measure]
        for pop_name, style in (("Kids_II", "o-"), ("Adults", "s--")):
            pop_sub = sub[sub["population"] == pop_name].sort_values("epsilon")
            ax.plot(pop_sub["epsilon"], pop_sub["pearson_r"], style, label=pop_name)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Epsilon (exploration rate)")
        ax.set_title(f"Model error rate vs. {measure}")
        ax.set_ylabel("Pearson r")
    axes[0].legend()
    fig.suptitle("Developmental profile: model-human correlation across training epsilon "
                 "(Weber fraction = 0.10, batch 600)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_epsilon_scatter(best_epsilon: float, epsilon_results, aufgabe_order,
                               kids_df, adults_df, out_path: str):
    res = next(r for r in epsilon_results if r.epsilon == best_epsilon)
    model_err = res.item_error_rate
    kids_by_item = kids_df.set_index("aufgabe").loc[aufgabe_order]
    adults_by_item = adults_df.set_index("aufgabe").loc[aufgabe_order]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (pop_name, human_df) in enumerate((("Kids_II", kids_by_item), ("Adults", adults_by_item))):
        for col, measure in enumerate(HUMAN_MEASURES):
            ax = axes[row, col]
            y = human_df[measure].to_numpy()
            valid = ~np.isnan(model_err)
            ax.scatter(model_err[valid], y[valid], alpha=0.6, edgecolor="k", linewidth=0.3)
            if valid.sum() >= 3 and np.std(model_err[valid]) > 0:
                r, p = pearsonr(model_err[valid], y[valid])
                z = np.polyfit(model_err[valid], y[valid], 1)
                xs = np.linspace(model_err[valid].min(), model_err[valid].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color="firebrick", linewidth=1.5)
                ax.set_title(f"{pop_name} - {measure}\nr={r:.2f}, p={p:.3g}")
            ax.set_xlabel("Model error rate")
            ax.set_ylabel(measure)
    fig.suptitle(f"Item-level behavioral validation (epsilon = {best_epsilon:.2f}, "
                 f"Weber fraction = 0.10, batch 600)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main():
    kids_df = load_itemanalyse(KIDS_XLS)
    adults_df = load_itemanalyse(ADULTS_XLS)

    aufgabe_order = kids_df["aufgabe"].tolist()
    assert set(aufgabe_order) == set(adults_df["aufgabe"]), \
        "Kids and Adults item sets do not match -- check the two Excel files."

    x_all, targets_all = build_model_inputs(aufgabe_order)
    print(f"Built model inputs for {x_all.shape[0]} items.")

    epsilon_results = []
    for epsilon in EPSILON_VALUES:
        print(f"--- epsilon = {epsilon:.2f} ---")
        res = run_epsilon(epsilon, x_all, targets_all)
        print(f"  {res.n_inits} matching initialization(s) at Weber={TARGET_OMEGA}")
        epsilon_results.append(res)

    # Save item-level model error rates
    item_rate_df = pd.DataFrame(
        {f"epsilon_{r.epsilon:.2f}": r.item_error_rate for r in epsilon_results},
        index=aufgabe_order,
    )
    item_rate_df.index.name = "aufgabe"
    item_rate_path = os.path.join(OUTPUT_DIR, "model_error_rates_by_epsilon.csv")
    item_rate_df.to_csv(item_rate_path)

    # Correlations
    corr_df = compute_correlations(epsilon_results, aufgabe_order, kids_df, adults_df)
    corr_path = os.path.join(OUTPUT_DIR, "correlation_results.csv")
    corr_df.to_csv(corr_path, index=False)

    if corr_df.empty:
        print("No correlations could be computed -- check that any models were found/loaded.")
        return

    # Plots
    profile_path = os.path.join(OUTPUT_DIR, "developmental_profile.png")
    plot_developmental_profile(corr_df, profile_path)

    # Pick the epsilon with the strongest |r| against zRT (either population)
    zrt_rows = corr_df[corr_df["measure"] == "zRT"].dropna(subset=["pearson_r"])
    best_row = zrt_rows.loc[zrt_rows["pearson_r"].abs().idxmax()]
    best_epsilon = best_row["epsilon"]
    scatter_path = os.path.join(OUTPUT_DIR, f"scatter_best_epsilon_{best_epsilon:.2f}.png")
    plot_best_epsilon_scatter(best_epsilon, epsilon_results, aufgabe_order,
                               kids_df, adults_df, scatter_path)

    # Console summary / paper paragraph
    kids_zrt = corr_df[(corr_df.epsilon == best_epsilon) & (corr_df.population == "Kids_II") & (corr_df.measure == "zRT")].iloc[0]
    adults_zrt = corr_df[(corr_df.epsilon == best_epsilon) & (corr_df.population == "Adults") & (corr_df.measure == "zRT")].iloc[0]

    print("\n================ SUMMARY ================")
    print(corr_df.sort_values(["measure", "population", "epsilon"]).to_string(index=False))

    print("\n--- Suggested paper paragraph ('Item-Level Behavioral Validation across Development') ---")
    print(
        f"To validate the model's item-level behavior against human performance, we "
        f"correlated the model's per-item error rate (averaged across {int(best_row['n_inits'])} "
        f"initializations at Weber fraction Omega = 0.10, batch 600) with the mean "
        f"standardized reaction time (zRT) of children and adults on the same 96 addition "
        f"problems. At epsilon = {best_epsilon:.2f}, the model's error rate correlated with "
        f"children's zRT at r = {kids_zrt['pearson_r']:.2f} (p = {kids_zrt['pearson_p']:.3g}) "
        f"and with adults' zRT at r = {adults_zrt['pearson_r']:.2f} "
        f"(p = {adults_zrt['pearson_p']:.3g})"
        + (", suggesting the model's item-level difficulty profile more closely "
           "resembles that of children than adults" if abs(kids_zrt['pearson_r']) > abs(adults_zrt['pearson_r'])
           else ", suggesting the model's item-level difficulty profile more closely "
                "resembles that of adults than children")
        + "."
    )

    print(f"\nSaved:\n  {item_rate_path}\n  {corr_path}\n  {profile_path}\n  {scatter_path}")


if __name__ == "__main__":
    main()
