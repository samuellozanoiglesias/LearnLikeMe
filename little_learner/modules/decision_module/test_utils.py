"""Test utilities for the decision module."""

import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

from little_learner.modules.decision_module.utils import load_dataset
from .model import decision_model_argmax, decision_model_vector

# Helper: parse the config file
def parse_config(path: str) -> dict:
    """
    Expects a plain‑text file with lines like
        Training ID: 2025-08-15_19-09-28
        EPOCHS: 100
        BATCH_SIZE: 32
        Omega Unit = 0.05
        Omega Carry = 0.03
    """
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("training id"):
                cfg["training_id"] = line.split(":")[1].strip()
            elif line.lower().startswith("epochs"):
                cfg["epochs"] = int(line.split(":")[1].strip())
            elif line.lower().startswith("batch size"):
                cfg["batch_size"] = int(line.split(":")[1].strip())
            elif line.lower().startswith("weber fraction") or line.lower().startswith("omega"):
                cfg["omega"] = float(line.split(":")[1].strip())
    return cfg

def predictions(decision_module: dict, unit_module: dict, carry_module: dict, 
                x_test: jnp.ndarray, y_test: jnp.ndarray, CODE_DIR: str, 
                unit_hidden1: int=256, unit_hidden2: int=128, unit_output_dim: int=10,
                carry_hidden1: int=16, carry_output_dim: int=2,
                model_fn=decision_model_argmax):
    """
    Evaluate model performance.
    
    Args:
        filepath : Full path where the CSV will be written.
        decision_module : Parameters of the decision module.
        unit_module : Parameters of the unit extractor.
        carry_module : Parameters of the carry detector.
        x_test : Test inputs shape (N, 4) where each row encodes two 2‑digit numbers.
        y_test : Test targets shape (N, 2) with the predicted tens / units.
        CODE_DIR : Base directory containing the datasets.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the counters, totals and derived accuracies.  The
        same DataFrame is written to `filepath` as CSV (index is omitted).
    """

    pred_tens, pred_units = model_fn(decision_module, x_test, unit_module, carry_module, unit_hidden1, unit_hidden2, unit_output_dim, carry_hidden1, carry_output_dim)

    # Convert outputs to class labels if they are vectors
    if pred_tens.ndim > 1:
        pred_tens_decoded = jnp.argmax(pred_tens, axis=1)
    else:
        pred_tens_decoded = jnp.round(pred_tens).astype(int)
    if pred_units.ndim > 1:
        pred_units_decoded = jnp.argmax(pred_units, axis=1)
    else:
        pred_units_decoded = jnp.round(pred_units).astype(int)

    pred_tens = pred_tens_decoded
    pred_units = pred_units_decoded

    # Load datasets and extractors
    DATASET_DIR = f"{CODE_DIR}/datasets"
    test_pairs = load_dataset(os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
    carry = load_dataset(os.path.join(DATASET_DIR, "carry_additions.txt"))
    small = load_dataset(os.path.join(DATASET_DIR, "small_additions.txt"))
    large = load_dataset(os.path.join(DATASET_DIR, "large_additions.txt"))

    # Helper sets for fast membership tests
    test_set = set(test_pairs)
    carry_set = set(carry)
    small_set = set(small)
    large_set = set(large)

    total_examples = x_test.shape[0]

    # Vectorized decoding of pairs
    a_arr = (x_test[:, 0].astype(int) * 10 + x_test[:, 1].astype(int)).tolist()
    b_arr = (x_test[:, 2].astype(int) * 10 + x_test[:, 3].astype(int)).tolist()
    pairs = list(zip(a_arr, b_arr))

    # Vectorized predictions and targets
    pred_arr = jnp.stack([pred_tens, pred_units], axis=1)
    true_arr = y_test.astype(int)
    correct_mask = (pred_arr == true_arr).all(axis=1)

    # Membership masks
    test_mask = np.array([pair in test_set for pair in pairs])
    carry_mask = np.array([pair in carry_set for pair in pairs])
    small_mask = np.array([pair in small_set for pair in pairs])
    large_mask = np.array([pair in large_set for pair in pairs])

    # All correct
    pred_count = int(np.sum(correct_mask))

    # Correct & in test set
    pred_count_test = int(np.sum(correct_mask & test_mask))
    # Correct & carry
    pred_count_carry = int(np.sum(correct_mask & carry_mask))
    # Correct & test & carry
    pred_count_test_carry = int(np.sum(correct_mask & test_mask & carry_mask))
    # Correct & small
    pred_count_small = int(np.sum(correct_mask & small_mask))
    # Correct & test & small
    pred_count_test_small = int(np.sum(correct_mask & test_mask & small_mask))
    # Correct & carry & small
    pred_count_carry_small = int(np.sum(correct_mask & carry_mask & small_mask))
    # Correct & test & carry & small
    pred_count_test_carry_small = int(np.sum(correct_mask & test_mask & carry_mask & small_mask))
    # Correct & large
    pred_count_large = int(np.sum(correct_mask & large_mask))
    # Correct & test & large
    pred_count_test_large = int(np.sum(correct_mask & test_mask & large_mask))
    # Correct & carry & large
    pred_count_carry_large = int(np.sum(correct_mask & carry_mask & large_mask))
    # Correct & test & carry & large
    pred_count_test_carry_large = int(np.sum(correct_mask & test_mask & carry_mask & large_mask))

    c = {
        "pred_count": pred_count,
        "pred_count_test": pred_count_test,
        "pred_count_carry": pred_count_carry,
        "pred_count_test_carry": pred_count_test_carry,
        "pred_count_small": pred_count_small,
        "pred_count_test_small": pred_count_test_small,
        "pred_count_carry_small": pred_count_carry_small,
        "pred_count_test_carry_small": pred_count_test_carry_small,
        "pred_count_large": pred_count_large,
        "pred_count_test_large": pred_count_test_large,
        "pred_count_carry_large": pred_count_carry_large,
        "pred_count_test_carry_large": pred_count_test_carry_large,
    }

    # Compute totals (used for accuracies)
    totals = {
        "total_examples": total_examples,
        "total_test": len(test_set),
        "total_carry": len(carry_set),
        "total_small": len(small_set),
        "total_large": len(large_set),
        # For the combined subsets we can use the same lengths as before
        "total_test_carry": len(test_set & carry_set),
        "total_test_small": len(test_set & small_set),
        "total_test_large": len(test_set & large_set),
        "total_carry_small": len(carry_set & small_set),
        "total_carry_large": len(carry_set & large_set),
        "total_test_carry_small": len(test_set & carry_set & small_set),
        "total_test_carry_large": len(test_set & carry_set & large_set),
    }

    # Build a single-row wide Pandas DataFrame: for each metric we create three
    # columns: <slug>_total, <slug>_count, <slug>_accuracy. This makes appending
    # results from different runs/checkpoints straightforward.
    metrics = [
        ("total", totals["total_examples"], c["pred_count"]),
        ("test_pairs", totals["total_test"], c["pred_count_test"]),
        ("carry", totals["total_carry"], c["pred_count_carry"]),
        ("test_pairs_carry", totals["total_test_carry"], c["pred_count_test_carry"]),
        ("small", totals["total_small"], c["pred_count_small"]),
        ("test_pairs_small", totals["total_test_small"], c["pred_count_test_small"]),
        ("carry_small", totals["total_carry_small"], c["pred_count_carry_small"]),
        ("test_pairs_carry_small", totals["total_test_carry_small"], c["pred_count_test_carry_small"]),
        ("large", totals["total_large"], c["pred_count_large"]),
        ("test_pairs_large", totals["total_test_large"], c["pred_count_test_large"]),
        ("carry_large", totals["total_carry_large"], c["pred_count_carry_large"]),
        ("test_pairs_carry_large", totals["total_test_carry_large"], c["pred_count_test_carry_large"]),
    ]

    row = {}
    for slug, total_val, count_val in metrics:
        # Compute accuracy as percent with 2 decimals when denominator > 0
        if total_val:
            acc = round(100 * count_val / total_val, 2)
        else:
            acc = None

        row[f"{slug}_total"] = int(total_val)
        row[f"{slug}_count"] = int(count_val)
        row[f"{slug}_accuracy"] = acc

    df = pd.DataFrame([row])
    return df
