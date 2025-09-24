"""Test utilities for the decision module."""

import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

from little_learner.modules.decision_module.utils import load_dataset
from .model import decision_model

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
                x_test: jnp.ndarray, y_test: jnp.ndarray, CODE_DIR: str):
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

    pred_tens, pred_units = decision_model(decision_module, x_test, unit_module, carry_module)

    # Load datasets and extractors
    DATASET_DIR = f"{CODE_DIR}/datasets"
    test_pairs = load_dataset(os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
    carry = load_dataset(os.path.join(DATASET_DIR, "with_carry_additions.txt"))
    small = load_dataset(os.path.join(DATASET_DIR, "small_additions.txt"))
    large = load_dataset(os.path.join(DATASET_DIR, "large_additions.txt"))

    # Helper sets for fast membership tests
    test_set = set(test_pairs)
    carry_set = set(carry)
    small_set = set(small)
    large_set = set(large)

    c = {
        "pred_count": 0,
        "pred_count_test": 0,
        "pred_count_carry": 0,
        "pred_count_test_carry": 0,
        "pred_count_small": 0,
        "pred_count_test_small": 0,
        "pred_count_carry_small": 0,
        "pred_count_test_carry_small": 0,
        "pred_count_large": 0,
        "pred_count_test_large": 0,
        "pred_count_carry_large": 0,
        "pred_count_test_carry_large": 0,
    }
    
    total_examples = x_test.shape[0]
        
    for i in range(total_examples):
        # Round the predictions to integers
        pred = [int(jnp.round(pred_tens[i].item())),
                int(jnp.round(pred_units[i].item()))]

        # Decode the two 2‑digit numbers from the input tensor
        a = int(str(int(x_test[i, 0])) + str(int(x_test[i, 1])))
        b = int(str(int(x_test[i, 2])) + str(int(x_test[i, 3])))
        pair = (a, b)

        if pred == [int(y_test[i, 0]), int(y_test[i, 1])]:
            # 4.1  Global correct
            c["pred_count"] += 1

            # 4.2  In the test‑pairs list
            if pair in test_set:
                c["pred_count_test"] += 1

                # 4.2.1  Carry‑present among the test pairs
                if pair in carry_set:
                    c["pred_count_test_carry"] += 1
                    if pair in small_set:
                        c["pred_count_test_carry_small"] += 1
                    elif pair in large_set:
                        c["pred_count_test_carry_large"] += 1

                # 4.2.2  Small / large classification
                if pair in small_set:
                    c["pred_count_test_small"] += 1
                elif pair in large_set:
                    c["pred_count_test_large"] += 1

            # 4.3  Carry‑present (overall)
            if pair in carry_set:
                c["pred_count_carry"] += 1
                if pair in small_set:
                    c["pred_count_carry_small"] += 1
                elif pair in large_set:
                    c["pred_count_carry_large"] += 1

            # 4.4  Small / large classification (overall)
            if pair in small_set:
                c["pred_count_small"] += 1
            elif pair in large_set:
                c["pred_count_large"] += 1

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
    # columns: <slug>_count, <slug>_total, <slug>_accuracy. This makes appending
    # results from different runs/checkpoints straightforward.
    metrics = [
        ("total", c["pred_count"], totals["total_examples"]),
        ("test_pairs", c["pred_count_test"], totals["total_test"]),
        ("carry", c["pred_count_carry"], totals["total_carry"]),
        ("test_pairs_carry", c["pred_count_test_carry"], totals["total_test_carry"]),
        ("small", c["pred_count_small"], totals["total_small"]),
        ("test_pairs_small", c["pred_count_test_small"], totals["total_test_small"]),
        ("carry_small", c["pred_count_carry_small"], totals["total_carry_small"]),
        ("test_pairs_carry_small", c["pred_count_test_carry_small"], totals["total_test_carry_small"]),
        ("large", c["pred_count_large"], totals["total_large"]),
        ("test_pairs_large", c["pred_count_test_large"], totals["total_test_large"]),
        ("carry_large", c["pred_count_carry_large"], totals["total_carry_large"]),
        ("test_pairs_carry_large", c["pred_count_test_carry_large"], totals["total_test_carry_large"]),
    ]

    row = {}
    for slug, count_val, total_val in metrics:
        # Compute accuracy as percent with 2 decimals when denominator > 0
        if total_val:
            acc = round(100 * count_val / total_val, 2)
        else:
            acc = None

        row[f"{slug}_count"] = int(count_val)
        row[f"{slug}_total"] = int(total_val)
        row[f"{slug}_accuracy"] = acc

    df = pd.DataFrame([row])
    return df
