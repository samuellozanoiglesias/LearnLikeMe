"""Training utilities for the decision module."""

import os
import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any, Union
from functools import partial

from .model import decision_model_argmax, decision_model_vector

def cross_entropy(pred, target, num_classes):
        target_onehot = jax.nn.one_hot(target.astype(int), num_classes)
        pred_log_softmax = jax.nn.log_softmax(pred)
        return -jnp.mean(jnp.sum(target_onehot * pred_log_softmax, axis=-1))

def compute_loss(params: dict, x: jnp.ndarray, y: jnp.ndarray, 
                unit_module: dict, carry_module: dict, 
                unit_structure: list=[256, 128], carry_structure: list=[16],
                model_fn=decision_model_argmax) -> float:
    y_pred = model_fn(params, x, unit_module, carry_module, unit_structure, carry_structure)
    loss = 0.0
    # If output is vector, use cross-entropy; if scalar, use MSE
    for i in range(len(y_pred)):
        y_pred[i] = jnp.array(y_pred[i])
        if y_pred[i].ndim > 1:
            loss += cross_entropy(y_pred[i], y[:, i], y_pred[i].shape[1])
        else:
            loss += jnp.mean((y_pred[i] - y[:, i]) ** 2)

    return loss

@partial(jax.jit, static_argnames=['model_fn', 'unit_structure', 'carry_structure'])
def update_params(params: dict, x: jnp.ndarray, y: jnp.ndarray, 
                 unit_module: dict, carry_module: dict, lr: float, 
                 unit_structure: list=[256, 128], carry_structure: list=[16],
                 model_fn=decision_model_argmax) -> dict:
    grads = jax.grad(compute_loss)(params, x, y, unit_module, carry_module, unit_structure, carry_structure, model_fn)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

@partial(jax.jit, static_argnames=['model_fn', 'carry_set', 'small_set', 'large_set', 'unit_structure', 'carry_structure'])
def evaluate_module(params: dict, x_test: jnp.ndarray, y_test: jnp.ndarray,
                  unit_module: dict, carry_module: dict,
                  test_pairs: List[Tuple[int, int]] = None,
                  carry_set=None, small_set=None, large_set=None,
                  unit_structure: list=[256, 128], carry_structure: list=[16],
                  model_fn=decision_model_argmax, return_predictions=None) -> Tuple[int, int, float, int, int, int, int]:
    pred = model_fn(params, x_test, unit_module, carry_module, unit_structure, carry_structure)
    loss = compute_loss(params, x_test, y_test, unit_module, carry_module, unit_structure, carry_structure, model_fn)
    # ...existing code...
    number_size = len(pred)
    pred_arr = []
    y_arr = []
    for i in range(number_size):
        y_arr.append(jnp.array(y_test[:, i]).astype(int))
        p = jnp.array(pred[i])
        if p.ndim > 1:
            p = jnp.argmax(p, axis=1)
        else:
            p = jnp.round(p).astype(int)
        pred_arr.append(p)
    pred_arr = jnp.stack(pred_arr, axis=1)
    y_arr = jnp.stack(y_arr, axis=1)
    powers = 10 ** jnp.arange(number_size - 1, -1, -1)
    predictions = jnp.sum(pred_arr * powers, axis=1)
    targets = jnp.sum(y_arr * powers, axis=1)
    pred_correct = predictions == targets
    pred_count = jnp.sum(pred_correct).astype(int)
    if test_pairs is not None:
        test_pairs_arr = jnp.array(test_pairs)
        num_inputs = x_test.shape[1]
        digits_per_number = num_inputs // 2
        a_inputs = jnp.sum(x_test[:, :digits_per_number] * (10 ** jnp.arange(digits_per_number - 1, -1, -1)), axis=1)
        b_inputs = jnp.sum(x_test[:, digits_per_number:] * (10 ** jnp.arange(digits_per_number - 1, -1, -1)), axis=1)
        inputs_stack = jnp.stack([a_inputs, b_inputs], axis=1)
        matches = jnp.any(jnp.all(inputs_stack[:, None, :] == test_pairs_arr[None, :, :], axis=-1), axis=1)
        pred_count_test = jnp.sum(pred_correct & matches).astype(int)
    else:
        pred_count_test = pred_count
    if return_predictions is not None:
        return pred_count, pred_count_test, loss.astype(float), predictions, targets
    else:
        tests = [0, 0, 0, 0]
        if carry_set is not None and small_set is not None and large_set is not None and test_pairs is not None:
            digits_per_number = x_test.shape[1] // 2
            a_arr = jnp.sum(x_test[:, :digits_per_number].astype(int) * (10 ** jnp.arange(digits_per_number - 1, -1, -1)), axis=1)
            b_arr = jnp.sum(x_test[:, digits_per_number:].astype(int) * (10 ** jnp.arange(digits_per_number - 1, -1, -1)), axis=1)
            pairs = jnp.stack([a_arr, b_arr], axis=1)  # shape (batch, 2)
            # Convert static sets to arrays for JAX-native membership tests
            test_pairs_arr = jnp.array(test_pairs)
            carry_arr = jnp.array(carry_set)
            small_arr = jnp.array(small_set)
            large_arr = jnp.array(large_set)
            # Membership masks
            def in_set(pairs, set_arr):
                return jnp.any(jnp.all(pairs[:, None, :] == set_arr[None, :, :], axis=-1), axis=1)
            test_mask = in_set(pairs, test_pairs_arr)
            carry_mask = in_set(pairs, carry_arr) & test_mask
            small_mask = in_set(pairs, small_arr) & test_mask
            large_mask = in_set(pairs, large_arr) & test_mask
            correct_mask = pred_correct & test_mask
            tests[0] = jnp.sum(correct_mask & (~carry_mask) & small_mask).astype(int)
            tests[1] = jnp.sum(correct_mask & (~carry_mask) & large_mask).astype(int)
            tests[2] = jnp.sum(correct_mask & carry_mask & small_mask).astype(int)
            tests[3] = jnp.sum(correct_mask & carry_mask & large_mask).astype(int)
        return pred_count, pred_count_test, loss.astype(float), tests

def save_trained_model(params: dict, filepath: str):
    """
    Save model parameters to a file.
    
    Args:
        params: Model parameters to save
        filepath: Path where to save the parameters
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable_params = {k: v.tolist() for k, v in params.items()}
    
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f)

def load_trained_model(filepath: str) -> dict:
    """
    Load model parameters from a file.
    
    Args:
        filepath: Path to the saved parameters
        
    Returns:
        Dictionary of model parameters
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return {k: jnp.array(v) for k, v in params.items()}

# -------------------- Parsing Utilities --------------------

def _make_hashable(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        # sort keys to keep deterministic order
        return tuple((k, _make_hashable(obj[k])) for k in sorted(obj.keys()))
    return obj

def _parse_structure(obj):
    """
    Ensure structure is a tuple of ints (or nested tuples of ints).
    Accepts:
      - already lists/tuples of ints
      - strings like '[16, 32]' or '16,32' or '16'
    """
    if isinstance(obj, str):
        s = obj.strip("[]() \t\n")
        if s == "":
            return tuple()
        parts = [p for p in s.split(",") if p != ""]
        return tuple(int(p) for p in parts)
    if isinstance(obj, list) or isinstance(obj, tuple):
        # convert elements recursively (in case nested)
        return tuple(_parse_structure(x) if isinstance(x, (list, tuple, str)) else int(x) for x in obj)
    if isinstance(obj, int):
        return (obj,)
    # fallback: return as-is but wrapped
    return (obj,)
