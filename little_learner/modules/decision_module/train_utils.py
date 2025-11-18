"""Training utilities for the decision module."""

import os
import pickle
import json
import numpy as np
import optax
import jax
import jax.numpy as jnp
from jax import random as jrandom
from typing import Tuple, List, Dict, Any, Union
from functools import partial

from .model import decision_model_argmax, decision_model_vector

def debug_decision_example(params, x, y, unit_module, carry_module, unit_structure=[256,128], carry_structure=[16], model_fn=decision_model_argmax):
    """
    Debug one example: print input, output from decision_model_argmax, and real y value.
    """
    # Select first example in batch
    x_ex = x[0:1]
    y_ex = y[0:1]
    print("Input x:", x_ex)
    print("Target y:", y_ex)
    print("Parameters:", params)
    y_pred = model_fn(params, x_ex, unit_module, carry_module, unit_structure, carry_structure)
    print("Model output (raw):", y_pred)
    # If output is logits, decode to digits
    pred_digits = jnp.argmax(y_pred, axis=-1) if y_pred.ndim == 3 else jnp.round(y_pred).astype(int)
    print("Predicted digits:", pred_digits)
    # Optionally, decode to integer value
    number_size = pred_digits.shape[1] - 1
    powers = 10 ** jnp.arange(number_size, -1, -1)
    print("Powers:", powers)
    pred_number = jnp.sum(pred_digits * powers, axis=1)
    target_number = jnp.sum(y_ex[:, :(number_size + 1)] * powers, axis=1)
    print("Predicted number:", pred_number)
    print("Target number:", target_number)

def compute_loss(params: dict, x: jnp.ndarray, y: jnp.ndarray, 
                unit_module: dict, carry_module: dict, 
                unit_structure: list=[256, 128], carry_structure: list=[16],
                model_fn=decision_model_argmax) -> float:
    y_pred = model_fn(params, x, unit_module, carry_module, unit_structure, carry_structure)
    loss = jnp.mean((y_pred - y[:, :y_pred.shape[1]]) ** 2)
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
    # pred is shape (batch, number_size+1) after optimization
    number_size = pred.shape[1] - 1
    # Vectorized decoding
    pred_arr = jnp.round(pred).astype(int)
    y_arr = jnp.array(y_test[:, :(number_size + 1)]).astype(int)
    powers = 10 ** jnp.arange(number_size, -1, -1)  # shape (number_size + 1,)
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable_params = {k: v.tolist() for k, v in params.items()}
    
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f)

def load_trained_model(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        params = json.load(f)
    return {k: jnp.array(v) for k, v in params.items()}


# ------------------- TRAINING DATASET GENERATION -------------------

def generate_train_dataset(train_pairs: List[Tuple[int, int]], batch_size: int, omega: float, distribution: str,
                            alpha: float=0.05, number_size: int=None, seed: int = 0, fixed_variability: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate training dataset with optional curriculum learning.
    
    Args:
        train_pairs: List of training number pairs
        batch_size: Number of samples to generate
        distribution: Whether to use curriculum learning
        alpha: Alpha parameter for exponential decay
        number_size: Number of digits in the numbers
        seed: Random seed
        fixed_variability: Whether to use fixed standard deviation
        
    Returns:
        Tuple (x_data, y_data) containing training inputs and target outputs
    """
    # Convert train_pairs to JAX array for vectorized ops
    tp = jnp.asarray(train_pairs, dtype=int)

    if number_size is None:
        max_ab = int(jnp.max(tp)) if tp.size > 0 else 0
        number_size = len(str(max_ab)) if max_ab > 0 else 1

    rng = jrandom.PRNGKey(seed)
    if distribution.lower() == "decreasing_exponential":
        probabilities = jnp.array([jnp.exp(-alpha * (a + b)) for a, b in train_pairs])
        probabilities = probabilities / jnp.sum(probabilities)
        indices = jrandom.choice(rng, len(train_pairs), shape=(batch_size,), p=probabilities)
        selected_pairs = jnp.array([train_pairs[i] for i in indices])
    elif distribution.lower() == "balanced":
        # Balanced sampling
        bal = generate_balanced_dataset(train_pairs, batch_size, number_size, rng)
        selected_pairs = jnp.asarray(bal, dtype=int)
    else:
        # Uniform random sampling
        selected_indices = jrandom.choice(rng, len(tp), shape=(batch_size,), replace=True)
        selected_pairs = tp[selected_indices]

    # Selected a and b arrays
    a_arr = selected_pairs[:, 0].astype(int)
    b_arr = selected_pairs[:, 1].astype(int)

    # Vectorized digit extraction: produce digits msb->lsb
    pow_a = 10 ** jnp.arange(number_size - 1, -1, -1, dtype=int)
    a_digits = ((a_arr[:, None] // pow_a[None, :]) % 10).astype(int)
    b_digits = ((b_arr[:, None] // pow_a[None, :]) % 10).astype(int)

    # Sum digits for target (number_size + 1 digits, msb->lsb)
    sum_arr = a_arr + b_arr
    pow_y = 10 ** jnp.arange(number_size, -1, -1, dtype=int)
    y_digits = ((sum_arr[:, None] // pow_y[None, :]) % 10).astype(int)

    # Concatenate inputs
    x_data = jnp.concatenate([a_digits, b_digits], axis=1)
    y_data = y_digits

    # Add noise (vectorized) using JAX RNG
    if omega and omega != 0.0:
        rng, noise_key = jrandom.split(rng)
        if fixed_variability:
            std = omega  # Fixed standard deviation
        else:
            std = omega * jnp.abs(x_data.astype(float))  # Variable standard deviation based on magnitude
        noise = jrandom.normal(noise_key, shape=x_data.shape) * std
        x_data = x_data + noise

    return x_data, y_data

def categorize_problems(train_pairs: List[Tuple[int, int]], number_size: int) -> Tuple[List[Tuple[int, int]], ...]:
    """
    Categorize problems into small, medium, and large based on their sum.
    
    Args:
        train_pairs: List of number pairs to categorize
        
    Returns:
        Tuple (small_problems, medium_problems, large_problems)
    """
    small = [pair for pair in train_pairs if (pair[0] + pair[1]) < 4 * 10 ** (number_size - 1)]
    medium = [pair for pair in train_pairs if 4 * 10 ** (number_size - 1) <= (pair[0] + pair[1]) <= 6 * 10 ** (number_size - 1)]
    large = [pair for pair in train_pairs if (pair[0] + pair[1]) > 6 * 10 ** (number_size - 1)]

    return small, medium, large


def generate_balanced_dataset(train_pairs: List[Tuple[int, int]], 
                           batch_size: int, number_size: int, rng: jrandom.PRNGKey) -> List[Tuple[int, int]]:
    """
    Generate a balanced dataset with equal representation of problem difficulties.
    
    Args:
        train_pairs: List of all available training pairs
        batch_size: Total number of samples to generate
        rng: JAX random key for sampling
        
    Returns:
        List of balanced training pairs
    """
    small, medium, large = categorize_problems(train_pairs, number_size)
    
    # Calculate balanced sample sizes
    total_classes = 3
    balanced_class_count = batch_size // total_classes
    remaining = batch_size - total_classes * balanced_class_count
    
    # Split RNG for different sampling operations
    rng1, rng2, rng3, rng4 = jrandom.split(rng, 4)
    
    # Sample equally from each category
    balanced_small = jrandom.choice(rng1, len(small), shape=(balanced_class_count,), replace=True)
    balanced_medium = jrandom.choice(rng2, len(medium), shape=(balanced_class_count,), replace=True)
    balanced_large = jrandom.choice(rng3, len(large), shape=(balanced_class_count,), replace=True)
    
    # Convert indices to pairs
    balanced_pairs = (
        [small[i] for i in balanced_small] +
        [medium[i] for i in balanced_medium] +
        [large[i] for i in balanced_large]
    )
    
    # Handle remaining samples
    remaining_pairs = []
    if remaining > 0:
        all_categories = [small, medium, large]
        category_indices = jrandom.choice(rng4, len(all_categories), shape=(remaining,), replace=True)
        for cat_idx in category_indices:
            category = all_categories[cat_idx]
            rng4, subkey = jrandom.split(rng4)
            idx = jrandom.choice(subkey, len(category))
            remaining_pairs.append(category[idx])
    
    balanced_pairs.extend(remaining_pairs)
    # Shuffle using JAX
    all_pairs = jnp.array(balanced_pairs)
    rng4, shuffle_key = jrandom.split(rng4)
    shuffled_indices = jrandom.permutation(shuffle_key, len(all_pairs))
    shuffled_pairs = all_pairs[shuffled_indices]
    
    return shuffled_pairs.tolist()



# --- Legacy: OPTIMIZER SETUP ---
def create_optimizer(lr=1e-3, grad_clip=1.0):
    """Create optax optimizer (Adam + gradient clipping)."""
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr)
    )

def init_optimizer_state(params, lr=1e-3, grad_clip=1.0):
    optimizer = create_optimizer(lr, grad_clip)
    return optimizer, optimizer.init(params)

@partial(jax.jit, static_argnames=['model_fn', 'unit_structure', 'carry_structure', 'optimizer'])
def optimizer_update_params(params, opt_state, x, y, unit_module, carry_module,
                 unit_structure=[256, 128], carry_structure=[16],
                 model_fn=decision_model_argmax, optimizer=None):
    """
    Update parameters using optax optimizer.
    Returns: new_params, new_opt_state, loss
    """
    def loss_fn(p):
        return compute_loss(p, x, y, unit_module, carry_module, unit_structure, carry_structure, model_fn)
    grads = jax.grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state