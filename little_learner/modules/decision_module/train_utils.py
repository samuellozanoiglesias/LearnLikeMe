"""Training utilities for the decision module."""

import os
import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any, Union

from .model import decision_model

def compute_loss(params: dict, x: jnp.ndarray, y: jnp.ndarray, 
                unit_module: dict, carry_module: dict) -> float:
    """
    Compute MSE loss for the decision module.
    
    Args:
        params: Decision module parameters
        x: Input data
        y: Target outputs
        unit_module: Pre-trained unit extractor parameters
        carry_module: Pre-trained carry detector parameters
        
    Returns:
        Mean squared error loss
    """
    y_pred_1, y_pred_2 = decision_model(params, x, unit_module, carry_module)
    return jnp.mean((y_pred_1 - y[:, 0]) ** 2) + jnp.mean((y_pred_2 - y[:, 1]) ** 2)


@jax.jit
def update_params(params: dict, x: jnp.ndarray, y: jnp.ndarray, 
                 unit_module: dict, carry_module: dict, lr: float) -> dict:
    """
    Update model parameters using gradient descent.
    
    Args:
        params: Current model parameters
        x: Input data
        y: Target outputs
        unit_module: Pre-trained unit extractor parameters
        carry_module: Pre-trained carry detector parameters
        lr: Learning rate
        
    Returns:
        Updated parameters
    """
    grads = jax.grad(compute_loss)(params, x, y, unit_module, carry_module)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def evaluate_module(params: dict, x_test: jnp.ndarray, y_test: jnp.ndarray,
                  unit_module: dict, carry_module: dict, 
                  test_pairs: List[Tuple[int, int]] = None,
                  return_predictions: bool = False) -> Union[Tuple[int, int, float], 
                                                          Tuple[int, int, float, jnp.ndarray]]:
    """
    Evaluate model performance.
    
    Args:
        params: Model parameters
        x_test: Test inputs
        y_test: Test targets
        unit_module: Pre-trained unit extractor parameters
        carry_module: Pre-trained carry detector parameters
        test_pairs: Optional list of test pairs for specific accuracy calculation
        return_predictions: If True, also return array of predictions
        
    Returns:
        If return_predictions is False:
            Tuple (total correct predictions, test set correct predictions, loss)
        If return_predictions is True:
            Tuple (total correct predictions, test set correct predictions, loss, predictions)
    """
    pred_tens, pred_units = decision_model(params, x_test, unit_module, carry_module)
    loss = compute_loss(params, x_test, y_test, unit_module, carry_module)
    
    # Reconstruct predictions and targets as integers
    predictions = jnp.round(pred_tens) * 10 + jnp.round(pred_units)
    predictions = predictions.astype(int)
    targets = y_test[:,0] * 10 + y_test[:,1]

    # Total correct predictions
    pred_correct = predictions == targets
    pred_count = int(jnp.sum(pred_correct))
    
    if test_pairs is not None:
        test_pairs_arr = jnp.array(test_pairs)  # shape (n_pairs, 2)
        a_inputs = x_test[:,0] * 10 + x_test[:,1]
        b_inputs = x_test[:,2] * 10 + x_test[:,3]
        inputs_stack = jnp.stack([a_inputs, b_inputs], axis=1)
        
        # Vectorized comparison
        matches = jnp.any(jnp.all(inputs_stack[:, None, :] == test_pairs_arr[None, :, :], axis=-1), axis=1)
        pred_count_test = int(jnp.sum(pred_correct & matches))
    else:
        pred_count_test = pred_count
    
    if return_predictions:
        return pred_count, pred_count_test, float(loss), predictions
    else:
        return pred_count, pred_count_test, float(loss)

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
