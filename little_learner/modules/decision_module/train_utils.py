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
                unit_model: dict, carry_model: dict) -> float:
    """
    Compute MSE loss for the decision module.
    
    Args:
        params: Decision module parameters
        x: Input data
        y: Target outputs
        unit_model: Pre-trained unit extractor parameters
        carry_model: Pre-trained carry detector parameters
        
    Returns:
        Mean squared error loss
    """
    y_pred_1, y_pred_2 = decision_model(params, x, unit_model, carry_model)
    return jnp.mean((y_pred_1 - y[:, 0]) ** 2) + jnp.mean((y_pred_2 - y[:, 1]) ** 2)


@jax.jit
def update_params(params: dict, x: jnp.ndarray, y: jnp.ndarray, 
                 unit_model: dict, carry_model: dict, lr: float) -> dict:
    """
    Update model parameters using gradient descent.
    
    Args:
        params: Current model parameters
        x: Input data
        y: Target outputs
        unit_model: Pre-trained unit extractor parameters
        carry_model: Pre-trained carry detector parameters
        lr: Learning rate
        
    Returns:
        Updated parameters
    """
    grads = jax.grad(compute_loss)(params, x, y, unit_model, carry_model)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def evaluate_model(params: dict, x_test: jnp.ndarray, y_test: jnp.ndarray,
                  unit_model: dict, carry_model: dict, 
                  test_pairs: List[Tuple[int, int]] = None,
                  return_predictions: bool = False) -> Union[Tuple[int, int, float], 
                                                          Tuple[int, int, float, jnp.ndarray]]:
    """
    Evaluate model performance.
    
    Args:
        params: Model parameters
        x_test: Test inputs
        y_test: Test targets
        unit_model: Pre-trained unit extractor parameters
        carry_model: Pre-trained carry detector parameters
        test_pairs: Optional list of test pairs for specific accuracy calculation
        return_predictions: If True, also return array of predictions
        
    Returns:
        If return_predictions is False:
            Tuple (total correct predictions, test set correct predictions, loss)
        If return_predictions is True:
            Tuple (total correct predictions, test set correct predictions, loss, predictions)
    """
    pred_tens, pred_units = decision_model(params, x_test, unit_model, carry_model)
    loss = compute_loss(params, x_test, y_test, unit_model, carry_model)
    
    pred_count = 0
    pred_count_test = 0
    predictions = jnp.zeros(len(x_test))
    
    for i in range(len(x_test)):
        normalized_pred = [
            int(jnp.round(pred_tens[i].item())),
            int(jnp.round(pred_units[i].item()))
        ]
        prediction = normalized_pred[0] * 10 + normalized_pred[1]
        predictions = predictions.at[i].set(prediction)
        
        # Get target value
        target = y_test[i, 0] * 10 + y_test[i, 1]
        
        if prediction == target:
            pred_count += 1
            if test_pairs is not None and i < len(test_pairs):
                pred_count_test += 1
    
    if return_predictions:
        return pred_count, pred_count_test, loss, predictions
    return pred_count, pred_count_test, loss


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
