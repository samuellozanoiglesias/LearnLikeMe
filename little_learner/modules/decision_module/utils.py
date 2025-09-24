"""Utilities for dataset preparation and curriculum learning."""
import os
import sys
import json
import random
import pickle
from datetime import datetime
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
from typing import List, Tuple

def create_and_save_decision_params(
    filepath: str, 
    epsilon: float = 0.01,
    param_type: str = 'RI', # 'WI' for wise initialization and 'RI' for random initialization
    timestamp: str = None
) -> dict:
    """
    Initialize and save decision network parameters.
    
    Args:
        filepath: Directory where to save the parameters
        epsilon: Noise factor for parameter initialization
        param_type: Type of initialization ('WI' for wise initialization or 'RI' for random initialization)
        timestamp: Optional timestamp for the filename
        
    Returns:
        Dictionary containing decision network parameters
    """
    # Initialize base parameters
    v_params = {
        f'v_{k}_{n}_{i}_{j}': 0.0 
        for k in range(2)  # 0: carry, 1: unit
        for n in range(1, 3)  # 1: tens output, 2: units output
        for i in [0, 1]  # Position in first number
        for j in [2, 3]  # Position in second number
    }
    
    # Set initial values based on parameter type
    # For wise initialization (WI), we set specific weights to 1
    # These represent the correct dependencies for the module to work
    if param_type == 'WI':
        v_params.update({
            'v_0_1_1_3': 1.0,  # Carry from units to tens
            'v_1_1_0_2': 1.0,  # Tens digits contribution
            'v_1_2_1_3': 1.0   # Units digits contribution
        })
    
    # Apply controlled noise based on epsilon
    params = {
        key: value + random.uniform(-10, 10) * epsilon
        for key, value in v_params.items()
    }
    
    # Save parameters with timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    save_path = os.path.join(filepath, f"trainable_model_{param_type}_{epsilon}_{timestamp}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=2, default=lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x)
        
    return params


def load_initial_params(filepath: str) -> dict:
    """
    Load and process saved model parameters.
    
    Args:
        filepath: Path to the saved parameters file
        
    Returns:
        Dictionary of model parameters with appropriate JAX array conversions
    """
    with open(filepath, 'r') as f:
        loaded_params = json.load(f)
        
    def to_jnp_array(data):
        """Convert nested structures to JAX arrays where appropriate."""
        if isinstance(data, dict):
            return {key: to_jnp_array(value) for key, value in data.items()}
        elif isinstance(data, list):
            return jnp.array(data)
        elif isinstance(data, (int, float)):
            return jnp.array(data)
        return data
    
    return to_jnp_array(loaded_params)

def load_dataset(filepath: str) -> List[Tuple[int, int]]:
    """Load number pairs from a file."""
    with open(filepath, "r") as file:
        return eval(file.read())


def generate_test_dataset(test_pairs: List[Tuple[int, int]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate test dataset from number pairs."""
    x_data = []
    y_data = []
    
    for a, b in test_pairs:
        # Split numbers into tens and units
        a_dec = a // 10
        a_unit = a % 10
        b_dec = b // 10
        b_unit = b % 10

        x_data.append([a_dec, a_unit, b_dec, b_unit])

        # Calculate target outputs
        sum_units = (a_unit + b_unit) % 10
        carry_units = 1 if (a_unit + b_unit) >= 10 else 0
        sum_dec = (a_dec + b_dec + carry_units) % 10
        y_data.append([sum_dec, sum_units])
    
    return jnp.array(x_data), jnp.array(y_data)


def generate_train_dataset(train_pairs: List[Tuple[int, int]], size_epoch: int, omega: float, distribution: str,
                           seed: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate training dataset with optional curriculum learning.
    
    Args:
        train_pairs: List of training number pairs
        size_epoch: Number of samples to generate
        distribution: Whether to use curriculum learning
        
    Returns:
        Tuple (x_data, y_data) containing training inputs and target outputs
    """
    if distribution == "Decreasing_exponential":
        # Use curriculum learning with probability-based sampling
        probabilities = decreasing_exponential(train_pairs)
        selected_indices = np.random.choice(len(train_pairs), size=size_epoch, p=probabilities)
        selected_pairs = [train_pairs[i] for i in selected_indices]
    else:
        # Use balanced sampling across problem difficulties
        selected_pairs = generate_balanced_dataset(train_pairs, size_epoch)
    
    x_data = []
    y_data = []
    
    for a, b in selected_pairs:
        # Split numbers into tens and units
        a_dec = a // 10
        a_unit = a % 10
        b_dec = b // 10
        b_unit = b % 10

        x_data.append([a_dec, a_unit, b_dec, b_unit])
        

        # Calculate target outputs
        sum_units = (a_unit + b_unit) % 10
        carry_units = 1 if (a_unit + b_unit) >= 10 else 0
        sum_dec = (a_dec + b_dec + carry_units) % 10
        y_data.append([sum_dec, sum_units])
    
    # Convert to JAX array
    x_data = jnp.array(x_data)
    y_data = jnp.array(y_data)

    # Generate samples from N(mean=x_data, std=omega*|x_data|)
    rng = jrandom.PRNGKey(seed)
    std = omega * jnp.abs(x_data)
    x_data = x_data + jrandom.normal(rng, shape=x_data.shape) * std

    return x_data, y_data

def load_extractor_module(omega: float, modules_dir: str, model_type: str) -> Tuple[dict, dict]:
    """
    Load pre-trained carry and unit extractor models for a given omega value.
    
    Args:
        omega: The omega value used for training
        modules_dir: Directory containing the pre-trained models
        model_type: Type of model to load ('carry_over_extractor' or 'unit_extractor')
    """
    # Look for Training_* folders under modules_dir/model_type and read their config.txt
    model_base = os.path.join(modules_dir, model_type)
    candidates = []
    if os.path.isdir(model_base):
        for name in os.listdir(model_base):
            if not name.startswith("Training_"):
                continue
            folder = os.path.join(model_base, name)
            if not os.path.isdir(folder):
                continue
            config_path = os.path.join(folder, "config.txt")
            if not os.path.exists(config_path):
                continue
            # read config and look for Weber fraction
            try:
                with open(config_path, "r") as f:
                    for line in f:
                        if "Weber fraction" in line:
                            # try to parse a float from the line
                            try:
                                val = float(line.strip().split(":")[-1])
                            except Exception:
                                # try to extract digits with replace
                                import re
                                m = re.search(r"([0-9]*\.?[0-9]+)", line)
                                if m:
                                    val = float(m.group(1))
                                else:
                                    val = None
                            if val is not None:
                                # consider it a candidate if close to requested omega
                                if abs(val - omega) <= 1e-6:
                                    candidates.append(folder)
                            break
            except Exception:
                continue

    chosen_folder = None
    if candidates:
        # pick the most recent matching folder (by name or mtime)
        try:
            chosen_folder = max(candidates, key=lambda p: os.path.getmtime(p))
        except Exception:
            chosen_folder = sorted(candidates)[-1]

    # fallback to legacy path structure if no training folder matched
    if chosen_folder is None:
        legacy_path = os.path.join(modules_dir, f"{model_type}", f"omega_{omega:.2f}")
        if os.path.isdir(legacy_path):
            chosen_folder = legacy_path

    if chosen_folder is None:
        raise FileNotFoundError(f"No trained extractor found for model_type={model_type} and omega={omega}")

    model_path = os.path.join(chosen_folder, "trained_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"trained_model.pkl not found in {chosen_folder}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, chosen_folder

def load_decision_module(model_dir: str, checkpoint: int = None) -> Tuple[dict, dict]:
    """
    Load pre-trained decision models for a given epsilon value.
    
    Args:
        model_dir: Directory containing the trained model
        checkpoint: Checkpoint file name
    """
    if checkpoint is None:
        model_path = os.path.join(model_dir, "trained_model.pkl")
    else:
        model_path = os.path.join(model_dir, f"trained_model_checkpoint_{checkpoint}.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

def decreasing_exponential(train_pairs: List[Tuple[int, int]], N: int = 100) -> np.ndarray:
    """    
    Args:
        train_pairs: List of number pairs to calculate probabilities for
        N: Normalization factor for the exponential function
        
    Returns:
        Array of probabilities for each pair
    """
    probabilities = np.array([np.exp(-(a + b) / N) for a, b in train_pairs])
    return probabilities / probabilities.sum()


def categorize_problems(train_pairs: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], ...]:
    """
    Categorize problems into small, medium, and large based on their sum.
    
    Args:
        train_pairs: List of number pairs to categorize
        
    Returns:
        Tuple (small_problems, medium_problems, large_problems)
    """
    small = [pair for pair in train_pairs if (pair[0] + pair[1]) < 40]
    medium = [pair for pair in train_pairs if 40 <= (pair[0] + pair[1]) <= 60]
    large = [pair for pair in train_pairs if (pair[0] + pair[1]) > 60]
    
    return small, medium, large


def generate_balanced_dataset(train_pairs: List[Tuple[int, int]], 
                           size_epoch: int) -> List[Tuple[int, int]]:
    """
    Generate a balanced dataset with equal representation of problem difficulties.
    
    Args:
        train_pairs: List of all available training pairs
        size_epoch: Total number of samples to generate
        
    Returns:
        List of balanced training pairs
    """
    small, medium, large = categorize_problems(train_pairs)
    
    # Calculate balanced sample sizes
    total_classes = 3
    balanced_class_count = size_epoch // total_classes
    remaining = size_epoch - total_classes * balanced_class_count
    
    # Sample equally from each category
    balanced_small = np.random.choice(len(small), size=balanced_class_count, replace=True)
    balanced_medium = np.random.choice(len(medium), size=balanced_class_count, replace=True)
    balanced_large = np.random.choice(len(large), size=balanced_class_count, replace=True)
    
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
        for _ in range(remaining):
            category = np.random.choice(all_categories)
            idx = np.random.choice(len(category))
            remaining_pairs.append(category[idx])
    
    balanced_pairs.extend(remaining_pairs)
    np.random.shuffle(balanced_pairs)
    
    return balanced_pairs

def save_results_and_module(df_results, final_accuracy, model_params, save_dir, best=False, checkpoint_number=None):
    """
    Save the results DataFrame, accuracy, and model parameters in a timestamped folder.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model parameters
    if checkpoint_number is not None:
        model_path = os.path.join(save_dir, f"trained_model_checkpoint_{checkpoint_number}.pkl")
    
    elif best:
        model_path = os.path.join(save_dir, "trained_model_best.pkl")

    else:
        model_path = os.path.join(save_dir, "trained_model.pkl")

        # Save results DataFrame
        results_path = os.path.join(save_dir, f"training_results.csv")
        if df_results is not None:
            df_results.to_csv(results_path, index=False)

        # Save accuracy
        accuracy_path = os.path.join(save_dir, f"training_accuracy.txt")
        if final_accuracy is not None:
            with open(accuracy_path, 'w') as f:
                f.write(f"Final accuracy: {final_accuracy:.4f}\n")
    
    with open(model_path, "wb") as f:
        pickle.dump(model_params, f)

    #print(f"Results, accuracy, and model saved in '{save_dir}'")
    return save_dir

