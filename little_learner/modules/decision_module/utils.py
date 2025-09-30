"""Utilities for dataset preparation and curriculum learning."""
import os
import sys
import json
import random
import pickle
from datetime import datetime
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple

# ------------------ Parameter Initialization -------------------


def initialize_decision_params(
    filepath: str, 
    epsilon: float = 0.01,
    param_type: str = 'RI',
    model_type: str = 'argmax',
    timestamp: str = None,
    number_size: int = 2
) -> dict:
    """
    Initialize and save decision network parameters.
    
    Args:
        filepath: Directory where to save the parameters
        epsilon: Noise factor for parameter initialization
        param_type: Type of initialization ('WI' for wise initialization or 'RI' for random initialization)
        model_type: Type of model ('argmax' or 'vector')
        timestamp: Optional timestamp for the filename
        number_size: Number of digits in each number (e.g., 2 for two-digit numbers)
    Returns:
        Dictionary containing decision network parameters
    """
    if model_type.lower() == 'argmax':
        unit_dim = 1  # argmax output dimension for units
        carry_dim = 1 # argmax output dimension for carry
        fixed_value = 1 # fixed value for argmax correct params
    elif model_type.lower() == 'vector':
        unit_dim = 10 # vector output dimension for units (0-9)
        carry_dim = 2 # vector output dimension for carry (0 or 1)
        fixed_value = None # no fixed value for vector correct params
    else:
        raise ValueError("model_type must be either 'argmax' or 'vector'")

    # Define feature size (for 4 pairs, each with unit/carry, e.g. 4*unit_dim + 4*carry_dim)
    feature_size = (unit_dim + carry_dim) * number_size ** 2 # number size ** number_of_addends * (unit + carry) per pair of single-digits
    params = {}

    if param_type == 'WI':
        for i in range(number_size + 1): # +1 for possible carry in the highest place
            params[f'dense_{i}'] = np.zeros(feature_size)
        
        params = wise_initialization(params, number_size, unit_dim=unit_dim, carry_dim=carry_dim, fixed_value=fixed_value)
        params = {key: value + np.random.randn(feature_size) * epsilon for key, value in params.items()}

    elif param_type == 'RI':
        # Random initialization
        for i in range(number_size + 1): # +1 for possible carry in the highest place
            params[f'dense_{i}'] = np.random.randn(feature_size) * epsilon
    
    else:
        raise ValueError("param_type must be either 'WI' or 'RI'")
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    save_path = os.path.join(filepath, f"trainable_model_{param_type}_{epsilon}_{timestamp}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(params, f, indent=2, default=lambda x: x.tolist() if isinstance(x, (jnp.ndarray, np.ndarray)) else x)

    return params

def wise_initialization(params: dict, number_size: int, unit_dim: int, carry_dim: int, fixed_value: float) -> dict:
    # For each output digit position (0=units, 1=tens, ...), set correct weights
    for i in range(number_size + 1):
        if fixed_value is not None:
            # Set carry contribution for position i
            if i != number_size:
                for j in range(carry_dim):
                    params[f'dense_{i}'][(number_size + 1) * i * carry_dim + j] = fixed_value
            # Set unit contribution for position i > 0
            if i > 0:
                for k in range(unit_dim):
                    params[f'dense_{i}'][carry_dim * number_size ** 2 + (i - 1) * (number_size + 1) * unit_dim + k] = fixed_value
        else:
            if i != number_size:
            # Set carry contribution for position i
                for j in range(carry_dim):
                    params[f'dense_{i}'][(number_size + 1) * i * carry_dim + j] = float(j)
            # Set unit contribution for position i > 0
            if i > 0:
                for k in range(unit_dim):
                    params[f'dense_{i}'][carry_dim * number_size ** 2 + (i - 1) * (number_size + 1) * unit_dim + k] = float(k)
    return params

def OLD_create_and_save_decision_params(
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

def generate_test_dataset(pairs: List[Tuple[int, int]], number_size: int=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate test dataset from number pairs."""
    pairs = jnp.array(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a list of (a, b) tuples.")

    if number_size is None:
        max_number = int(jnp.max(pairs))
        number_size = len(str(max_number))

    a = pairs[:, 0]
    b = pairs[:, 1]
    if jnp.any(a >= 10 ** number_size) or jnp.any(b >= 10 ** number_size):
        raise ValueError(f"Some numbers in pairs exceed the specified number_size of {number_size} digits.")
    if jnp.any(a < 0) or jnp.any(b < 0):
        raise ValueError("Negative numbers are not supported.")

    # Prepare powers of ten for digit extraction
    powers = 10 ** jnp.arange(number_size)
    # Reverse powers for correct digit order
    powers = powers[::-1]

    # Extract digits for a and b
    x1 = ((a[:, None] // powers) % 10).astype(int)
    x2 = ((b[:, None] // powers) % 10).astype(int)

    # Extract digits for y (sum), with one extra digit for possible carry
    y_sum = a + b
    powers_y = 10 ** jnp.arange(number_size + 1)
    powers_y = powers_y[::-1]
    y = ((y_sum[:, None] // powers_y) % 10).astype(int)

    # Concatenate x1 and x2 for each sample
    x_data = jnp.concatenate([x1, x2], axis=1)
    y_data = y

    return x_data, y_data

def load_extractor_module(omega: float, modules_dir: str, model_type: str, study_name: str) -> Tuple[dict, dict]:
    """
    Load pre-trained carry and unit extractor models for a given omega value.
    
    Args:
        omega: The omega value used for training
        modules_dir: Directory containing the pre-trained models
        model_type: Type of model to load ('carry_extractor' or 'unit_extractor')
    """
    # Look for Training_* folders under modules_dir/model_type and read their config.txt
    model_base = os.path.join(modules_dir, model_type, study_name)
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

    if chosen_folder is None:
        raise FileNotFoundError(f"No trained extractor found for model_type={model_type} and omega={omega}")

    # Read hidden layer sizes from config.txt
    config_path = os.path.join(chosen_folder, "config.txt")
    structure = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("structure"):
                    try:
                        structure = line.split(":")[1].strip()
                    except Exception:
                        pass

    # Fallback to defaults if not found
    if model_type == "unit_extractor":
        if structure is None:
            structure = [256, 128]
    elif model_type == "carry_extractor":
        if structure is None:
            structure = [16]

    model_path = os.path.join(chosen_folder, "trained_model.pkl")
    if not os.path.exists(model_path):
        print(f"[WARN] trained_model.pkl not found in {chosen_folder}")
        raise FileNotFoundError(f"trained_model.pkl not found in {chosen_folder}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, chosen_folder, structure

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