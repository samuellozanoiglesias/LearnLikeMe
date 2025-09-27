import os
import json
import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple

# -------------------- Data Utilities --------------------
def save_results_and_module(df_results, final_accuracy, model_params, save_dir, checkpoint_number=None):
    """
    Save the results DataFrame, accuracy, and model parameters in a timestamped folder.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save results DataFrame
    results_path = os.path.join(save_dir, f"training_results.csv")
    if df_results is not None:
        df_results.to_csv(results_path, index=False)

    # Save accuracy
    accuracy_path = os.path.join(save_dir, f"training_accuracy.txt")
    if final_accuracy is not None:
        with open(accuracy_path, 'w') as f:
            f.write(f"Final accuracy: {final_accuracy:.4f}\n")

    # Save model parameters
    if checkpoint_number is not None:
        model_path = os.path.join(save_dir, f"trained_model_checkpoint_{checkpoint_number}.pkl")
    else:
        model_path = os.path.join(save_dir, "trained_model.pkl")
        print(f"Results, accuracy, and model saved in '{save_dir}'")
    
    with open(model_path, "wb") as f:
        pickle.dump(model_params, f)

    return save_dir

# -------------------- Parameter Utilities --------------------
def create_and_save_initial_params(model, rng, input_shape, file_path):
    params = model.init(rng, jnp.ones(input_shape))["params"]

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (jnp.ndarray, jax.Array)):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    with open(file_path, 'w') as f:
        json.dump(to_serializable(params), f)

    return params

def load_initial_params(file_path):
    with open(file_path, 'r') as f:
        loaded_params = json.load(f)
    def to_jnp_array(data):
        if isinstance(data, dict):
            return {key: to_jnp_array(value) for key, value in data.items()}
        elif isinstance(data, list):
            return jnp.array(data)
        else:
            return data
    return to_jnp_array(loaded_params)

# -------------------- Data Generation --------------------
# Weber fraction applied when generating data through gaussian noise
def generate_train_data(omega=0.0, seed=0, module_name=None):
    x_data = jnp.array([[a, b] for a in range(10) for b in range(10)], dtype=jnp.float32)
    if module_name == "carry_extractor":
        y_data = jnp.array([1 if (a + b) >= 10 else 0 for a in range(10) for b in range(10)], dtype=jnp.int32)
    elif module_name == "unit_extractor":
        y_data = jnp.array([(a + b) % 10 for a in range(10) for b in range(10)], dtype=jnp.int32)
    else:
        raise ValueError("module_name must be 'carry_extractor' or 'unit_extractor'")
    
    rng = random.PRNGKey(seed)
    # Generate samples from N(mean=x_data, std=omega*|x_data|)
    std = omega #* jnp.abs(x_data)
    x_data = x_data + random.normal(rng, shape=x_data.shape) * std
    return x_data, y_data

def load_dataset(filepath: str) -> List[Tuple[int, int]]:
    """Load number pairs from a file."""
    with open(filepath, "r") as file:
        return eval(file.read())

def generate_test_dataset(test_pairs: List[Tuple[int, int]], module_name: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate test dataset from number pairs."""
    x_data = []
    y_data = []
    
    for a, b in test_pairs:
        x_data.append([a, b])
    
        # Calculate target outputs
        sum_units = (a + b) % 10
        carry_units = 1 if (a + b) >= 10 else 0
        if module_name == "carry_extractor":
            y_data.append(carry_units)
        elif module_name == "unit_extractor":
            y_data.append(sum_units)   
         
    return jnp.array(x_data), jnp.array(y_data)

# -------------------- One-hot Encoding --------------------
def one_hot_encode(y, num_classes=None):
    """
    Devuelve un jnp.array denso (float32) de shape (N, num_classes).
    """
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y_reshaped = jnp.asarray(y).reshape(-1, 1) 
    y_np = jnp.asarray(y_reshaped).astype('int32')
    y_one_hot = encoder.fit_transform(y_np)
    return jnp.array(y_one_hot, dtype=jnp.float32)
