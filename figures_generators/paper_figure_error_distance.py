# USE: nohup python paper_figure_error_distance.py 2 STUDY [RI|WI] argmax [0.15|none] CHECKPOINT > logs_paper_error_distance.out 2>&1 &
# If omega is specified (e.g., 0.15), generates figure for that specific omega value
# If omega is 'none' or omitted, generates figure aggregated over all omegas
# Always aggregates over all epsilons
# CHECKPOINT: specific checkpoint/batch number to load model from

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import pickle
from pathlib import Path

# Add the parent directory to Python path to find little_learner module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Set JAX to CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
from jax import jit

from little_learner.modules.decision_module.utils import (
    load_extractor_module, _make_hashable, _parse_structure
)
from little_learner.modules.decision_module.model import decision_model_argmax, decision_model_vector

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def find_latest_checkpoint(training_dir):
    """Find the latest checkpoint number in a training directory."""
    max_checkpoint = -1
    try:
        for filename in os.listdir(training_dir):
            if filename.startswith('trained_model_checkpoint_') and filename.endswith('.pkl'):
                # Extract checkpoint number from filename
                checkpoint_str = filename.replace('trained_model_checkpoint_', '').replace('.pkl', '')
                try:
                    checkpoint_num = int(checkpoint_str)
                    max_checkpoint = max(max_checkpoint, checkpoint_num)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error scanning directory {training_dir}: {e}")
    
    return max_checkpoint if max_checkpoint >= 0 else None

def load_model_checkpoint(checkpoint_path):
    """Load model checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        print(f"Could not load checkpoint from {checkpoint_path}: {e}")
        return None

def load_test_set(number_size):
    """Load test set for the given number size."""
    # Try local path first (relative to script location)
    test_file = f"datasets/{number_size}-digit/stimuli_test_pairs.txt"
    
    if not os.path.exists(test_file):
        # Try cluster path
        test_file = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/datasets/{number_size}-digit/stimuli_test_pairs.txt"
    
    if not os.path.exists(test_file):
        # Try relative to current working directory
        test_file = f"/home/samuloza/LearnLikeMe/datasets/{number_size}-digit/stimuli_test_pairs.txt"
    
    test_pairs = []
    if os.path.exists(test_file):
        try:
            with open(test_file, 'r') as f:
                content = f.read().strip()
                # Parse the Python list representation
                pairs_list = eval(content)
                
                # Convert to format expected by the model: (num1, num2, result)
                for num1, num2 in pairs_list:
                    result = num1 + num2  # Calculate the correct result
                    test_pairs.append((num1, num2, result))
                    
            print(f"Loaded {len(test_pairs)} test pairs from: {test_file}")
        except Exception as e:
            print(f"Error parsing test file {test_file}: {e}")
            return []
    else:
        print(f"Could not find test file at any of the attempted paths")
    
    return test_pairs

def calculate_error_distance(predicted, actual):
    """Calculate the absolute difference between predicted and actual values."""
    return abs(predicted - actual)

def prepare_batch_inputs(test_pairs, number_size):
    """Pre-convert all test pairs to batched input format for efficiency."""
    num_pairs = len(test_pairs)
    
    if number_size == 2:
        batch_inputs = np.zeros((num_pairs, 4), dtype=np.int32)
        for i, (num1, num2, _) in enumerate(test_pairs):
            tens1, units1 = divmod(num1, 10)
            tens2, units2 = divmod(num2, 10)
            batch_inputs[i] = [tens1, units1, tens2, units2]
    elif number_size == 3:
        batch_inputs = np.zeros((num_pairs, 6), dtype=np.int32)
        for i, (num1, num2, _) in enumerate(test_pairs):
            hundreds1, remainder1 = divmod(num1, 100)
            tens1, units1 = divmod(remainder1, 10)
            hundreds2, remainder2 = divmod(num2, 100)
            tens2, units2 = divmod(remainder2, 10)
            batch_inputs[i] = [hundreds1, tens1, units1, hundreds2, tens2, units2]
    else:
        batch_inputs = np.zeros((num_pairs, number_size * 2), dtype=np.int32)
        for i, (num1, num2, _) in enumerate(test_pairs):
            digits1 = [int(d) for d in str(num1).zfill(number_size)]
            digits2 = [int(d) for d in str(num2).zfill(number_size)]
            batch_inputs[i] = digits1 + digits2
    
    return jnp.array(batch_inputs)

def test_model_and_get_errors(params, test_pairs, unit_module, carry_module, model_fn, unit_structure, carry_structure, number_size):
    """Test model on test set and return error distances for incorrect predictions (batched version)."""
    error_distances = []
    
    # Prepare all inputs at once
    batch_inputs = prepare_batch_inputs(test_pairs, number_size)
    correct_results = np.array([result for _, _, result in test_pairs])
    
    # Process in batches to avoid memory issues
    batch_size = 256
    num_batches = (len(test_pairs) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_pairs))
        
        batch_x = batch_inputs[start_idx:end_idx]
        batch_correct = correct_results[start_idx:end_idx]
        
        try:
            # Get predictions for entire batch
            predictions = model_fn(params, batch_x, unit_module, carry_module, 
                                 unit_structure=unit_structure, 
                                 carry_structure=carry_structure)
            
            # Convert predictions to numbers
            if isinstance(predictions, jnp.ndarray) and len(predictions.shape) == 2:
                # predictions has shape (batch_size, number_size+1)
                for i in range(predictions.shape[0]):
                    predicted_digits = predictions[i]
                    
                    # Convert digit predictions back to a number
                    prediction_val = 0
                    for j, digit_pred in enumerate(predicted_digits):
                        if hasattr(digit_pred, 'item'):
                            digit = int(round(digit_pred.item()))
                        else:
                            digit = int(round(float(digit_pred)))
                        # Most significant digit first
                        prediction_val += digit * (10 ** (len(predicted_digits) - 1 - j))
                    
                    # Only record errors (incorrect predictions)
                    if prediction_val != batch_correct[i]:
                        distance = calculate_error_distance(prediction_val, batch_correct[i])
                        error_distances.append(distance)
            else:
                print(f"Unexpected prediction format: {type(predictions)}, shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    return error_distances

def test_model_and_get_errors_legacy(params, test_pairs, unit_module, carry_module, model_fn, unit_structure, carry_structure):
    """LEGACY: Test model on test set (one-by-one - slower but fallback)."""
    error_distances = []
    
    for num1, num2, correct_result in test_pairs:
        try:
            # Convert numbers to individual digits for 2-digit addition
            # num1=41 -> [4, 1], num2=35 -> [3, 5] -> input=[4, 1, 3, 5]
            if NUMBER_SIZE == 2:
                tens1, units1 = divmod(num1, 10)
                tens2, units2 = divmod(num2, 10)
                x = jnp.array([[tens1, units1, tens2, units2]])
            elif NUMBER_SIZE == 3:
                hundreds1, remainder1 = divmod(num1, 100)
                tens1, units1 = divmod(remainder1, 10)
                hundreds2, remainder2 = divmod(num2, 100)
                tens2, units2 = divmod(remainder2, 10)
                x = jnp.array([[hundreds1, tens1, units1, hundreds2, tens2, units2]])
            else:
                # For other number sizes, convert to digit representation
                digits1 = [int(d) for d in str(num1).zfill(NUMBER_SIZE)]
                digits2 = [int(d) for d in str(num2).zfill(NUMBER_SIZE)]
                x = jnp.array([digits1 + digits2])
            
            # Get prediction using the model function
            prediction = model_fn(params, x, unit_module, carry_module, 
                                unit_structure=unit_structure, 
                                carry_structure=carry_structure)
            
            # Handle prediction format - model returns (batch, number_size+1) digits
            if isinstance(prediction, jnp.ndarray) and len(prediction.shape) == 2:
                # prediction has shape (1, number_size+1) - convert digits back to number
                predicted_digits = prediction[0]  # Remove batch dimension
                
                # Convert digit predictions back to a number
                prediction_val = 0
                for i, digit_pred in enumerate(predicted_digits):
                    if hasattr(digit_pred, 'item'):
                        digit = int(round(digit_pred.item()))
                    else:
                        digit = int(round(float(digit_pred)))
                    # Most significant digit first
                    prediction_val += digit * (10 ** (len(predicted_digits) - 1 - i))
                    
            else:
                print(f"Unexpected prediction format: {type(prediction)}, {prediction}")
                continue
            
            # Only record errors (incorrect predictions)
            if prediction_val != correct_result:
                distance = calculate_error_distance(prediction_val, correct_result)
                error_distances.append(distance)
                
        except Exception as e:
            print(f"Error testing model on {num1} + {num2} = {correct_result}: {e}")
            print(f"  Prediction type: {type(prediction) if 'prediction' in locals() else 'undefined'}")
            if 'prediction' in locals():
                print(f"  Prediction value: {prediction}")
            continue
    
    return error_distances

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].lower() != 'none' else None  # Specific omega value to analyze, or None for all omegas
CHECKPOINT = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].lower() != 'none' else None  # Checkpoint/batch number to load, or None for latest

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"
RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"

def analyze_multidigit_module(raw_dir, figures_dir, omega_value, param_type, checkpoint_num):
    os.makedirs(figures_dir, exist_ok=True)
    
    use_latest_checkpoint = checkpoint_num is None
    if use_latest_checkpoint:
        print("Using latest checkpoint from each training directory")
    
    # Load test set
    test_pairs = load_test_set(NUMBER_SIZE)
    if not test_pairs:
        print(f"Could not load test set for {NUMBER_SIZE}-digit numbers")
        return
    
    print(f"Loaded {len(test_pairs)} test pairs")
    
    # Load extractor modules (needed for decision model)
    try:
        carry_module, carry_dir, carry_structure = load_extractor_module(omega_value if omega_value is not None else 0.15, 
                                                                        f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe", 
                                                                        model_type='carry_extractor', 
                                                                        study_name=STUDY_NAME)
        unit_module, unit_dir, unit_structure = load_extractor_module(omega_value if omega_value is not None else 0.15, 
                                                                     f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe", 
                                                                     model_type='unit_extractor', 
                                                                     study_name=STUDY_NAME)
        
        carry_structure = _parse_structure(carry_structure)
        unit_structure = _parse_structure(unit_structure)
        carry_structure = _make_hashable(carry_structure)
        unit_structure = _make_hashable(unit_structure)
        
        print(f"Loaded extractor modules from {carry_dir} and {unit_dir}")
        
    except Exception as e:
        print(f"Error loading extractor modules: {e}")
        return
    
    # Select model function
    if MODEL_TYPE == "vector":
        model_fn = decision_model_vector
    elif MODEL_TYPE == "argmax":
        model_fn = decision_model_argmax
    else:
        print(f"Unknown model type: {MODEL_TYPE}")
        return
    
    # Collect error distances from all models
    all_error_distances = []
    models_tested = 0
    
    # Scan all epsilon directories and read config files to find matching omega values
    filter_desc = f"omega={omega_value if omega_value is not None else 'all'}, param={param_type}, checkpoint={checkpoint_num}"
    
    # Find all epsilon directories
    epsilon_dirs = []
    try:
        for item in os.listdir(raw_dir):
            item_path = os.path.join(raw_dir, item)
            if os.path.isdir(item_path) and item.startswith("epsilon_"):
                try:
                    epsilon_str = item.replace("epsilon_", "")
                    epsilon_val = float(epsilon_str)
                    epsilon_dirs.append((epsilon_val, item_path))
                except ValueError:
                    print(f"Could not parse epsilon value from folder: {item}")
        
        epsilon_dirs.sort()  # Sort by epsilon value
        print(f"Found {len(epsilon_dirs)} epsilon directories")
        
    except Exception as e:
        print(f"Error scanning raw directory {raw_dir}: {e}")
        return
    
    if not epsilon_dirs:
        print(f"No epsilon folders found in {raw_dir}")
        return
    
    # Process each epsilon directory
    print(f"\nProcessing {len(epsilon_dirs)} epsilon directories...")
    for epsilon_idx, (epsilon_val, epsilon_dir) in enumerate(epsilon_dirs, 1):
        try:
            # Find Training_* subdirectories
            training_dirs = [d for d in os.listdir(epsilon_dir) if d.startswith('Training_')]
            print(f"\n[{epsilon_idx}/{len(epsilon_dirs)}] Epsilon={epsilon_val}: Found {len(training_dirs)} training directories")
            
            for training_dir in training_dirs:
                training_path = os.path.join(epsilon_dir, training_dir)
                config_path = os.path.join(training_path, 'config.txt')
                
                # Read config file to get omega value
                if not os.path.exists(config_path):
                    print(f"Config file not found: {config_path}")
                    continue
                
                try:
                    with open(config_path, 'r') as f:
                        config_content = f.read()
                        
                    # Extract omega value from config
                    omega_from_config = None
                    for line in config_content.split('\n'):
                        if 'Weber fraction (Omega):' in line:
                            omega_from_config = float(line.split(':')[1].strip())
                            break
                    
                    if omega_from_config is None:
                        print(f"Could not find omega value in config: {config_path}")
                        continue
                    
                    # Filter by omega value if specified
                    if omega_value is not None and abs(omega_from_config - omega_value) > 1e-6:
                        continue  # Skip this training if omega doesn't match
                    
                    print(f"  Processing: omega={omega_from_config}, epsilon={epsilon_val}")
                    
                    # Determine which checkpoint to use
                    actual_checkpoint_num = checkpoint_num
                    if use_latest_checkpoint:
                        latest_checkpoint = find_latest_checkpoint(training_path)
                        if latest_checkpoint is None:
                            print(f"No checkpoints found in {training_path}")
                            continue
                        actual_checkpoint_num = latest_checkpoint
                        print(f"Using latest checkpoint: {actual_checkpoint_num}")
                    
                    # Look for checkpoint
                    checkpoint_path = os.path.join(training_path, f"trained_model_checkpoint_{actual_checkpoint_num}.pkl")
                    
                    if os.path.exists(checkpoint_path):
                        print(f"    Loading checkpoint {actual_checkpoint_num}...")
                        params = load_model_checkpoint(checkpoint_path)
                        
                        if params is not None:
                            try:
                                print(f"    Testing model on {len(test_pairs)} test pairs...")
                                error_distances = test_model_and_get_errors(params, test_pairs, unit_module, carry_module, 
                                                                           model_fn, unit_structure, carry_structure, NUMBER_SIZE)
                                all_error_distances.extend(error_distances)
                                models_tested += 1
                                print(f"    ✓ Completed: {len(error_distances)} errors found (Total so far: {len(all_error_distances)} errors from {models_tested} models)")
                            except Exception as e:
                                print(f"Error testing model omega={omega_from_config}, epsilon={epsilon_val}, checkpoint={actual_checkpoint_num}: {e}")
                        else:
                            print(f"Could not load model from {checkpoint_path}")
                    else:
                        print(f"Checkpoint not found: {checkpoint_path}")
                        
                except Exception as e:
                    print(f"Error reading config file {config_path}: {e}")
                    
        except Exception as e:
            print(f"Error processing epsilon directory {epsilon_dir}: {e}")
    
    if not all_error_distances:
        print(f"\n❌ No error data found for {filter_desc}")
        return
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete: {len(all_error_distances)} total errors from {models_tested} models")
    print(f"{'='*60}\n")
    
    # --- Figure: Errors by distance (normalized/proportion) ---
    # Calculate normalized error distance distribution
    error_distances_series = pd.Series(all_error_distances)
    overall_error = error_distances_series.value_counts(normalize=True).sort_index()
    # Convert the normalized values to percentages for better readability
    overall_error = overall_error * 100  # Convert to percentage
    
    if overall_error.empty:
        print(f"No error distance data available")
        return
    
    # Create the figure with improved styling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to show only distances up to a reasonable maximum (e.g., 20 or max if smaller)
    max_distance = min(200, int(overall_error.index.max()))
    overall_error_filtered = overall_error[overall_error.index <= max_distance]
    
    # Create bar plot with a nice color scheme
    bars = ax.bar(overall_error_filtered.index, overall_error_filtered.values, 
                  color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax.set_ylim(0, max(overall_error_filtered.values) * 1.1)
    plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_axisbelow(True)

    save_configs = [
        (25, 5, 54, 48),
        (100, 10, 32, 28),
        (140, 10, 32, 28),
        (200, 20, 32, 28),
        (1000, 100, 32, 28)
    ]

    checkpoint_label = "latest" if use_latest_checkpoint else str(checkpoint_num)

    for limit, step, label_size, ticks_size in save_configs:
        ax.set_xlim(0, limit)
        # Set labels and title with appropriate font sizes
        ax.set_xlabel('Error Distance', fontsize=label_size)
        ax.set_ylabel('Total Errors (%)', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=ticks_size)

        # Generates 1, 5, 10... or 1, 10, 20...
        current_ticks = [1]
        if max_distance >= 5:
            current_ticks.extend(range(step, limit + 1, step))
        ax.set_xticks(current_ticks)
        ax.set_xticklabels([str(x) for x in current_ticks])
        
        if omega_value is not None:
            safe_om = str(omega_value).replace('.', '_')
            fname = os.path.join(figures_dir, f"errors_by_dist_omega_{safe_om}_ckpt_{checkpoint_label}_limit_{limit}.png")
        else:
            fname = os.path.join(figures_dir, f"errors_by_dist_all_omegas_ckpt_{checkpoint_label}_limit_{limit}.png")
        
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"Saved figure for limit {limit} with step {step}: {fname}")

    plt.close()
    print(f"Percentage figures saved")
    
    # --- Figure: Errors by distance (absolute counts) ---
    # Calculate absolute error distance distribution
    overall_error_counts = error_distances_series.value_counts().sort_index()
    
    if overall_error_counts.empty:
        print(f"No error distance data available for counts")
        return
    
    # Create the figure with improved styling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to show only distances up to a reasonable maximum
    max_distance_counts = min(200, int(overall_error_counts.index.max()))
    overall_error_counts_filtered = overall_error_counts[overall_error_counts.index <= max_distance_counts]
    
    # Create bar plot with a nice color scheme
    bars = ax.bar(overall_error_counts_filtered.index, overall_error_counts_filtered.values, 
                  color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax.set_ylim(0, max(overall_error_counts_filtered.values) * 1.1)
    plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_axisbelow(True)

    for limit, step, label_size, ticks_size in save_configs:
        ax.set_xlim(0, limit)
        ax.set_ylim(0, 305)
        # Set labels and title with appropriate font sizes
        ax.set_xlabel('Error Distance', fontsize=label_size)
        ax.set_ylabel('Number of Errors', fontsize=label_size)
        ax.tick_params(axis='both', labelsize=ticks_size)

        # Generates 1, 5, 10... or 1, 10, 20...
        current_ticks = [1]
        if max_distance_counts >= 5:
            current_ticks.extend(range(step, limit + 1, step))
        ax.set_xticks(current_ticks)
        ax.set_xticklabels([str(x) for x in current_ticks])
        
        if omega_value is not None:
            safe_om = str(omega_value).replace('.', '_')
            fname = os.path.join(figures_dir, f"errors_by_dist_counts_omega_{safe_om}_ckpt_{checkpoint_label}_limit_{limit}.png")
        else:
            fname = os.path.join(figures_dir, f"errors_by_dist_counts_all_omegas_ckpt_{checkpoint_label}_limit_{limit}.png")
        
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"Saved count figure for limit {limit} with step {step}: {fname}")

    plt.close()
    print(f"Count figures saved")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE, CHECKPOINT)