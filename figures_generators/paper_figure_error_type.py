# USE: nohup python paper_figure_error_type.py 2 STUDY [RI|WI] argmax [0.15|none] CHECKPOINT > logs_paper_error_type.out 2>&1 &
# If omega is specified (e.g., 0.15), generates figure for that specific omega value
# If omega is 'none' or omitted, generates figure aggregated over all omegas
# Always aggregates over all epsilons
# CHECKPOINT: specific checkpoint/batch number to load model from
# This script analyzes error types by input digit - tracks which input digits cause errors
# when the model fails in units vs tens positions, creating a 10-column bar plot (digits 0-9)

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

# Allow JAX to use GPU for better performance
import jax.numpy as jnp
import jax
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

def calculate_digit_frequencies(test_pairs, number_size):
    """Calculate how many times each digit (0-9) appears in the test set."""
    digit_counts = {i: 0 for i in range(10)}
    
    for pair in test_pairs:
        num1, num2 = pair[0], pair[1]  # Unpack from (num1, num2, result) tuple
        # Extract digits from num1
        num1_str = str(num1).zfill(number_size)
        for digit_char in num1_str:
            digit_counts[int(digit_char)] += 1
        
        # Extract digits from num2
        num2_str = str(num2).zfill(number_size)
        for digit_char in num2_str:
            digit_counts[int(digit_char)] += 1
    
    return digit_counts

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

def detect_error_positions(predicted, actual, number_size):
    """Detect which positions (units, tens, etc.) have errors. Returns list of error positions."""
    predicted = int(predicted)
    actual = max(0, int(actual))  # Actual should always be non-negative
    
    # If prediction is negative, all positions have errors (negative != positive)
    if predicted < 0:
        return list(range(number_size + 1))  # All positions: 0=units, 1=tens, 2=hundreds, etc.
    
    # Convert numbers to digit arrays for comparison
    pred_str = str(predicted).zfill(number_size + 1)
    actual_str = str(actual).zfill(number_size + 1)
    
    # Handle case where predicted number has more digits than expected
    if len(pred_str) > number_size + 1:
        pred_str = pred_str[-(number_size + 1):]  # Take the rightmost digits
    
    try:
        pred_digits = [int(d) for d in pred_str]
        actual_digits = [int(d) for d in actual_str]
    except ValueError as e:
        print(f"Error converting digits: predicted={predicted}, actual={actual}, pred_str='{pred_str}', actual_str='{actual_str}'")
        # For conversion errors, assume all positions have errors
        return list(range(number_size + 1))
    
    error_positions = []
    # Find all positions where digits differ (from right to left, 0=units, 1=tens, etc.)
    for i in range(len(pred_digits)):
        pos_from_right = len(pred_digits) - 1 - i
        if i < len(actual_digits) and pred_digits[i] != actual_digits[i]:
            error_positions.append(pos_from_right)  # 0 for units, 1 for tens, etc.
    
    return error_positions if error_positions else []  # Return empty list if no differences found

@jit
def prepare_batch_inputs_2digit(nums1, nums2):
    """JIT-compiled vectorized preparation for 2-digit inputs."""
    tens1 = nums1 // 10
    units1 = nums1 % 10
    tens2 = nums2 // 10
    units2 = nums2 % 10
    return jnp.column_stack([tens1, units1, tens2, units2])

@jit
def prepare_batch_inputs_3digit(nums1, nums2):
    """JIT-compiled vectorized preparation for 3-digit inputs."""
    hundreds1 = nums1 // 100
    tens1 = (nums1 % 100) // 10
    units1 = nums1 % 10
    hundreds2 = nums2 // 100
    tens2 = (nums2 % 100) // 10
    units2 = nums2 % 10
    return jnp.column_stack([hundreds1, tens1, units1, hundreds2, tens2, units2])

def prepare_batch_inputs(test_pairs, number_size):
    """Vectorized preparation of all test inputs."""
    batch_size = len(test_pairs)
    
    # Vectorized conversion to digit arrays  
    nums1 = jnp.array([pair[0] for pair in test_pairs])
    nums2 = jnp.array([pair[1] for pair in test_pairs])
    
    if number_size == 2:
        # Use JIT-compiled function for 2-digit case
        inputs = prepare_batch_inputs_2digit(nums1, nums2)
    elif number_size == 3:
        # Use JIT-compiled function for 3-digit case  
        inputs = prepare_batch_inputs_3digit(nums1, nums2)
    
    else:
        # For other number sizes
        input_width = number_size * 2
        inputs = jnp.zeros((batch_size, input_width), dtype=jnp.int32)
        
        for i, (num1, num2) in enumerate(test_pairs):
            digits1 = [int(d) for d in str(num1).zfill(number_size)]
            digits2 = [int(d) for d in str(num2).zfill(number_size)]
            inputs = inputs.at[i].set(jnp.array(digits1 + digits2))
    
    return inputs

def vectorized_error_analysis(predictions, expected_results, test_pairs, number_size):
    """Vectorized error detection and counting with fractional weights."""
    batch_size = len(test_pairs)
    
    # Round predictions to integers
    pred_digits = jnp.round(predictions).astype(jnp.int32)
    
    # Convert expected results to digit arrays
    expected_digits = jnp.zeros((batch_size, number_size + 1), dtype=jnp.int32)
    for i, result in enumerate(expected_results):
        result_str = str(int(result)).zfill(number_size + 1)
        expected_digits = expected_digits.at[i].set(jnp.array([int(d) for d in result_str]))
    
    # Find errors: True where predicted != expected
    error_mask = pred_digits != expected_digits  # Shape: (batch_size, number_size+1)
    
    # Initialize counters (use float for fractional weights)
    digit_error_counts = jnp.zeros(10, dtype=jnp.float32)
    position_error_counts = jnp.zeros(3, dtype=jnp.float32)
    
    # Extract input numbers for digit association
    nums1 = jnp.array([pair[0] for pair in test_pairs])
    nums2 = jnp.array([pair[1] for pair in test_pairs])
    
    # Extract digits from input numbers
    if number_size == 2:
        units1 = nums1 % 10
        tens1 = (nums1 // 10) % 10
        units2 = nums2 % 10  
        tens2 = (nums2 // 10) % 10
        
        # Process position errors vectorized
        for pos in range(min(3, number_size + 1)):
            pos_errors = error_mask[:, -(pos+1)]  # Position from right (0=units, 1=tens, etc.)
            position_error_counts = position_error_counts.at[pos].add(jnp.sum(pos_errors))
            
            if pos == 0:  # Units errors - weight 0.5 each (2 input digits)
                # Units error → associate with units digits (0.5 weight each)
                for digit in range(10):
                    count = 0.5 * jnp.sum(pos_errors & (units1 == digit)) + 0.5 * jnp.sum(pos_errors & (units2 == digit))
                    digit_error_counts = digit_error_counts.at[digit].add(count)
                    
            elif pos == 1:  # Tens errors - weight 0.25 each (4 input digits)
                # Tens error → associate with units and tens digits (0.25 weight each)
                for digit in range(10):
                    count = (0.25 * jnp.sum(pos_errors & (units1 == digit)) + 
                           0.25 * jnp.sum(pos_errors & (units2 == digit)) +
                           0.25 * jnp.sum(pos_errors & (tens1 == digit)) +
                           0.25 * jnp.sum(pos_errors & (tens2 == digit)))
                    digit_error_counts = digit_error_counts.at[digit].add(count)
                    
            elif pos == 2:  # Hundreds errors - weight 0.5 each (2 input digits)
                # Hundreds error → associate with tens digits (0.5 weight each)
                for digit in range(10):
                    count = 0.5 * jnp.sum(pos_errors & (tens1 == digit)) + 0.5 * jnp.sum(pos_errors & (tens2 == digit))
                    digit_error_counts = digit_error_counts.at[digit].add(count)
    
    elif number_size == 3:
        units1 = nums1 % 10
        tens1 = (nums1 // 10) % 10
        hundreds1 = (nums1 // 100) % 10
        units2 = nums2 % 10
        tens2 = (nums2 // 10) % 10
        hundreds2 = (nums2 // 100) % 10
        
        for pos in range(min(3, number_size + 1)):
            pos_errors = error_mask[:, -(pos+1)]
            position_error_counts = position_error_counts.at[pos].add(jnp.sum(pos_errors))
            
            if pos == 0:  # Units - weight 0.5 each (2 input digits)
                for digit in range(10):
                    count = 0.5 * jnp.sum(pos_errors & (units1 == digit)) + 0.5 * jnp.sum(pos_errors & (units2 == digit))
                    digit_error_counts = digit_error_counts.at[digit].add(count)
            elif pos == 1:  # Tens - weight 0.25 each (4 input digits)
                for digit in range(10):
                    count = (0.25 * jnp.sum(pos_errors & (units1 == digit)) +
                           0.25 * jnp.sum(pos_errors & (units2 == digit)) +
                           0.25 * jnp.sum(pos_errors & (tens1 == digit)) +
                           0.25 * jnp.sum(pos_errors & (tens2 == digit)))
                    digit_error_counts = digit_error_counts.at[digit].add(count)
            elif pos == 2:  # Hundreds - weight 0.5 each (2 input digits)
                for digit in range(10):
                    count = 0.5 * jnp.sum(pos_errors & (tens1 == digit)) + 0.5 * jnp.sum(pos_errors & (tens2 == digit))
                    digit_error_counts = digit_error_counts.at[digit].add(count)
    
    return digit_error_counts, position_error_counts

def test_model_and_get_errors(params, test_pairs, unit_module, carry_module, model_fn, unit_structure, carry_structure):
    """Optimized batch processing version with chunking for memory efficiency."""
    try:
        total_pairs = len(test_pairs)
        chunk_size = min(BATCH_CHUNK_SIZE, total_pairs)  # Process in chunks to manage memory
        
        total_digit_errors = jnp.zeros(10, dtype=jnp.float32)
        total_position_errors = jnp.zeros(3, dtype=jnp.float32)
        
        # Process in chunks for memory efficiency
        num_chunks = (total_pairs + chunk_size - 1) // chunk_size
        for chunk_idx, i in enumerate(range(0, total_pairs, chunk_size)):
            if chunk_idx % 10 == 0:  # Progress reporting
                print(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({i}/{total_pairs} pairs)")
                
            end_idx = min(i + chunk_size, total_pairs)
            chunk_pairs = test_pairs[i:end_idx]
            
            # Prepare inputs for this chunk
            batch_inputs = prepare_batch_inputs(chunk_pairs, NUMBER_SIZE)
            expected_results = jnp.array([pair[2] for pair in chunk_pairs])
            
            # Single model call for this chunk - direct call (JIT handled internally by model)
            predictions = model_fn(params, batch_inputs, unit_module, carry_module,
                                   unit_structure=unit_structure, carry_structure=carry_structure)
            
            # Vectorized error analysis for this chunk
            digit_errors, position_errors = vectorized_error_analysis(predictions, expected_results, chunk_pairs, NUMBER_SIZE)
            
            # Accumulate results
            total_digit_errors = total_digit_errors + digit_errors
            total_position_errors = total_position_errors + position_errors
            
            # Memory cleanup between chunks
            del batch_inputs, expected_results, predictions
            if hasattr(jax, 'clear_caches'):
                jax.clear_caches()
        
        # Convert back to Python lists for compatibility
        return list(total_digit_errors), list(total_position_errors)
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback to empty results
        return [0] * 10, [0] * 3

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].lower() != 'none' else None  # Specific omega value to analyze, or None for all omegas
CHECKPOINT = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].lower() != 'none' else None  # Checkpoint/batch number to load, or None for latest

# Performance optimization settings
BATCH_CHUNK_SIZE = 2000  # Adjust based on GPU memory (larger = faster but more memory)

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
    
    # Calculate digit frequencies in test set
    digit_frequencies = calculate_digit_frequencies(test_pairs, NUMBER_SIZE)
    total_digit_occurrences = sum(digit_frequencies.values())
    
    # Save digit frequencies to file
    freq_file = os.path.join(figures_dir, f"digit_frequencies_{NUMBER_SIZE}digit_test_set.txt")
    with open(freq_file, 'w') as f:
        f.write(f"Digit frequencies in {NUMBER_SIZE}-digit test set\n")
        f.write(f"Total test pairs: {len(test_pairs)}\n")
        f.write(f"Total digit occurrences: {total_digit_occurrences}\n")
        f.write("\n")
        f.write("Digit | Count | Percentage\n")
        f.write("------|-------|----------\n")
        for digit in range(10):
            count = digit_frequencies[digit]
            percentage = (count / total_digit_occurrences) * 100
            f.write(f"  {digit}   | {count:5d} | {percentage:6.2f}%\n")
    
    print(f"Digit frequencies saved to: {freq_file}")
    print("Digit frequencies in test set:")
    for digit in range(10):
        print(f"  Digit {digit}: {digit_frequencies[digit]} occurrences ({digit_frequencies[digit]/total_digit_occurrences*100:.2f}%)")
    
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
    
    # Collect digit error counts from all models to calculate means
    all_model_digit_errors = []  # List to store digit error arrays from each model
    all_model_position_errors = []  # List to store position error arrays from each model
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
    
    # Process each epsilon directory with better memory management
    for epsilon_val, epsilon_dir in epsilon_dirs:
        try:
            # Find Training_* subdirectories
            training_dirs = [d for d in os.listdir(epsilon_dir) if d.startswith('Training_')]
            
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
                    
                    print(f"Found training with omega={omega_from_config}, epsilon={epsilon_val}")
                    
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
                        print(f"Loading model from: {checkpoint_path}")
                        params = load_model_checkpoint(checkpoint_path)
                        
                        if params is not None:
                            try:
                                digit_errors, position_errors = test_model_and_get_errors(params, test_pairs, unit_module, carry_module, 
                                                                                          model_fn, unit_structure, carry_structure)
                                # Store results from this model
                                all_model_digit_errors.append(digit_errors)
                                all_model_position_errors.append(position_errors)
                                models_tested += 1
                                total_errors = sum(digit_errors)
                                print(f"Tested model omega={omega_from_config}, epsilon={epsilon_val}, checkpoint={actual_checkpoint_num}: {total_errors:.2f} digit errors")
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
    
    if len(all_model_digit_errors) == 0:
        print(f"No error data found for {filter_desc}")
        return
    
    # Calculate mean errors across all models
    all_model_digit_errors = np.array(all_model_digit_errors)  # Shape: (num_models, 10)
    all_model_position_errors = np.array(all_model_position_errors)  # Shape: (num_models, 3)
    
    mean_digit_error_counts = np.mean(all_model_digit_errors, axis=0)  # Mean across models
    mean_position_error_counts = np.mean(all_model_position_errors, axis=0)  # Mean across models
    
    # Normalize error counts by digit frequency to get error rates
    mean_digit_error_rates = np.zeros(10)
    for digit in range(10):
        if digit_frequencies[digit] > 0:
            mean_digit_error_rates[digit] = (mean_digit_error_counts[digit] / digit_frequencies[digit]) * 100
        else:
            mean_digit_error_rates[digit] = 0
    
    total_mean_errors = np.sum(mean_digit_error_counts)
    print(f"Mean errors per model: {total_mean_errors:.2f} from {models_tested} models")
    print(f"Mean error counts by digit: {[f'{x:.2f}' for x in mean_digit_error_counts]}")
    print(f"Mean error rates by digit (%): {[f'{x:.2f}' for x in mean_digit_error_rates]}")
    
    # --- Figure: Errors by input digit ---
    # Create the figure with improved styling
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bar plot for each digit (0-9) - now showing error rates
    digits = list(range(10))
    bars = ax.bar(digits, mean_digit_error_rates, 
                  color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # Set labels and title with appropriate font sizes
    ax.set_xlabel('Input Digit', fontsize=32)
    ax.set_ylabel('Error Rate (%)', fontsize=32)
    ax.tick_params(axis='both', labelsize=28)
    
    # Set x-axis to show all digits 0-9
    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits])
    
    # Set y-axis limits
    max_error_rate = max(mean_digit_error_rates) if max(mean_digit_error_rates) > 0 else 1
    ax.set_ylim(0, max_error_rate * 1.1)
    
    # Add grid for better readability
    plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for i, rate in enumerate(mean_digit_error_rates):
        if rate > 0:
            ax.text(i, rate + max_error_rate * 0.01, f'{rate:.1f}%', 
                   ha='center', va='bottom', fontsize=20)
    
    # Save the figure
    checkpoint_label = "latest" if use_latest_checkpoint else str(checkpoint_num)
    if omega_value is not None:
        safe_om = str(omega_value).replace('.', '_')
        fname = os.path.join(figures_dir, f"errors_by_input_digit_omega_{safe_om}_checkpoint_{checkpoint_label}.png")
    else:
        fname = os.path.join(figures_dir, f"errors_by_input_digit_all_omegas_checkpoint_{checkpoint_label}.png")
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Error by input digit figure saved to: {fname}")
    
    # --- Figure: Errors by position (units, tens, hundreds) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create bar plot for each position
    position_labels = ['Units', 'Tens', 'Hundreds']
    # Only show positions relevant to the number size
    relevant_positions = min(NUMBER_SIZE + 1, 3)  # +1 for carry digit
    positions = list(range(relevant_positions))
    relevant_counts = mean_position_error_counts[:relevant_positions]
    relevant_labels = position_labels[:relevant_positions]
    
    bars = ax.bar(positions, relevant_counts, 
                  color='#FF6B6B', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # Set labels and title with appropriate font sizes
    ax.set_xlabel('Error Position', fontsize=32)
    ax.set_ylabel('Mean Number of Errors', fontsize=32)
    ax.tick_params(axis='both', labelsize=28)
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(relevant_labels)
    
    # Set y-axis limits
    max_pos_errors = max(relevant_counts) if max(relevant_counts) > 0 else 1
    ax.set_ylim(0, max_pos_errors * 1.1)
    
    # Add grid for better readability
    plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for i, count in enumerate(relevant_counts):
        if count > 0:
            ax.text(i, count + max_pos_errors * 0.01, f'{count:.2f}', 
                   ha='center', va='bottom', fontsize=20)
    
    # Save the position error figure
    if omega_value is not None:
        safe_om = str(omega_value).replace('.', '_')
        fname_pos = os.path.join(figures_dir, f"errors_by_position_omega_{safe_om}_checkpoint_{checkpoint_label}.png")
    else:
        fname_pos = os.path.join(figures_dir, f"errors_by_position_all_omegas_checkpoint_{checkpoint_label}.png")
    plt.savefig(fname_pos, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Error by position figure saved to: {fname_pos}")
    
    print(f"Mean position error counts: Units={mean_position_error_counts[0]:.2f}, Tens={mean_position_error_counts[1]:.2f}, Hundreds={mean_position_error_counts[2]:.2f}")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE, CHECKPOINT)