# USE: nohup python paper_figure_error_type_video.py 2 STUDY [RI|WI] argmax 0.15 > logs_paper_error_video_type.out 2>&1 &
# Creates plots for all checkpoints for the specified omega value
# Creates TWO videos/gifs from all checkpoint plots:
# 1) Error by input digit (0-9 bars)
# 2) Error by position (units, tens, hundreds)
# Always aggregates over all epsilons

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
import imageio
import glob

# Add the parent directory to Python path to find little_learner module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Allow JAX to use GPU if available
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

def find_all_checkpoints(training_dir):
    """Find all checkpoint numbers in a training directory."""
    checkpoints = []
    try:
        for filename in os.listdir(training_dir):
            if filename.startswith('trained_model_checkpoint_') and filename.endswith('.pkl'):
                # Extract checkpoint number from filename
                checkpoint_str = filename.replace('trained_model_checkpoint_', '').replace('.pkl', '')
                try:
                    checkpoint_num = int(checkpoint_str)
                    checkpoints.append(checkpoint_num)
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error scanning directory {training_dir}: {e}")
    
    return sorted(checkpoints)

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

def test_model_and_get_errors_vectorized(params, batch_inputs, expected_results, unit_module, carry_module, model_fn, unit_structure, carry_structure, number_size, test_pairs):
    """Optimized batch processing version with chunking for memory efficiency."""
    try:
        # Single model call for entire batch - direct call (JIT handled internally by model)
        predictions = model_fn(params, batch_inputs, unit_module, carry_module,
                               unit_structure=unit_structure, carry_structure=carry_structure)
        
        # Vectorized error analysis
        digit_errors, position_errors = vectorized_error_analysis(predictions, expected_results, test_pairs, number_size)
        
        # Convert back to Python lists for compatibility
        return list(digit_errors), list(position_errors)
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback to empty results
        return [0] * 10, [0] * 3

def create_video_from_images(image_folder, output_path, duration_per_frame=0.25, pattern="*.png"):
    """Create a video/gif from a folder of images."""
    # Get all PNG files in the folder and sort them numerically by checkpoint
    image_files = glob.glob(os.path.join(image_folder, pattern))
    
    # Sort by checkpoint number extracted from filename
    def extract_checkpoint(filename):
        try:
            # Extract checkpoint number from filename like "errors_by_digit_checkpoint_X.png"
            basename = os.path.basename(filename)
            checkpoint_str = basename.split('_')[-1].replace('.png', '')
            return int(checkpoint_str)
        except:
            return 0
    
    image_files.sort(key=extract_checkpoint)
    
    if not image_files:
        print(f"No images found in {image_folder} with pattern {pattern}")
        return
    
    print(f"Creating video from {len(image_files)} images...")
    
    # Create gif
    gif_path = output_path.replace('.mp4', '.gif')
    with imageio.get_writer(gif_path, mode='I', duration=duration_per_frame) as writer:
        for image_file in image_files:
            try:
                image = imageio.imread(image_file)
                writer.append_data(image)
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
    
    print(f"Video/gif saved to: {gif_path}")

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module
OMEGA_VALUE = float(sys.argv[5])  # Specific omega value to analyze

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

def analyze_multidigit_module(raw_dir, figures_dir, omega_value, param_type):
    # Create output directory for this omega
    safe_om = str(omega_value).replace('.', '_')
    omega_output_dir = os.path.join(figures_dir, f"errors_by_type_omega_{safe_om}")
    os.makedirs(omega_output_dir, exist_ok=True)
    
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
    freq_file = os.path.join(omega_output_dir, f"digit_frequencies_{NUMBER_SIZE}digit_test_set.txt")
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
    
    # Prepare batch inputs once for all models (major optimization)
    print("Preparing batch inputs for vectorized processing...")
    batch_inputs = prepare_batch_inputs(test_pairs, NUMBER_SIZE)
    expected_results = jnp.array([pair[2] for pair in test_pairs])
    print(f"Batch inputs shape: {batch_inputs.shape}")
    
    # Load extractor modules (needed for decision model)
    try:
        carry_module, carry_dir, carry_structure = load_extractor_module(omega_value, 
                                                                        f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe", 
                                                                        model_type='carry_extractor', 
                                                                        study_name=STUDY_NAME)
        unit_module, unit_dir, unit_structure = load_extractor_module(omega_value, 
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
    
    # Find all checkpoints across all matching training directories
    all_checkpoints = set()
    training_paths_by_checkpoint = {}
    
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
    
    # First pass: collect all checkpoints and training paths
    for epsilon_val, epsilon_dir in epsilon_dirs:
        try:
            # Find Training_* subdirectories
            training_dirs = [d for d in os.listdir(epsilon_dir) if d.startswith('Training_')]
            
            for training_dir in training_dirs:
                training_path = os.path.join(epsilon_dir, training_dir)
                config_path = os.path.join(training_path, 'config.txt')
                
                # Read config file to get omega value
                if not os.path.exists(config_path):
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
                        continue
                    
                    # Filter by omega value
                    if abs(omega_from_config - omega_value) > 1e-6:
                        continue  # Skip this training if omega doesn't match
                    
                    # Find all checkpoints in this training directory
                    checkpoints = find_all_checkpoints(training_path)
                    for checkpoint in checkpoints:
                        all_checkpoints.add(checkpoint)
                        if checkpoint not in training_paths_by_checkpoint:
                            training_paths_by_checkpoint[checkpoint] = []
                        training_paths_by_checkpoint[checkpoint].append((training_path, epsilon_val, omega_from_config))
                        
                except Exception as e:
                    print(f"Error reading config file {config_path}: {e}")
                    
        except Exception as e:
            print(f"Error processing epsilon directory {epsilon_dir}: {e}")
    
    if not all_checkpoints:
        print(f"No checkpoints found for omega={omega_value}")
        return
    
    all_checkpoints = sorted(list(all_checkpoints))
    print(f"Found {len(all_checkpoints)} unique checkpoints: {all_checkpoints}")
    
    # Fixed axes limits for all plots (determined from typical ranges)
    fixed_digit_ylim = (0, 100)  # For digit error rates (0-100%)
    fixed_position_ylim = (0, 200)  # For position errors (units, tens, hundreds)
    
    # Second pass: create plots for each checkpoint
    for i, checkpoint_num in enumerate(all_checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_num}...")
        
        # Collect error counts for this checkpoint from all models
        all_model_digit_errors = []  # List to store digit error arrays from each model
        all_model_position_errors = []  # List to store position error arrays from each model
        models_tested = 0
        
        for training_path, epsilon_val, omega_from_config in training_paths_by_checkpoint[checkpoint_num]:
            checkpoint_path = os.path.join(training_path, f"trained_model_checkpoint_{checkpoint_num}.pkl")
            
            if os.path.exists(checkpoint_path):
                params = load_model_checkpoint(checkpoint_path)
                
                if params is not None:
                    try:
                        # Use vectorized testing for much faster processing
                        digit_errors, position_errors = test_model_and_get_errors_vectorized(
                            params, batch_inputs, expected_results, unit_module, carry_module, 
                            model_fn, unit_structure, carry_structure, NUMBER_SIZE, test_pairs
                        )
                        
                        # Store results from this model
                        all_model_digit_errors.append(digit_errors)
                        all_model_position_errors.append(position_errors)
                        models_tested += 1
                        total_errors = sum(digit_errors)
                        print(f"  Tested model omega={omega_from_config}, epsilon={epsilon_val}: {total_errors:.2f} digit errors")
                    except Exception as e:
                        print(f"  Error testing model omega={omega_from_config}, epsilon={epsilon_val}: {e}")
        
        if len(all_model_digit_errors) == 0:
            print(f"  No error data found for checkpoint {checkpoint_num}")
            continue
        
        print(f"  Found error data from {models_tested} models")
        
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
        
        # --- Figure 1: Errors by input digit ---
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create bar plot for each digit (0-9) - now showing error rates
        digits = list(range(10))
        bars = ax.bar(digits, mean_digit_error_rates, 
                      color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Set labels and title with appropriate font sizes
        ax.set_xlabel('Input Digit', fontsize=32)
        ax.set_ylabel('Error Rate (%)', fontsize=32)
        ax.set_title(f'Checkpoint {checkpoint_num}', fontsize=24, pad=20)
        ax.tick_params(axis='both', labelsize=28)
        
        # Set x-axis to show all digits 0-9
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits])
        
        # Set fixed y-axis limits
        ax.set_ylim(fixed_digit_ylim)
        
        # Add grid for better readability
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add value labels on top of bars (if not too crowded)
        for i, rate in enumerate(mean_digit_error_rates):
            if rate > fixed_digit_ylim[1] * 0.05:  # Only show labels if significant
                ax.text(i, min(rate + fixed_digit_ylim[1] * 0.01, fixed_digit_ylim[1] * 0.95), f'{rate:.1f}%', 
                       ha='center', va='bottom', fontsize=16)
        
        # Save the digit error figure
        fname_digit = os.path.join(omega_output_dir, f"errors_by_digit_checkpoint_{checkpoint_num}.png")
        plt.savefig(fname_digit, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Digit error figure saved to: {fname_digit}")
        
        # --- Figure 2: Errors by position (units, tens, hundreds) ---
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
        ax.set_title(f'Checkpoint {checkpoint_num}', fontsize=24, pad=20)
        ax.tick_params(axis='both', labelsize=28)
        
        # Set x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels(relevant_labels)
        
        # Set fixed y-axis limits
        ax.set_ylim(fixed_position_ylim)
        
        # Add grid for better readability
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add value labels on top of bars
        for i, count in enumerate(relevant_counts):
            if count > fixed_position_ylim[1] * 0.02:  # Only show labels if significant
                ax.text(i, min(count + fixed_position_ylim[1] * 0.01, fixed_position_ylim[1] * 0.95), f'{count:.1f}', 
                       ha='center', va='bottom', fontsize=16)
        
        # Save the position error figure
        fname_position = os.path.join(omega_output_dir, f"errors_by_position_checkpoint_{checkpoint_num}.png")
        plt.savefig(fname_position, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Position error figure saved to: {fname_position}")
    
    # Create videos/gifs from all plots
    print(f"\nCreating videos from {len(all_checkpoints)} plots...")
    
    # Create video for digit errors
    digit_video_path = os.path.join(omega_output_dir, f"error_by_digit_evolution_omega_{safe_om}.gif")
    create_video_from_images(omega_output_dir, digit_video_path, duration_per_frame=0.5, pattern="errors_by_digit_*.png")
    
    # Create video for position errors
    position_video_path = os.path.join(omega_output_dir, f"error_by_position_evolution_omega_{safe_om}.gif")
    create_video_from_images(omega_output_dir, position_video_path, duration_per_frame=0.5, pattern="errors_by_position_*.png")
    
    print(f"\nAnalysis complete for omega={omega_value}. Results saved in: {omega_output_dir}")
    print(f"Two videos created:")
    print(f"  - Digit errors: {digit_video_path}")
    print(f"  - Position errors: {position_video_path}")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)