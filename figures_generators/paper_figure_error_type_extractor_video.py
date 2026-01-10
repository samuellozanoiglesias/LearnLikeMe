# USE: nohup python paper_figure_error_type_extractor_video.py unit STUDY 0.15 > logs_paper_error_extractor_video.out 2>&1 &
# Creates plots for all checkpoints for the specified omega value for unit/carry extractors
# Creates TWO videos/gifs from all checkpoint plots:
# 1) Error by input digit (0-9 bars)
# 2) Error by position (units vs carries)
# Always aggregates over all training runs
# Uses full test set of 100 single-digit additions (0+0 to 9+9)

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

def generate_single_digit_test_set():
    """Generate all 100 possible single-digit additions (0+0 to 9+9)."""
    test_pairs = []
    for num1 in range(10):
        for num2 in range(10):
            result = num1 + num2
            test_pairs.append((num1, num2, result))
    return test_pairs

def calculate_digit_frequencies(test_pairs):
    """Calculate how many times each digit (0-9) appears in the test set."""
    digit_counts = {i: 0 for i in range(10)}
    
    for pair in test_pairs:
        num1, num2 = pair[0], pair[1]  # Unpack from (num1, num2, result) tuple
        digit_counts[num1] += 1
        digit_counts[num2] += 1
    
    return digit_counts

@jit
def prepare_batch_inputs_single_digit(nums1, nums2):
    """JIT-compiled vectorized preparation for single-digit inputs."""
    return jnp.column_stack([nums1, nums2])

def prepare_batch_inputs(test_pairs):
    """Vectorized preparation of all test inputs for single-digit addition."""
    batch_size = len(test_pairs)
    
    # Vectorized conversion to arrays
    nums1 = jnp.array([pair[0] for pair in test_pairs])
    nums2 = jnp.array([pair[1] for pair in test_pairs])
    
    # Use JIT-compiled function for single-digit case
    inputs = prepare_batch_inputs_single_digit(nums1, nums2)
    
    return inputs

def vectorized_extractor_error_analysis(predictions, expected_results, test_pairs, extractor_type):
    """Vectorized error detection and counting for extractors."""
    batch_size = len(test_pairs)
    
    # Initialize counters (use float for fractional weights)
    digit_error_counts = jnp.zeros(10, dtype=jnp.float32)
    type_error_counts = jnp.zeros(2, dtype=jnp.float32)  # [unit_errors, carry_errors]
    
    # Extract input numbers for digit association
    nums1 = jnp.array([pair[0] for pair in test_pairs])
    nums2 = jnp.array([pair[1] for pair in test_pairs])
    
    # Find errors (predictions are already class indices, expected_results are class indices)
    errors = predictions != expected_results
    
    # Count errors by input digit (weight 0.5 each for both input digits)
    for digit in range(10):
        count = (0.5 * jnp.sum(errors & (nums1 == digit)) + 
                0.5 * jnp.sum(errors & (nums2 == digit)))
        digit_error_counts = digit_error_counts.at[digit].add(count)
    
    # Count errors by type
    if extractor_type == 'unit':
        type_error_counts = type_error_counts.at[0].add(jnp.sum(errors))
    else:  # carry
        type_error_counts = type_error_counts.at[1].add(jnp.sum(errors))
    
    return digit_error_counts, type_error_counts

def test_extractor_and_get_errors_vectorized(params, batch_inputs, expected_results, extractor_module, model_structure, extractor_type, test_pairs):
    """Optimized batch processing version for extractor testing."""
    try:
        # Create a fresh extractor model with the appropriate architecture
        if extractor_type == 'unit':
            structure = [128, 64]  # Unit model hidden layers
            output_dim = 10
        else:  # carry
            structure = [16]  # Carry model hidden layers  
            output_dim = 2
            
        # Import the model class
        from little_learner.modules.extractor_modules.models import ExtractorModel
        from jax import random
        
        model = ExtractorModel(structure=structure, output_dim=output_dim)
        
        # Initialize the model with dummy data to get parameter structure
        rng = random.PRNGKey(42)
        dummy_input = batch_inputs[:1]  # Use first sample as dummy
        initialized_params = model.init(rng, dummy_input)
        
        # The loaded params should be applied to the initialized structure
        # If params is nested, extract the actual parameters
        if isinstance(params, dict):
            if 'params' in params:
                model_params = {'params': params['params']}
            else:
                model_params = {'params': params}
        else:
            model_params = {'params': params}
            
        # Apply the model to get predictions
        predictions = model.apply(model_params, batch_inputs)
        
        # Model outputs logits/probabilities for each class
        # Take argmax to get actual predictions (class indices)
        if extractor_type == 'unit':
            # Unit extractor: 10 classes (0-9)
            pred_classes = jnp.argmax(predictions, axis=1)
        else:  # carry
            # Carry extractor: 2 classes (0 or 1)
            pred_classes = jnp.argmax(predictions, axis=1)
        
        # Convert to expected results for comparison
        expected_classes = []
        for _, _, result in test_pairs:
            if extractor_type == 'unit':
                expected_classes.append(result % 10)  # Unit digit
            else:  # carry
                expected_classes.append(1 if result >= 10 else 0)  # Carry bit
        
        expected_classes = jnp.array(expected_classes)
        
        # Vectorized error analysis
        digit_errors, type_errors = vectorized_extractor_error_analysis(pred_classes, expected_classes, test_pairs, extractor_type)
        
        # Convert back to Python lists for compatibility
        return list(digit_errors), list(type_errors)
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback to empty results
        return [0] * 10, [0] * 2

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
EXTRACTOR_TYPE = str(sys.argv[1]).lower()  # 'unit' or 'carry' extractor
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
OMEGA_VALUE = float(sys.argv[3])  # Specific omega value to analyze

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"
RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{EXTRACTOR_TYPE}_extractor"

def analyze_extractor_module(raw_dir, figures_dir, omega_value, extractor_type):
    # Create output directory for this omega and extractor type
    safe_om = str(omega_value).replace('.', '_')
    omega_output_dir = os.path.join(figures_dir, f"errors_by_type_{extractor_type}_omega_{safe_om}")
    os.makedirs(omega_output_dir, exist_ok=True)
    
    # Generate complete test set (all 100 single-digit additions)
    test_pairs = generate_single_digit_test_set()
    print(f"Generated {len(test_pairs)} test pairs (all single-digit additions)")
    
    # Calculate digit frequencies in test set
    digit_frequencies = calculate_digit_frequencies(test_pairs)
    total_digit_occurrences = sum(digit_frequencies.values())
    
    # Save digit frequencies to file
    freq_file = os.path.join(omega_output_dir, f"digit_frequencies_single_digit_test_set.txt")
    with open(freq_file, 'w') as f:
        f.write(f"Digit frequencies in single-digit test set\n")
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
    batch_inputs = prepare_batch_inputs(test_pairs)
    expected_results = jnp.array([pair[2] for pair in test_pairs])
    print(f"Batch inputs shape: {batch_inputs.shape}")
    
    # Find all checkpoints across all matching training directories
    all_checkpoints = set()
    training_paths_by_checkpoint = {}
    
    # Find all study directories that match the omega value
    study_dirs = []
    try:
        for item in os.listdir(raw_dir):
            item_path = os.path.join(raw_dir, item)
            if os.path.isdir(item_path):
                study_dirs.append((item, item_path))
        
        print(f"Found {len(study_dirs)} study directories matching omega {omega_value}")
        
    except Exception as e:
        print(f"Error scanning raw directory {raw_dir}: {e}")
        return
    
    if not study_dirs:
        print(f"No study folders found in {raw_dir} for omega {omega_value}")
        return
    
    # First pass: collect all checkpoints and training paths
    for study_name, study_dir in study_dirs:
        # Find Training_* subdirectories
        training_dirs = [d for d in os.listdir(study_dir) if d.startswith('Training_')]
        
        for training_dir in training_dirs:
            training_path = os.path.join(study_dir, training_dir)
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
                    if 'Weber fraction:' in line:
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
                    training_paths_by_checkpoint[checkpoint].append((training_path, omega_from_config))
                    
            except Exception as e:
                print(f"Error reading config file {config_path}: {e}")
                    
    if not all_checkpoints:
        print(f"No checkpoints found for omega={omega_value}")
        return
    
    all_checkpoints = sorted(list(all_checkpoints))
    print(f"Found {len(all_checkpoints)} unique checkpoints: {all_checkpoints}")
    
    # Fixed axes limits for all plots (determined from typical ranges)
    fixed_digit_ylim = (0, 50)  # For digit error rates (0-50%)
    fixed_type_ylim = (0, 100)  # For type errors (units vs carries)
    
    # Second pass: create plots for each checkpoint
    for i, checkpoint_num in enumerate(all_checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_num}...")
        
        # Collect error counts for this checkpoint from all models
        all_model_digit_errors = []  # List to store digit error arrays from each model
        all_model_type_errors = []  # List to store type error arrays from each model
        models_tested = 0
        
        for training_path, omega_from_config in training_paths_by_checkpoint[checkpoint_num]:
            checkpoint_path = os.path.join(training_path, f"trained_model_checkpoint_{checkpoint_num}.pkl")
            
            if os.path.exists(checkpoint_path):
                params = load_model_checkpoint(checkpoint_path)
                
                if params is not None:
                    try:
                        # Use vectorized testing for much faster processing
                        # Checkpoint contains raw model parameters
                        digit_errors, type_errors = test_extractor_and_get_errors_vectorized(
                            params, batch_inputs, expected_results, None, 
                            None, extractor_type, test_pairs
                        )
                        
                        # Store results from this model
                        all_model_digit_errors.append(digit_errors)
                        all_model_type_errors.append(type_errors)
                        models_tested += 1
                        total_errors = sum(digit_errors)
                        print(f"  Tested {extractor_type} extractor omega={omega_from_config}: {total_errors:.2f} digit errors")
                    except Exception as e:
                        print(f"  Error testing {extractor_type} extractor omega={omega_from_config}: {e}")
        
        if len(all_model_digit_errors) == 0:
            print(f"  No error data found for checkpoint {checkpoint_num}")
            continue
        
        print(f"  Found error data from {models_tested} models")
        
        # Calculate mean errors across all models
        all_model_digit_errors = np.array(all_model_digit_errors)  # Shape: (num_models, 10)
        all_model_type_errors = np.array(all_model_type_errors)  # Shape: (num_models, 2)
        
        mean_digit_error_counts = np.mean(all_model_digit_errors, axis=0)  # Mean across models
        mean_type_error_counts = np.mean(all_model_type_errors, axis=0)  # Mean across models
        
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
        ax.set_title(f'{extractor_type.capitalize()} Extractor - Checkpoint {checkpoint_num}', fontsize=24, pad=20)
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
        
        # --- Figure 2: Errors by extractor type ---
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Create bar plot for extractor output type
        if extractor_type == 'unit':
            type_labels = ['Unit Errors']
            type_counts = [mean_type_error_counts[0]]
            bar_color = '#FF6B6B'
        else:  # carry
            type_labels = ['Carry Errors']
            type_counts = [mean_type_error_counts[1]]
            bar_color = '#4ECDC4'
        
        bars = ax.bar([0], type_counts, color=bar_color, alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Set labels and title with appropriate font sizes
        ax.set_xlabel('Error Type', fontsize=32)
        ax.set_ylabel('Mean Number of Errors', fontsize=32)
        ax.set_title(f'{extractor_type.capitalize()} Extractor - Checkpoint {checkpoint_num}', fontsize=24, pad=20)
        ax.tick_params(axis='both', labelsize=28)
        
        # Set x-axis labels
        ax.set_xticks([0])
        ax.set_xticklabels(type_labels)
        
        # Set fixed y-axis limits
        ax.set_ylim(fixed_type_ylim)
        
        # Add grid for better readability
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add value labels on top of bars
        for i, count in enumerate(type_counts):
            if count > fixed_type_ylim[1] * 0.02:  # Only show labels if significant
                ax.text(i, min(count + fixed_type_ylim[1] * 0.01, fixed_type_ylim[1] * 0.95), f'{count:.1f}', 
                       ha='center', va='bottom', fontsize=16)
        
        # Save the type error figure
        fname_type = os.path.join(omega_output_dir, f"errors_by_type_checkpoint_{checkpoint_num}.png")
        plt.savefig(fname_type, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Type error figure saved to: {fname_type}")
    
    # Create videos/gifs from all plots
    print(f"\nCreating videos from {len(all_checkpoints)} plots...")
    
    # Create video for digit errors
    digit_video_path = os.path.join(omega_output_dir, f"error_by_digit_evolution_{extractor_type}_omega_{safe_om}.gif")
    create_video_from_images(omega_output_dir, digit_video_path, duration_per_frame=0.5, pattern="errors_by_digit_*.png")
    
    # Create video for type errors
    type_video_path = os.path.join(omega_output_dir, f"error_by_type_evolution_{extractor_type}_omega_{safe_om}.gif")
    create_video_from_images(omega_output_dir, type_video_path, duration_per_frame=0.5, pattern="errors_by_type_*.png")
    
    print(f"\nAnalysis complete for {extractor_type} extractor omega={omega_value}. Results saved in: {omega_output_dir}")
    print(f"Two videos created:")
    print(f"  - Digit errors: {digit_video_path}")
    print(f"  - Type errors: {type_video_path}")

analyze_extractor_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, EXTRACTOR_TYPE)