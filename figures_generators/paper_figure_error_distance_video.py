# USE: nohup python paper_figure_error_distance_video.py 2 STUDY [RI|WI] argmax 0.15 > logs_paper_error_video_distance.out 2>&1 &
# Creates plots for all checkpoints for the specified omega value
# Creates a video/gif from all checkpoint plots
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
    """Prepare all test inputs as a batch for vectorized processing."""
    batch_inputs = []
    correct_results = []
    
    for num1, num2, correct_result in test_pairs:
        if number_size == 2:
            tens1, units1 = divmod(num1, 10)
            tens2, units2 = divmod(num2, 10)
            x = [tens1, units1, tens2, units2]
        elif number_size == 3:
            hundreds1, remainder1 = divmod(num1, 100)
            tens1, units1 = divmod(remainder1, 10)
            hundreds2, remainder2 = divmod(num2, 100)
            tens2, units2 = divmod(remainder2, 10)
            x = [hundreds1, tens1, units1, hundreds2, tens2, units2]
        else:
            # For other number sizes, convert to digit representation
            digits1 = [int(d) for d in str(num1).zfill(number_size)]
            digits2 = [int(d) for d in str(num2).zfill(number_size)]
            x = digits1 + digits2
        
        batch_inputs.append(x)
        correct_results.append(correct_result)
    
    return jnp.array(batch_inputs), jnp.array(correct_results)

def convert_predictions_to_numbers(predictions, number_size):
    """Convert batch predictions from digits back to numbers."""
    batch_size = predictions.shape[0]
    prediction_values = jnp.zeros(batch_size, dtype=jnp.int32)
    
    # Vectorized conversion of digits to numbers
    powers = jnp.array([10 ** (number_size - i) for i in range(number_size + 1)])
    
    # Round predictions to integers and clip to valid digit range
    digit_predictions = jnp.clip(jnp.round(predictions).astype(jnp.int32), 0, 9)
    
    # Convert digits to numbers using matrix multiplication
    prediction_values = jnp.sum(digit_predictions * powers, axis=1)
    
    return prediction_values

def test_model_and_get_errors_vectorized(params, batch_inputs, correct_results, unit_module, carry_module, model_fn, unit_structure, carry_structure, number_size):
    """Test model on batch of test cases and return error distances for incorrect predictions."""
    try:
        # Get predictions for entire batch at once
        predictions = model_fn(params, batch_inputs, unit_module, carry_module, 
                             unit_structure=unit_structure, 
                             carry_structure=carry_structure)
        
        # Convert predictions to numbers
        prediction_values = convert_predictions_to_numbers(predictions, number_size)
        
        # Find incorrect predictions
        incorrect_mask = prediction_values != correct_results
        
        # Calculate error distances only for incorrect predictions
        error_distances = jnp.abs(prediction_values - correct_results)
        error_distances = error_distances[incorrect_mask]
        
        return error_distances.tolist()
        
    except Exception as e:
        print(f"Error in vectorized model testing: {e}")
        return []

def create_video_from_images(image_folder, output_path, duration_per_frame=0.25):
    """Create a video/gif from a folder of images."""
    # Get all PNG files in the folder and sort them numerically by checkpoint
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    # Sort by checkpoint number extracted from filename
    def extract_checkpoint(filename):
        try:
            # Extract checkpoint number from filename like "errors_by_distance_checkpoint_X.png"
            basename = os.path.basename(filename)
            checkpoint_str = basename.split('_')[-1].replace('.png', '')
            return int(checkpoint_str)
        except:
            return 0
    
    image_files.sort(key=extract_checkpoint)
    
    if not image_files:
        print(f"No images found in {image_folder}")
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
    omega_output_dir = os.path.join(figures_dir, f"errors_by_distance_omega_{safe_om}")
    os.makedirs(omega_output_dir, exist_ok=True)
    
    # Prepare CSV data collection for all individual errors
    all_errors_data = []
    
    # Load test set
    test_pairs = load_test_set(NUMBER_SIZE)
    if not test_pairs:
        print(f"Could not load test set for {NUMBER_SIZE}-digit numbers")
        return
    
    print(f"Loaded {len(test_pairs)} test pairs")
    
    # Prepare batch inputs once for all models (major optimization)
    print("Preparing batch inputs for vectorized processing...")
    batch_inputs, correct_results = prepare_batch_inputs(test_pairs, NUMBER_SIZE)
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
    
    # Fixed axes limits for all plots
    fixed_xlim = (0, 26)
    fixed_ylim = (0, 20)  # Reduced since we're showing mean counts now
    
    # Second pass: create plots for each checkpoint
    for i, checkpoint_num in enumerate(all_checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_num}...")
        
        # Collect error distributions for this checkpoint (one per model)
        model_error_distributions = []
        models_tested = 0
        
        for training_path, epsilon_val, omega_from_config in training_paths_by_checkpoint[checkpoint_num]:
            checkpoint_path = os.path.join(training_path, f"trained_model_checkpoint_{checkpoint_num}.pkl")
            
            if os.path.exists(checkpoint_path):
                params = load_model_checkpoint(checkpoint_path)
                
                if params is not None:
                    try:
                        # Use vectorized testing for much faster processing
                        error_distances = test_model_and_get_errors_vectorized(
                            params, batch_inputs, correct_results, unit_module, carry_module, 
                            model_fn, unit_structure, carry_structure, NUMBER_SIZE
                        )
                        
                        # Save individual errors to CSV data
                        for error_distance in error_distances:
                            all_errors_data.append({
                                'checkpoint': checkpoint_num,
                                'omega': omega_from_config,
                                'epsilon': epsilon_val,
                                'training_path': training_path,
                                'error_distance': error_distance
                            })
                        
                        # Get error distribution for this individual model
                        if error_distances:
                            error_dist = pd.Series(error_distances).value_counts(normalize=False).sort_index()
                            model_error_distributions.append(error_dist)
                        models_tested += 1
                        print(f"  Tested model omega={omega_from_config}, epsilon={epsilon_val}: {len(error_distances)} errors")
                    except Exception as e:
                        print(f"  Error testing model omega={omega_from_config}, epsilon={epsilon_val}: {e}")
        
        if not model_error_distributions:
            print(f"  No error data found for checkpoint {checkpoint_num}")
            continue
        
        print(f"  Found error distributions from {models_tested} models")
        
        # Average error distributions across models
        # First, get all unique error distances across all models
        all_distances = set()
        for dist in model_error_distributions:
            all_distances.update(dist.index)
        
        # Calculate mean count for each error distance
        mean_error_counts = {}
        for distance in sorted(all_distances):
            counts = [dist.get(distance, 0) for dist in model_error_distributions]
            mean_error_counts[distance] = np.mean(counts)
        
        overall_error = pd.Series(mean_error_counts).sort_index()
        
        if overall_error.empty:
            print(f"  No error distance data available for checkpoint {checkpoint_num}")
            continue
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter to show only distances up to a reasonable maximum
        max_distance = min(25, int(overall_error.index.max()))
        overall_error_filtered = overall_error[overall_error.index <= max_distance]
        
        # Create bar plot
        bars = ax.bar(overall_error_filtered.index, overall_error_filtered.values, 
                      color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Set labels and title
        ax.set_xlabel('Error Distance', fontsize=32)
        ax.set_ylabel('Mean Error Count', fontsize=32)
        ax.set_title(f'Checkpoint {checkpoint_num}', fontsize=24, pad=20)
        ax.tick_params(axis='both', labelsize=28)
        
        # Set fixed axes limits for all plots
        ax.set_xlim(fixed_xlim)
        ax.set_ylim(fixed_ylim)
        
        # Set x-axis ticks at specific values
        x_ticks = [1, 5, 10, 15, 20, 25]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])
        
        # Add grid
        plt.grid(axis="y", linestyle="--", linewidth=1, color="gray", alpha=0.7)
        ax.set_axisbelow(True)
        
        # Save the figure
        fname = os.path.join(omega_output_dir, f"errors_by_distance_checkpoint_{checkpoint_num}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Figure saved to: {fname}")
    
    # Create video/gif from all plots
    print(f"\nCreating video from {len(all_checkpoints)} plots...")
    video_path = os.path.join(omega_output_dir, f"error_distance_evolution_omega_{safe_om}.gif")
    create_video_from_images(omega_output_dir, video_path, duration_per_frame=0.25)
    
    # Save all individual errors to CSV
    if all_errors_data:
        csv_path = os.path.join(omega_output_dir, f"all_errors_omega_{safe_om}.csv")
        errors_df = pd.DataFrame(all_errors_data)
        errors_df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(all_errors_data)} individual errors to: {csv_path}")
    else:
        print("\nNo error data to save to CSV")
    
    print(f"Analysis complete for omega={omega_value}. Results saved in: {omega_output_dir}")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, PARAM_TYPE)