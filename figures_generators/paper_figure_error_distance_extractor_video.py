# USE: nohup python paper_figure_error_distance_extractor_video.py unit STUDY 0.15 > logs_paper_error_extractor_distance_video.out 2>&1 &
# Creates plots for all checkpoints for the specified omega value for unit/carry extractors
# Creates a video/gif from all checkpoint plots showing error distance distributions
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

def calculate_error_distance(predicted, actual):
    """Calculate the absolute difference between predicted and actual values."""
    return abs(predicted - actual)

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

def test_extractor_and_get_error_distances_vectorized(params, batch_inputs, test_pairs, extractor_module, model_structure, extractor_type):
    """Test extractor model on batch of test cases and return error distances."""
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
        
        # Calculate expected values and error distances
        error_distances = []
        
        for i, (num1, num2, result) in enumerate(test_pairs):
            pred_value = int(pred_classes[i])  # Already an integer class index
            
            if extractor_type == 'unit':
                # For unit extractor: predict unit digit of result (0-9)
                actual_value = result % 10
            elif extractor_type == 'carry':
                # For carry extractor: predict carry bit (0 or 1)
                actual_value = 1 if result >= 10 else 0
            
            # Calculate error distance
            if pred_value != actual_value:
                distance = abs(pred_value - actual_value)
                error_distances.append(distance)
        
        return error_distances
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
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
    omega_output_dir = os.path.join(figures_dir, f"errors_by_distance_{extractor_type}_omega_{safe_om}")
    os.makedirs(omega_output_dir, exist_ok=True)
    
    # Prepare CSV data collection for all individual errors
    all_errors_data = []
    
    # Generate complete test set (all 100 single-digit additions)
    test_pairs = generate_single_digit_test_set()
    print(f"Generated {len(test_pairs)} test pairs (all single-digit additions)")
    
    # Prepare batch inputs once for all models (major optimization)
    print("Preparing batch inputs for vectorized processing...")
    batch_inputs = prepare_batch_inputs(test_pairs)
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
                # Check if this directory contains the target omega value
                if f"OMEGA_{omega_value:.2f}" in item or f"OMEGA_{omega_value}" in item:
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
    
    # Fixed axes limits for all plots (adapted for extractor ranges)
    if extractor_type == 'unit':
        fixed_xlim = (0, 10)  # Unit extractor can have distances 0-9
        fixed_ylim = (0, 30)  # Adjusted for unit errors
    else:  # carry
        fixed_xlim = (0, 2)   # Carry extractor can only have distances 0-1
        fixed_ylim = (0, 50)  # Adjusted for carry errors
    
    # Second pass: create plots for each checkpoint
    for i, checkpoint_num in enumerate(all_checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_num}...")
        
        # Collect error distributions for this checkpoint (one per model)
        model_error_distributions = []
        models_tested = 0
        
        for training_path, omega_from_config in training_paths_by_checkpoint[checkpoint_num]:
            checkpoint_path = os.path.join(training_path, f"trained_model_checkpoint_{checkpoint_num}.pkl")
            
            if os.path.exists(checkpoint_path):
                params = load_model_checkpoint(checkpoint_path)
                
                if params is not None:
                    try:
                        # Use vectorized testing for much faster processing
                        # Checkpoint contains raw model parameters
                        error_distances = test_extractor_and_get_error_distances_vectorized(
                            params, batch_inputs, test_pairs, None, 
                            None, extractor_type
                        )
                        
                        # Save individual errors to CSV data
                        for error_distance in error_distances:
                            all_errors_data.append({
                                'checkpoint': checkpoint_num,
                                'omega': omega_from_config,
                                'training_path': training_path,
                                'error_distance': error_distance,
                                'extractor_type': extractor_type
                            })
                        
                        # Get error distribution for this individual model
                        if error_distances:
                            error_dist = pd.Series(error_distances).value_counts(normalize=False).sort_index()
                        else:
                            # Create empty distribution for 0 errors case
                            error_dist = pd.Series([], dtype='int64').value_counts(normalize=False).sort_index()
                        model_error_distributions.append(error_dist)
                        models_tested += 1
                        print(f"  Tested {extractor_type} extractor omega={omega_from_config}: {len(error_distances)} errors")
                    except Exception as e:
                        print(f"  Error testing {extractor_type} extractor omega={omega_from_config}: {e}")
        
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
        
        # If no error distances found (all models had 0 errors), create a plot showing 0 errors
        if not all_distances:
            # Create a distribution showing 0 errors for distance 0
            if extractor_type == 'unit':
                mean_error_counts = {0: 0}  # No errors at any distance
            else:  # carry
                mean_error_counts = {0: 0}  # No errors at any distance
        
        overall_error = pd.Series(mean_error_counts).sort_index()
        
        # Create the figure even if there are 0 errors (this is meaningful data)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter to show only relevant distances based on extractor type
        if extractor_type == 'unit':
            max_distance = min(9, int(overall_error.index.max())) if not overall_error.empty else 0
        else:  # carry
            max_distance = min(1, int(overall_error.index.max())) if not overall_error.empty else 0
            
        if not overall_error.empty:
            overall_error_filtered = overall_error[overall_error.index <= max_distance]
        else:
            # Create empty series for 0 errors case
            overall_error_filtered = pd.Series([0], index=[0])
        
        # Create bar plot - even if showing 0 errors
        bars = ax.bar(overall_error_filtered.index, overall_error_filtered.values, 
                      color='#7F6BB3', alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Set labels and title
        ax.set_xlabel('Error Distance', fontsize=32)
        ax.set_ylabel('Mean Error Count', fontsize=32)
        ax.set_title(f'{extractor_type.capitalize()} Extractor - Checkpoint {checkpoint_num}', fontsize=24, pad=20)
        ax.tick_params(axis='both', labelsize=28)
        
        # Set fixed axes limits for all plots
        ax.set_xlim(fixed_xlim)
        ax.set_ylim(fixed_ylim)
        
        # Set x-axis ticks based on extractor type
        if extractor_type == 'unit':
            x_ticks = list(range(0, 10))  # 0-9 for unit extractor
        else:  # carry
            x_ticks = [0, 1]  # 0-1 for carry extractor
            
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
    video_path = os.path.join(omega_output_dir, f"error_distance_evolution_{extractor_type}_omega_{safe_om}.gif")
    create_video_from_images(omega_output_dir, video_path, duration_per_frame=0.25)
    
    # Save all individual errors to CSV
    if all_errors_data:
        csv_path = os.path.join(omega_output_dir, f"all_errors_{extractor_type}_omega_{safe_om}.csv")
        errors_df = pd.DataFrame(all_errors_data)
        errors_df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(all_errors_data)} individual errors to: {csv_path}")
    else:
        print("\nNo error data to save to CSV")
    
    print(f"Analysis complete for {extractor_type} extractor omega={omega_value}. Results saved in: {omega_output_dir}")

analyze_extractor_module(RAW_DIR, FIGURES_DIR, OMEGA_VALUE, EXTRACTOR_TYPE)