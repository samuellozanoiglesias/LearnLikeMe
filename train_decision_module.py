# USE: nohup python train_decision_module.py 0.01 WI 0.05 argmax > logs_train_decision.out 2>&1 &
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
#print(jax.devices())  # should only show CPU

import jax.numpy as jnp
import pandas as pd
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from little_learner.modules.decision_module.utils import (
    load_dataset, generate_train_dataset, generate_test_dataset,
    save_results_and_module, load_extractor_module, initialize_decision_params, load_initial_params
)
from little_learner.modules.decision_module.train_utils import (
    evaluate_module, update_params, _make_hashable, _parse_structure
)
from little_learner.modules.decision_module.model import decision_model_argmax, decision_model_vector

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
EPSILON = float(sys.argv[1])  # Noise factor for parameter initialization
PARAM_TYPE = str(sys.argv[2]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
OMEGA = float(sys.argv[3])  # Omega value for loading pre-trained modules
MODEL_TYPE = str(sys.argv[4]).lower() # "argmax" for argmax outputs, "vector" for probability vector outputs

# --- Training Parameters ---
LEARNING_RATE = 0.005
EPOCHS = 7250
BATCH_SIZE = 100
FINISH_TOLERANCE = 0.0  # Tolerance for stopping training when accuracy reaches 1.0
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 500
PARAMS_FILE = None  # Set to None to create new params, or provide a path to load existing params
TRAINING_DISTRIBUTION_TYPE = "Decreasing_exponential"  # Use curriculum learning for training

# --- Paths ---
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
    CODE_DIR = "/home/samuel_lozano/LearnLikeMe"
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
RAW_DIR = f"{MODULES_DIR}/decision_module"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/{MODEL_TYPE}_version/epsilon_{EPSILON:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"
DATASET_DIR = f"{CODE_DIR}/datasets"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Data Preparation ---
# Load datasets and extractors
all_pairs = load_dataset(os.path.join(DATASET_DIR, "all_valid_additions.txt"))
train_pairs = load_dataset(os.path.join(DATASET_DIR, "train_pairs_not_in_stimuli.txt"))
test_pairs = load_dataset(os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
carry = load_dataset(os.path.join(DATASET_DIR, "carry_additions.txt"))
small = load_dataset(os.path.join(DATASET_DIR, "small_additions.txt"))
large = load_dataset(os.path.join(DATASET_DIR, "large_additions.txt"))

test_set = set(test_pairs)
carry_set = set(carry)
small_set = set(small)
large_set = set(large)

totals = [0, 0, 0, 0, 0, 0]
totals[4] = len(test_set & small_set)
totals[5] = len(test_set & large_set)
totals[2] = len(test_set & carry_set & small_set)
totals[3] = len(test_set & carry_set & large_set)
totals[0] = totals[4] - totals[2]
totals[1] = totals[5] - totals[3]

# For JAX static arguments, also create tuple versions
carry_tuple = tuple(sorted(carry_set))
small_tuple = tuple(sorted(small_set))
large_tuple = tuple(sorted(large_set))

carry_module, carry_dir, carry_structure = load_extractor_module(OMEGA, MODULES_DIR, model_type='carry_extractor')
unit_module, unit_dir, unit_structure = load_extractor_module(OMEGA, MODULES_DIR, model_type='unit_extractor')

carry_structure = _parse_structure(carry_structure)
unit_structure = _parse_structure(unit_structure)

carry_structure = _make_hashable(carry_structure)
unit_structure = _make_hashable(unit_structure)

# Search for number_size as the number of digits of the biggest number in all_pairs
max_number = max(max(a, b) for a, b in all_pairs)
number_size = len(str(max_number))

# Generate initial test dataset
x_val, y_val = generate_test_dataset(all_pairs, number_size=number_size)
x_test, y_test = generate_test_dataset(test_pairs, number_size=number_size)

# --- Save Config File ---
config_path = os.path.join(SAVE_DIR, "config.txt")
with open(config_path, "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Module Name: decision_module\n")
    f.write(f"Parameter Initialization Type (Wise initialization or Random initialization): {PARAM_TYPE}\n")
    f.write(f"Noise Factor for Initialization Parameters (Epsilon): {EPSILON}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Parameters File: {PARAMS_FILE if PARAMS_FILE else 'New parameters generated'}\n")
    f.write(f"Weber fraction (Omega): {OMEGA}\n")
    f.write(f"Unit Extractor imported: {unit_dir}\n")
    f.write(f"Carry Extractor imported: {carry_dir}\n")
    f.write(f"Distribution used for the training set: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"Show Every N Epochs: {SHOW_EVERY_N_EPOCHS}\n")
    f.write(f"Checkpoint Every: {CHECKPOINT_EVERY}\n")
    f.write(f"Training Pairs: {len(train_pairs)}\n")
    f.write(f"Test Pairs: {len(test_pairs)}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Initialize Model Parameters ---
if PARAMS_FILE is not None:
    PARAMS_FILE = os.path.join(PARAMS_DIR, PARAMS_FILE)
    params = load_initial_params(PARAMS_FILE)
else:
    params = initialize_decision_params(PARAMS_DIR, epsilon=EPSILON, param_type=PARAM_TYPE, model_type=MODEL_TYPE, timestamp=timestamp, number_size=number_size)

# Select model function
if MODEL_TYPE == "vector":
    model_fn = decision_model_vector
elif MODEL_TYPE == "argmax":
    model_fn = decision_model_argmax
else:
    raise ValueError("Invalid model type. Choose 'argmax' or 'vector'.")

# --- Training Loop (prepare log and pre-training checkpoint 0) ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True

# Pre-training evaluation: save metrics and a checkpoint for epoch 0
try:
    pred_count, pred_count_test, loss, tests = evaluate_module(
        params, x_val, y_val, unit_module, carry_module, test_pairs, model_fn=model_fn,
        unit_structure=unit_structure, carry_structure=carry_structure,
        carry_set=carry_tuple, small_set=small_tuple, large_set=large_tuple
    )
    accuracy = pred_count / len(x_val) if len(x_val) > 0 else 0.0

    # Write epoch 0 row to training log
    pd.DataFrame([{
            "epoch": 0,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "total_correct": pred_count,
            "test_correct": pred_count_test,
            "test_pairs_no_carry_small_total": totals[0],
            "test_pairs_no_carry_small_count": tests[0],
            "test_pairs_no_carry_small_accuracy": 100 * (tests[0] / totals[0]) if totals[0] > 0 else None,
            "test_pairs_no_carry_large_total": totals[1],
            "test_pairs_no_carry_large_count": tests[1],
            "test_pairs_no_carry_large_accuracy": 100 * (tests[1] / totals[1]) if totals[1] > 0 else None,
            "test_pairs_carry_small_total": totals[2],
            "test_pairs_carry_small_count": tests[2],
            "test_pairs_carry_small_accuracy": 100 * (tests[2] / totals[2]) if totals[2] > 0 else None,
            "test_pairs_carry_large_total": totals[3],
            "test_pairs_carry_large_count": tests[3],
            "test_pairs_carry_large_accuracy": 100 * (tests[3] / totals[3]) if totals[3] > 0 else None,
        }]).to_csv(log_path, mode='a', index=False, header=first_write)
    first_write = False

    # Save checkpoint 0 (initial parameters)
    save_results_and_module(None, accuracy, params, SAVE_DIR, checkpoint_number=0)
    print(f"Saved pre-training checkpoint 0 in {SAVE_DIR}")
except Exception as e:
    print(f"Warning: pre-training evaluation or checkpoint save failed: {e}")

threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))

for epoch in range(EPOCHS):
    # Generate training batch with curriculum learning
    x_train, y_train = generate_train_dataset(train_pairs, BATCH_SIZE, OMEGA, distribution=TRAINING_DISTRIBUTION_TYPE)

    # Update parameters
    params = update_params(
        params, x_train, y_train, unit_module, carry_module, LEARNING_RATE, model_fn=model_fn,
        unit_structure=unit_structure, carry_structure=carry_structure
    )
    
    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        # Evaluate on test set
        pred_count, pred_count_test, loss, tests = evaluate_module(
            params, x_val, y_val, unit_module, carry_module, test_pairs, carry_set=carry_tuple, small_set=small_tuple, large_set=large_tuple, model_fn=model_fn,
            unit_structure=unit_structure, carry_structure=carry_structure
        )
        accuracy = pred_count_test / len(test_pairs)

        # Log results
        pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "total_correct": pred_count,
            "test_correct": pred_count_test,
            "test_pairs_no_carry_small_total": totals[0],
            "test_pairs_no_carry_small_count": tests[0],
            "test_pairs_no_carry_small_accuracy": 100 * (tests[0] / totals[0]) if totals[0] > 0 else None,
            "test_pairs_no_carry_large_total": totals[1],
            "test_pairs_no_carry_large_count": tests[1],
            "test_pairs_no_carry_large_accuracy": 100 * (tests[1] / totals[1]) if totals[1] > 0 else None,
            "test_pairs_carry_small_total": totals[2],
            "test_pairs_carry_small_count": tests[2],
            "test_pairs_carry_small_accuracy": 100 * (tests[2] / totals[2]) if totals[2] > 0 else None,
            "test_pairs_carry_large_total": totals[3],
            "test_pairs_carry_large_count": tests[3],
            "test_pairs_carry_large_accuracy": 100 * (tests[3] / totals[3]) if totals[3] > 0 else None,
        }]).to_csv(log_path, mode='a', index=False, header=first_write)
        first_write = False

        #print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
        #      f"Correct: {pred_count}/{len(x_val)}, Test Correct: {pred_count_test}/{len(test_pairs)}")

    # Save periodic checkpoint
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_results_and_module(None, accuracy, params, SAVE_DIR, checkpoint_number=epoch + 1)
    
    # Early stopping check
    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        #print(f"Reached target accuracy of {threshold}! Stopping training.")
        break

# --- Final Evaluation ---
#print("Final evaluation on test set...")
final_pred_count, final_pred_count_test, final_loss, final_preds, targets = evaluate_module(
    params, x_test, y_test, unit_module, carry_module, test_pairs, model_fn=model_fn,
    unit_structure=unit_structure, carry_structure=carry_structure,
    return_predictions=True
)
final_accuracy = final_pred_count_test / len(test_pairs)

results = []
for i in range(len(test_pairs)):
    x1, x2 = test_pairs[i]
    results.append({
        "x1": x1,
        "x2": x2,
        "y (true)": targets[i],
        "y (pred)": final_preds[i],
        "correct": final_preds[i] == targets[i]
    })
df_results = pd.DataFrame(results)

# --- Save Final Model ---
save_results_and_module(df_results, final_accuracy, params, SAVE_DIR)
