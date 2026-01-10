# USE: nohup python train_decision_module.py 2 SEVENTH_STUDY WI argmax 0.10 0.05 5000 100 1000 No > logs_train_decision.out 2>&1 &

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
# print(jax.devices())  # should only show CPU

import jax.numpy as jnp
import pandas as pd
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset,
    _make_hashable, _parse_structure,
    save_results_and_module, load_extractor_module, initialize_decision_params, load_initial_params
)
from little_learner.modules.decision_module.train_utils import (
    evaluate_module, update_params, generate_train_dataset,
    debug_decision_example
)
from little_learner.modules.decision_module.model import decision_model_argmax, decision_model_vector

# --- Config ---
CLUSTER = str(sys.argv[1]).lower()  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[2])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[3]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[4]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[5]).lower() # "argmax" for argmax outputs, "vector" for probability vector outputs
EPSILON = float(sys.argv[6])  # Noise factor for parameter initialization
OMEGA = float(sys.argv[7])  # Omega value for loading pre-trained modules
EPOCHS = int(sys.argv[8]) if len(sys.argv) > 8 else 5000  # Number of training epochs
BATCH_SIZE = int(sys.argv[9]) if len(sys.argv) > 9 else 100  # Batch size for training
EPOCH_SIZE = int(sys.argv[10]) if len(sys.argv) > 10 else 1000  # Number of examples per epoch
FIXED_VARIABILITY = len(sys.argv) > 11 and sys.argv[11].lower() in ['yes', 'true', '1']  # Fixed variability flag (Yes/No)
TRAINING_DISTRIBUTION_TYPE = str(sys.argv[12]).lower() if len(sys.argv) > 12 else "none"  # Use curriculum learning for training (decreasing_exponential or balanced)
ALPHA_CURRICULUM = float(sys.argv[13]) if len(sys.argv) > 13 else 0.1  # Only used if TRAINING_DISTRIBUTION_TYPE is "decreasing_exponential"

# --- Training Parameters ---
LEARNING_RATE = 0.003
FINISH_TOLERANCE = 0.0  # Tolerance for stopping training when accuracy reaches 1.0
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 10
PARAMS_FILE = None  # Set to None to create new params, or provide a path to load existing params

# --- General paths ---
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

DATASET_DIR = f"{CODE_DIR}/datasets/{NUMBER_SIZE}-digit"
MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
RAW_DIR = f"{MODULES_DIR}/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/{MODEL_TYPE}_version/epsilon_{EPSILON:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Load datasets and extractors (with debug)
def _safe_load(name, path):
    try:
        data = load_dataset(path)
        return data
    except Exception as e:
        print(f"[ERROR] Could not load {name} from {path}: {e}")
        return []

all_pairs = _safe_load('all_pairs', os.path.join(DATASET_DIR, "all_valid_additions.txt"))
train_pairs = _safe_load('train_pairs', os.path.join(DATASET_DIR, "train_pairs_not_in_stimuli.txt"))
test_pairs = _safe_load('test_pairs', os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
carry = _safe_load('carry', os.path.join(DATASET_DIR, "carry_additions.txt"))
small = _safe_load('small', os.path.join(DATASET_DIR, "small_additions.txt"))
large = _safe_load('large', os.path.join(DATASET_DIR, "large_additions.txt"))

# For NUMBER_SIZE > 2, load precomputed test categories
if NUMBER_SIZE > 2:
    def load_category(filename):
        path = os.path.join(DATASET_DIR, filename)
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                return eval(content) if content else []
        except Exception as e:
            print(f"[ERROR] Could not load {filename}: {e}")
            return []
    test_carry_small = load_category("test_carry_small.txt")
    test_carry_large = load_category("test_carry_large.txt")
    test_no_carry_small = load_category("test_no_carry_small.txt")
    test_no_carry_large = load_category("test_no_carry_large.txt")

# --- Sanity checks & debug info for datasets (helpful for N!=2 where stimuli may be empty)
def _dbg_len(name, data):
    try:
        ln = len(data)
    except Exception:
        ln = 0
    return ln

_dbg_len('all_pairs', all_pairs)
_dbg_len('train_pairs', train_pairs)
_dbg_len('test_pairs (stimuli)', test_pairs)
_dbg_len('carry', carry)
_dbg_len('small', small)
_dbg_len('large', large)
if NUMBER_SIZE > 2:
    _dbg_len('test_carry_small', test_carry_small)
    _dbg_len('test_carry_large', test_carry_large)
    _dbg_len('test_no_carry_small', test_no_carry_small)
    _dbg_len('test_no_carry_large', test_no_carry_large)

# If test_pairs is empty (common for number sizes without pre-defined stimuli),
# fall back to using a validation split from all_pairs so training/evaluation continues
if not test_pairs:
    print(f"[WARN] No stimuli_test_pairs found or empty for number_size={NUMBER_SIZE}. Falling back to validation set from all_pairs.")
    # create a small deterministic holdout from all_pairs (every 10th example) as fallback
    if all_pairs and len(all_pairs) > 0:
        test_pairs = [p for i, p in enumerate(all_pairs) if i % max(1, len(all_pairs)//100) == 0]
    else:
        raise RuntimeError(f"No data available to create a fallback test set for number_size={NUMBER_SIZE}.")

if NUMBER_SIZE > 2:
    totals = [len(test_no_carry_small), len(test_no_carry_large), len(test_carry_small), len(test_carry_large), len(test_no_carry_small)+len(test_carry_small), len(test_no_carry_large)+len(test_carry_large)]
    # For JAX static arguments, also create tuple versions for >2 digits
    carry_tuple = tuple(sorted(test_carry_small + test_carry_large))
    small_tuple = tuple(sorted(test_no_carry_small + test_carry_small))
    large_tuple = tuple(sorted(test_no_carry_large + test_carry_large))
else:
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

# --- Modules initialization ---
try:
    carry_module, carry_dir, carry_structure = load_extractor_module(OMEGA, MODULES_DIR, model_type='carry_extractor', study_name=STUDY_NAME)
except Exception as e:
    print(f"[ERROR] Failed to load carry extractor (omega={OMEGA}): {e}")
    raise

try:
    unit_module, unit_dir, unit_structure = load_extractor_module(OMEGA, MODULES_DIR, model_type='unit_extractor', study_name=STUDY_NAME)
except Exception as e:
    print(f"[ERROR] Failed to load unit extractor (omega={OMEGA}): {e}")
    raise

carry_structure = _parse_structure(carry_structure)
unit_structure = _parse_structure(unit_structure)

carry_structure = _make_hashable(carry_structure)
unit_structure = _make_hashable(unit_structure)

# Generate initial test dataset
x_val, y_val = generate_test_dataset(all_pairs, number_size=NUMBER_SIZE)
x_test, y_test = generate_test_dataset(test_pairs, number_size=NUMBER_SIZE)

# --- Save Config File ---
config_path = os.path.join(SAVE_DIR, "config.txt")
with open(config_path, "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Module Name: decision_module\n")
    f.write(f"Study Name: {STUDY_NAME}\n")
    f.write(f"Model Type (Argmax or Vector): {MODEL_TYPE}\n")
    f.write(f"Number Size: {NUMBER_SIZE}\n")
    f.write(f"Parameter Initialization Type (Wise initialization or Random initialization): {PARAM_TYPE}\n")
    f.write(f"Noise Factor for Initialization Parameters (Epsilon): {EPSILON}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epoch Size: {EPOCH_SIZE}\n")
    f.write(f"Parameters File: {PARAMS_FILE if PARAMS_FILE else 'New parameters generated'}\n")
    f.write(f"Weber fraction (Omega): {OMEGA}\n")
    f.write(f"Fixed Variability: {'Yes' if FIXED_VARIABILITY else 'No'}\n")
    f.write(f"Unit Extractor imported: {unit_dir}\n")
    f.write(f"Carry Extractor imported: {carry_dir}\n")
    f.write(f"Distribution used for the training set: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Alpha for curriculum learning: {ALPHA_CURRICULUM}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"Show Every N Epochs: {SHOW_EVERY_N_EPOCHS}\n")
    f.write(f"Checkpoint Every: {CHECKPOINT_EVERY}\n")
    f.write(f"Training Pairs: {len(train_pairs)}\n")
    f.write(f"Test Pairs: {len(test_pairs)}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Initialize Model Parameters ---
if PARAMS_FILE is not None:
    PARAMS_FILE = os.path.join(PARAMS_DIR, PARAMS_FILE)
    try:
        params = load_initial_params(PARAMS_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to load initial params from {PARAMS_FILE}: {e}")
        raise
else:
    try:
        params = initialize_decision_params(PARAMS_DIR, epsilon=EPSILON, param_type=PARAM_TYPE, model_type=MODEL_TYPE, timestamp=timestamp, number_size=NUMBER_SIZE)
    except Exception as e:
        print(f"[ERROR] Failed to initialize params: {e}")
        raise

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
    if NUMBER_SIZE > 2:
        eval_x, eval_y = x_test, y_test
    else:
        eval_x, eval_y = x_val, y_val
    pred_count, pred_count_test, loss, tests = evaluate_module(
        params, eval_x, eval_y, unit_module, carry_module, test_pairs, model_fn=model_fn,
        unit_structure=unit_structure, carry_structure=carry_structure,
        carry_set=carry_tuple, small_set=small_tuple, large_set=large_tuple
    )
    accuracy = pred_count / len(eval_x) if len(eval_x) > 0 else 0.0

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

# Calculate batches per epoch based on EPOCH_SIZE
batches_per_epoch = max(1, EPOCH_SIZE // BATCH_SIZE)

for epoch in range(EPOCHS):
    # Train multiple batches per epoch
    for batch_idx in range(batches_per_epoch):
        # Generate training batch with curriculum learning
        x_train, y_train = generate_train_dataset(train_pairs, BATCH_SIZE, OMEGA, distribution=TRAINING_DISTRIBUTION_TYPE, alpha=ALPHA_CURRICULUM, number_size=NUMBER_SIZE, seed=epoch * batches_per_epoch + batch_idx, fixed_variability=FIXED_VARIABILITY)
        
        # Update parameters
        try:
            params = update_params(
                params, x_train, y_train, unit_module, carry_module, LEARNING_RATE, model_fn=model_fn,
                unit_structure=unit_structure, carry_structure=carry_structure
            )
        except Exception as e:
            print(f"[ERROR] update_params failed at epoch {epoch+1}, batch {batch_idx+1}: {e}")
            raise
    
    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        # Evaluate on test set
        try:
            if NUMBER_SIZE > 2:
                eval_x, eval_y = x_test, y_test
            else:
                eval_x, eval_y = x_val, y_val
            pred_count, pred_count_test, loss, tests = evaluate_module(
                params, eval_x, eval_y, unit_module, carry_module, test_pairs, model_fn=model_fn,
                unit_structure=unit_structure, carry_structure=carry_structure,
                carry_set=carry_tuple, small_set=small_tuple, large_set=large_tuple
            )
        except Exception as e:
            print(f"[ERROR] evaluate_module failed at epoch {epoch+1}: {e}")
            # continue but set safe defaults
            pred_count = 0
            pred_count_test = 0
            loss = float('nan')
            tests = [0, 0, 0, 0]
        accuracy = (pred_count_test / len(test_pairs)) if (len(test_pairs) > 0) else 0.0

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

    # Save periodic checkpoint
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        try:
            save_results_and_module(None, accuracy, params, SAVE_DIR, checkpoint_number=epoch + 1)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint {epoch+1}: {e}")
    
    # Early stopping check
    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        # Save last epoch info for filling remaining epochs
        last_epoch = epoch + 1
        last_metrics = {
            "epoch": None,  # will be set in loop
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
        }
        # Fill remaining epochs with last metrics
        for fill_epoch in range(last_epoch + SHOW_EVERY_N_EPOCHS - (last_epoch % SHOW_EVERY_N_EPOCHS), EPOCHS + 1, SHOW_EVERY_N_EPOCHS):
            last_metrics["epoch"] = fill_epoch
            pd.DataFrame([last_metrics]).to_csv(log_path, mode='a', index=False, header=False)
        break
        

# --- Final Evaluation ---
try:
    final_pred_count, final_pred_count_test, final_loss, final_preds, targets = evaluate_module(
        params, x_test, y_test, unit_module, carry_module, test_pairs, model_fn=model_fn,
        unit_structure=unit_structure, carry_structure=carry_structure,
        return_predictions=True
    )
    final_accuracy = (final_pred_count_test / len(test_pairs)) if (len(test_pairs) > 0) else 0.0
except Exception as e:
    print(f"[ERROR] Final evaluation failed: {e}")
    final_preds = []
    targets = []
    final_accuracy = 0.0

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
print('Training complete.')