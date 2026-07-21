# USE:
#
# nohup python train_all_at_once.py cuenca 2 NEW_STUDY WI vector 0.05 0.05 0.10 15500 100 1000 No decreasing_exponential 0.1 No > logs_train_all_at_once.out 2>&1 &
#
# "All-at-once" curriculum, for comparison against the existing step-by-step
# pipeline (train_extractor_modules.py -> freeze -> train_decision_module.py).
# Here carry_module, unit_module and the decision-layer params are ALL randomly
# initialized and updated together in a single training loop -- nothing is
# pretrained, nothing is frozen.
#
# MODEL_TYPE choices -- see the header WARNING below and
# little_learner/modules/decision_module/model.py for full detail:
#   - "vector": full softmax probability vectors feed the decision layer. Genuinely joint-trainable.
#   - "argmax": hard argmax index feeds the decision layer. carry_module/unit_module get EXACTLY
#     ZERO gradient every step under this all-at-once setup -- they never move from random init,
#     making this NOT a fair "harder/more discrete" comparison against "vector" (see WARNING below).
#   - "straight_through": same hard forward output as "argmax" (so predictions/discreteness match),
#     but gradient flows through a soft expected-index instead of vanishing -- carry_module and
#     unit_module actually train. Use this instead of "argmax" for a legitimate discrete-condition
#     comparison against "vector".
#

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import jax
print(jax.devices())  # should only show CPU

import jax.numpy as jnp
from jax import random as jrandom
import optax
import pandas as pd

from little_learner.modules.extractor_modules.models import ExtractorModel
from little_learner.modules.extractor_modules.utils import (
    create_and_save_initial_params as create_and_save_extractor_params,
)
from little_learner.modules.decision_module.model import decision_model_vector, decision_model_argmax, decision_model_straight_through
from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset, _make_hashable, _parse_structure,
    save_results_and_module, initialize_decision_params,
)
from little_learner.modules.decision_module.train_utils import (
    evaluate_module, compute_loss, generate_train_dataset,
)

# --- Config ---
CLUSTER = str(sys.argv[1]).lower()  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[2])  # Number of digits in the numbers to be added
STUDY_NAME = str(sys.argv[3]).upper()  # Name of the study
PARAM_TYPE = str(sys.argv[4]).upper()  # 'WI' (wise init) or 'RI' (random init) for the decision layer
MODEL_TYPE = str(sys.argv[5]).lower()  # 'vector', 'argmax', or 'straight_through' -- READ THE HEADER NOTE ABOVE before using 'argmax'
EPSILON_DECISION = float(sys.argv[6])  # Noise/init scale for the decision-layer weights
EPSILON_EXTRACTOR = float(sys.argv[7])  # Noise/init scale for carry_module & unit_module weights
OMEGA = float(sys.argv[8])  # Weber fraction applied to the raw digit inputs
EPOCHS = int(sys.argv[9]) if len(sys.argv) > 9 else 5000
BATCH_SIZE = int(sys.argv[10]) if len(sys.argv) > 10 else 100
EPOCH_SIZE = int(sys.argv[11]) if len(sys.argv) > 11 else 1000
FIXED_VARIABILITY = len(sys.argv) > 12 and sys.argv[12].lower() in ['yes', 'true', '1']
TRAINING_DISTRIBUTION_TYPE = str(sys.argv[13]).lower() if len(sys.argv) > 13 else "none"  # decreasing_exponential / balanced / none
ALPHA_CURRICULUM = float(sys.argv[14]) if len(sys.argv) > 14 else 0.1  # only used if TRAINING_DISTRIBUTION_TYPE == decreasing_exponential
EARLY_STOP = len(sys.argv) > 15 and sys.argv[15].lower() in ['yes', 'true', '1']  # stop as soon as target accuracy is hit (Yes), or keep training the full EPOCHS budget (No)

if MODEL_TYPE not in ("vector", "argmax", "straight_through"):
    raise ValueError("Invalid model type. Choose 'argmax', 'vector', or 'straight_through'.")

# --- Training Parameters ---
LEARNING_RATE = 0.003
FINISH_TOLERANCE = 0.0
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 10

# NOTE: matches train_extractor_modules.py's actual hidden-layer sizes for
# unit_extractor/carry_extractor, so this is an apples-to-apples architecture
# comparison against the step-by-step pipeline.
UNIT_STRUCTURE = [128, 64]
CARRY_STRUCTURE = [16]
# Hashable/static versions, required because evaluate_module() is jitted with
# unit_structure/carry_structure declared as static_argnames.
unit_structure_static = _make_hashable(_parse_structure(UNIT_STRUCTURE))
carry_structure_static = _make_hashable(_parse_structure(CARRY_STRUCTURE))

# --- Paths (mirrors train_decision_module.py, new "ALL_AT_ONCE" branch) ---
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
RAW_DIR = f"{MODULES_DIR}/all_at_once/{NUMBER_SIZE}-digit/{STUDY_NAME}"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/{MODEL_TYPE}_version/epsilon_decision_{EPSILON_DECISION:.2f}_epsilon_extractor_{EPSILON_EXTRACTOR:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)


# --- Dataset loading (identical to train_decision_module.py) ---
def _safe_load(name, path):
    try:
        return load_dataset(path)
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

# If test_pairs is empty, fall back to a deterministic holdout from all_pairs
if not test_pairs:
    print(f"[WARN] No stimuli_test_pairs found or empty for number_size={NUMBER_SIZE}. Falling back to validation set from all_pairs.")
    if all_pairs and len(all_pairs) > 0:
        test_pairs = [p for i, p in enumerate(all_pairs) if i % max(1, len(all_pairs) // 100) == 0]
    else:
        raise RuntimeError(f"No data available to create a fallback test set for number_size={NUMBER_SIZE}.")

if NUMBER_SIZE > 2:
    totals = [len(test_no_carry_small), len(test_no_carry_large), len(test_carry_small), len(test_carry_large),
              len(test_no_carry_small) + len(test_carry_small), len(test_no_carry_large) + len(test_carry_large)]
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
    carry_tuple = tuple(sorted(carry_set))
    small_tuple = tuple(sorted(small_set))
    large_tuple = tuple(sorted(large_set))

x_val, y_val = generate_test_dataset(all_pairs, number_size=NUMBER_SIZE)
x_test, y_test = generate_test_dataset(test_pairs, number_size=NUMBER_SIZE)

# --- Fresh, joint initialization: nothing pretrained, nothing frozen ---
carry_model = ExtractorModel(structure=CARRY_STRUCTURE, output_dim=2)
unit_model = ExtractorModel(structure=UNIT_STRUCTURE, output_dim=10)

rng = jrandom.PRNGKey(hash((STUDY_NAME, EPSILON_DECISION, EPSILON_EXTRACTOR)) % (2 ** 31))
rng_carry, rng_unit = jrandom.split(rng, 2)

carry_params_path = os.path.join(PARAMS_DIR, f"initial_params_carry_{timestamp}.json")
unit_params_path = os.path.join(PARAMS_DIR, f"initial_params_unit_{timestamp}.json")
carry_params = create_and_save_extractor_params(carry_model, rng_carry, (1, 2), carry_params_path, epsilon=EPSILON_EXTRACTOR)
unit_params = create_and_save_extractor_params(unit_model, rng_unit, (1, 2), unit_params_path, epsilon=EPSILON_EXTRACTOR)

# Decision-layer params: same WI/RI init the step-by-step pipeline uses.
decision_params = initialize_decision_params(
    PARAMS_DIR, epsilon=EPSILON_DECISION, param_type=PARAM_TYPE,
    model_type=MODEL_TYPE, timestamp=timestamp, number_size=NUMBER_SIZE,
)

all_params = {'decision': decision_params, 'unit': unit_params, 'carry': carry_params}

if MODEL_TYPE == "vector":
    model_fn = decision_model_vector
elif MODEL_TYPE == "straight_through":
    model_fn = decision_model_straight_through
    print("=" * 70)
    print("[INFO] MODEL_TYPE='straight_through': forward-pass predictions are")
    print("numerically identical to 'argmax' (hard index per digit pair), but")
    print("gradients flow through a soft expected-index instead of vanishing,")
    print("so carry_module and unit_module DO move from their random init this")
    print("run -- unlike 'argmax'. carry_param_change_norm/unit_param_change_norm")
    print("are logged every epoch below so this is directly checkable.")
    print("=" * 70)
else:
    model_fn = decision_model_argmax
    print("=" * 70)
    print("[WARNING] MODEL_TYPE='argmax': carry_module and unit_module receive")
    print("EXACTLY ZERO gradient every step (jnp.argmax has zero gradient a.e.,")
    print("and its integer output feeds straight into the decision layer's")
    print("linear readout). They will NOT move from their random initialization")
    print("at ANY point during this run -- only decision_params ('dense_i') are")
    print("actually being trained. This is equivalent to 'decision layer trained")
    print("on top of frozen-at-init random carry/unit features', not a genuine")
    print("joint/all-at-once condition for those two modules. See the header")
    print("comment for the full explanation. carry_param_change_norm and")
    print("unit_param_change_norm are logged every epoch below (they should sit")
    print("at ~0.0 throughout) so this is directly checkable, not just asserted.")
    print("Use MODEL_TYPE='straight_through' instead for a genuine discrete-and-")
    print("trainable all-at-once condition.")
    print("=" * 70)

# Snapshot the random init so we can verify, every epoch, whether carry_module/
# unit_module actually moved -- decisive under 'vector' (they should), and a
# concrete, checkable confirmation of the zero-gradient claim under 'argmax'
# (they shouldn't, at all).
_initial_carry_params = all_params['carry']
_initial_unit_params = all_params['unit']


def _param_change_norm(current, initial):
    diffs = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b) ** 2), current, initial)
    return float(jnp.sqrt(sum(jax.tree_util.tree_leaves(diffs))))


tx = optax.adam(LEARNING_RATE)
opt_state = tx.init(all_params)


# --- Joint loss/update step ---
# compute_loss() below is imported unchanged from decision_module.train_utils
# -- it's the exact same MSE-over-digits loss the step-by-step pipeline uses.
# update_params()/optimizer_update_params() in that same module can't be
# reused as-is here: both call jax.grad(compute_loss) w.r.t. the decision
# params ONLY, treating unit_module/carry_module as fixed extra arguments --
# correct for the frozen-extractor step-by-step case, but exactly what we
# need to avoid for "all-at-once". The wrapper below bundles decision/unit/
# carry params into a single pytree so autodiff reaches all three; it does
# not change what loss is computed, only what it's differentiated against.
def joint_loss_fn(all_params, x, y):
    return compute_loss(
        all_params['decision'], x, y, all_params['unit'], all_params['carry'],
        unit_structure=unit_structure_static, carry_structure=carry_structure_static, model_fn=model_fn,
    )


@jax.jit
def joint_train_step(all_params, opt_state, x, y):
    loss, grads = jax.value_and_grad(joint_loss_fn)(all_params, x, y)
    updates, opt_state = tx.update(grads, opt_state, all_params)
    all_params = optax.apply_updates(all_params, updates)
    return all_params, opt_state, loss


def run_evaluation(all_params, eval_x, eval_y):
    return evaluate_module(
        all_params['decision'], eval_x, eval_y, all_params['unit'], all_params['carry'], test_pairs,
        model_fn=model_fn, unit_structure=unit_structure_static, carry_structure=carry_structure_static,
        carry_set=carry_tuple, small_set=small_tuple, large_set=large_tuple,
    )


# --- Config file ---
with open(os.path.join(SAVE_DIR, "config.txt"), "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Curriculum: all_at_once\n")
    f.write(f"Study Name: {STUDY_NAME}\n")
    f.write(f"Model Type (argmax, vector, or straight_through): {MODEL_TYPE}"
            + (" -- WARNING: carry_module/unit_module get exactly zero gradient under argmax, "
               "see header comment; they never move from random init in this run.\n" if MODEL_TYPE == "argmax"
               else " -- forward pass identical to argmax, but gradient flows through a soft "
               "expected-index so carry_module/unit_module DO train.\n" if MODEL_TYPE == "straight_through"
               else "\n"))
    f.write(f"Number Size: {NUMBER_SIZE}\n")
    f.write(f"Decision Parameter Initialization Type (Wise/Random): {PARAM_TYPE}\n")
    f.write(f"Epsilon (decision-layer init scale): {EPSILON_DECISION}\n")
    f.write(f"Epsilon (extractor modules init scale): {EPSILON_EXTRACTOR}\n")
    f.write(f"Weber fraction (Omega): {OMEGA}\n")
    f.write(f"Fixed Variability: {'Yes' if FIXED_VARIABILITY else 'No'}\n")
    f.write(f"Distribution used for the training set: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Alpha for curriculum learning: {ALPHA_CURRICULUM}\n")
    f.write(f"Early Stop: {'Yes' if EARLY_STOP else 'No'}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epoch Size: {EPOCH_SIZE}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"Show Every N Epochs: {SHOW_EVERY_N_EPOCHS}\n")
    f.write(f"Checkpoint Every: {CHECKPOINT_EVERY}\n")
    f.write(f"Unit Structure: {UNIT_STRUCTURE}\n")
    f.write(f"Carry Structure: {CARRY_STRUCTURE}\n")
    f.write(f"Training Pairs: {len(train_pairs)}\n")
    f.write(f"Test Pairs: {len(test_pairs)}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Training loop (mirrors train_decision_module.py's loop/logging) ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True
threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))
batches_per_epoch = max(1, EPOCH_SIZE // BATCH_SIZE)


def _log_row(epoch, loss, accuracy, pred_count, pred_count_test, tests, carry_change, unit_change):
    return {
        "epoch": epoch,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "total_correct": pred_count,
        "test_correct": pred_count_test,
        "carry_param_change_norm": carry_change,  # L2 distance from random init -- should be ~0 under argmax, growing under vector
        "unit_param_change_norm": unit_change,
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


# Pre-training evaluation: checkpoint 0
try:
    eval_x, eval_y = (x_test, y_test) if NUMBER_SIZE > 2 else (x_val, y_val)
    pred_count, pred_count_test, loss, tests = run_evaluation(all_params, eval_x, eval_y)
    accuracy = pred_count / len(eval_x) if len(eval_x) > 0 else 0.0
    carry_change = _param_change_norm(all_params['carry'], _initial_carry_params)
    unit_change = _param_change_norm(all_params['unit'], _initial_unit_params)
    pd.DataFrame([_log_row(0, loss, accuracy, pred_count, pred_count_test, tests, carry_change, unit_change)]).to_csv(
        log_path, mode='a', index=False, header=first_write)
    first_write = False
    save_results_and_module(None, accuracy, all_params, SAVE_DIR, checkpoint_number=0)
    print(f"Saved pre-training checkpoint 0 in {SAVE_DIR}")
except Exception as e:
    print(f"Warning: pre-training evaluation or checkpoint save failed: {e}")

for epoch in range(EPOCHS):
    for batch_idx in range(batches_per_epoch):
        seed = epoch * batches_per_epoch + batch_idx
        x_train, y_train = generate_train_dataset(
            train_pairs, BATCH_SIZE, OMEGA, distribution=TRAINING_DISTRIBUTION_TYPE,
            alpha=ALPHA_CURRICULUM, number_size=NUMBER_SIZE, seed=seed, fixed_variability=FIXED_VARIABILITY,
        )
        all_params, opt_state, train_loss = joint_train_step(all_params, opt_state, x_train, y_train)

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        try:
            eval_x, eval_y = (x_test, y_test) if NUMBER_SIZE > 2 else (x_val, y_val)
            pred_count, pred_count_test, loss, tests = run_evaluation(all_params, eval_x, eval_y)
        except Exception as e:
            print(f"[ERROR] evaluate_module failed at epoch {epoch + 1}: {e}")
            pred_count, pred_count_test, loss, tests = 0, 0, float('nan'), [0, 0, 0, 0]
        accuracy = (pred_count_test / len(test_pairs)) if len(test_pairs) > 0 else 0.0
        carry_change = _param_change_norm(all_params['carry'], _initial_carry_params)
        unit_change = _param_change_norm(all_params['unit'], _initial_unit_params)

        pd.DataFrame([_log_row(epoch + 1, loss, accuracy, pred_count, pred_count_test, tests, carry_change, unit_change)]).to_csv(
            log_path, mode='a', index=False, header=first_write)
        first_write = False
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        try:
            save_results_and_module(None, accuracy, all_params, SAVE_DIR, checkpoint_number=epoch + 1)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint {epoch + 1}: {e}")

    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        print(f"All-at-once model reached target accuracy at epoch {epoch + 1}.")
        if EARLY_STOP:
            last_epoch = epoch + 1
            last_metrics = _log_row(None, loss, accuracy, pred_count, pred_count_test, tests, carry_change, unit_change)
            try:
                save_results_and_module(None, accuracy, all_params, SAVE_DIR, checkpoint_number=last_epoch)
                print(f"Saved final checkpoint {last_epoch} after achieving target accuracy in {SAVE_DIR}")
            except Exception as e:
                print(f"[ERROR] Failed to save final checkpoint {last_epoch}: {e}")
            for fill_epoch in range(last_epoch + SHOW_EVERY_N_EPOCHS - (last_epoch % SHOW_EVERY_N_EPOCHS), EPOCHS + 1, SHOW_EVERY_N_EPOCHS):
                last_metrics["epoch"] = fill_epoch
                pd.DataFrame([last_metrics]).to_csv(log_path, mode='a', index=False, header=False)
            break
        # else: EARLY_STOP is False -- keep training the full EPOCHS budget,
        # same as train_extractor_modules.py's default behaviour.

# --- Final Evaluation ---
try:
    final_pred_count, final_pred_count_test, final_loss, final_preds, targets = evaluate_module(
        all_params['decision'], x_test, y_test, all_params['unit'], all_params['carry'], test_pairs,
        model_fn=model_fn, unit_structure=unit_structure_static, carry_structure=carry_structure_static,
        return_predictions=True,
    )
    final_accuracy = (final_pred_count_test / len(test_pairs)) if len(test_pairs) > 0 else 0.0
except Exception as e:
    print(f"[ERROR] Final evaluation failed: {e}")
    final_preds, targets, final_accuracy = [], [], 0.0

results = []
for i in range(len(test_pairs)):
    x1, x2 = test_pairs[i]
    results.append({
        "x1": x1, "x2": x2,
        "y (true)": targets[i], "y (pred)": final_preds[i],
        "correct": final_preds[i] == targets[i],
    })
df_results = pd.DataFrame(results)

# --- Save Final Model ---
save_results_and_module(df_results, final_accuracy, all_params, SAVE_DIR)
print(f"All-at-once training complete. Saved to {SAVE_DIR}")
print("Compare training_log.csv here against the step-by-step pipeline's logs at the same "
      "OMEGA/EPSILON/NUMBER_SIZE/TRAINING_DISTRIBUTION_TYPE to evaluate the two curricula.")