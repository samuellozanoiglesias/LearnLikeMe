# USE: 
# 
# nohup python train_all_at_once_perceptual.py cuenca 2 NEW_STUDY RI 0.05 0.05 0.10 5000 100 1000 No Decreasing_exponential 0.1 No > logs_train_all_at_once_perceptual.out 2>&1 &
#
# Fully joint perceptual curriculum: DigitCNN + carry_module + unit_module +
# the decision layer are ALL randomly initialized and updated together in a
# single training loop, straight from raw MNIST pixels to the N-digit sum --
# nothing is pretrained, nothing is frozen. This is the perceptual analogue
# of train_all_at_once.py, extended one stage further upstream (that script
# already joint-trains carry/unit/decision from symbolic digit values; this
# one ALSO joint-trains the CNN that reads the digit off an image).
#
# ============================ PLAUSIBILITY ================================
# You asked me to build this and flag problems rather than refuse it, so:
# it will very likely run and slowly learn *something*, but there are real,
# structural reasons to expect it to be much harder to train well than
# train_all_at_once.py, and to be a different (not strictly "harder version
# of the same") experiment scientifically. In rough order of severity:
#
# 1. COLD-START / SYMMETRY-BREAKING: at initialization the CNN is untrained,
#    so perceptual_magnitude()'s softmax-weighted expected value is close to
#    uniform over 0-9 for EVERY image, regardless of the digit shown (~4.5
#    for all of them). Unlike the symbolic all-at-once script, where the raw
#    input already numerically encodes roughly the right digit (just
#    perturbed by Weber noise), here the initial representation carries
#    ~zero information about which digit is shown. Every digit position
#    looks almost identical to the downstream extractors at step 0, so the
#    early gradient signal telling the CNN "which way to move" is extremely
#    weak. This is a much harder credit-assignment problem, and there's a
#    real risk of the run stalling in that degenerate regime rather than
#    breaking symmetry, especially at the default learning rate/epoch budget.
# 2. FOUR-WAY CONFOUNDED FAILURE MODES: if joint accuracy is poor, it's hard
#    to tell whether the CNN, the extractors, or the decision layer is the
#    bottleneck -- the same interpretability problem train_all_at_once.py's
#    header already flags for 3 modules, now with a 4th (and the most
#    expensive-to-diagnose) one added.
# 3. COMPUTE COST: CNN forward/backward passes over image batches are far
#    more expensive per example than the small dense nets everything else
#    uses, and point 1 likely means MANY more steps are needed to converge
#    (if it converges) than the symbolic all-at-once baseline -- so this is
#    a substantially more expensive experiment per unit of learning progress.
# 4. OMEGA'S MEANING GETS MUDDIER: in the frozen-CNN scripts, OMEGA is a
#    clean, stable "extra noise on top of a fixed estimate" -- comparable
#    across checkpoints. Here it's extra noise on top of an ESTIMATE FROM A
#    STILL-LEARNING CNN, whose own error characteristics change every step.
#    The same OMEGA value doesn't correspond to a stable, comparable
#    perceptual-noise condition across a run, weakening the controlled-
#    comparison rationale Weber-fraction conditions were meant to provide.
# 5. ONE OPTIMIZER/LR FOR STRUCTURALLY DIFFERENT MODULES: conv filters and
#    small dense heads usually want different learning rates. A single
#    shared Adam+LR for all four is a coarser compromise here than in the
#    symbolic all-at-once case, where at least all trainable pieces were
#    similarly-shaped dense nets.
# 6. FRAMING: if the point of the staged curriculum is to test whether
#    human-like STAGED learning matters (this is the "LearnLikeMe" project),
#    a fully-joint CNN+everything run doesn't cleanly isolate that question
#    anymore -- a poor result here says "this is a much harder optimization
#    problem without ANY staging, including at the perceptual level," not
#    cleanly "staging helps arithmetic." Worth keeping in mind when
#    interpreting results, not just when running the script.
#
# Mitigations you could layer on top of this script if a plain run struggles
# (not built in, so the baseline stays a clean "fully joint, no help" run):
# briefly pretrain the CNN on plain digit classification before switching to
# the joint sum loss; give the CNN a lower LR than the dense heads; or add an
# auxiliary digit-classification loss term alongside the sum loss.
# ============================================================================
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
import numpy as np
import optax
import pandas as pd

from little_learner.modules.perceptual_module.digit_recognizer import DigitCNN, perceptual_magnitude
from little_learner.modules.perceptual_module.mnist_utils import load_mnist, index_by_digit
from little_learner.modules.extractor_modules.models import ExtractorModel
from little_learner.modules.extractor_modules.utils import (
    create_and_save_initial_params as create_and_save_extractor_params,
)
from little_learner.modules.decision_module.model import decision_model_vector
from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset, _make_hashable, _parse_structure,
    save_results_and_module, initialize_decision_params,
)
from little_learner.modules.decision_module.train_utils import compute_loss, generate_train_dataset

# --- Config (same CLI shape as train_all_at_once.py) ---
CLUSTER = str(sys.argv[1]).lower()
NUMBER_SIZE = int(sys.argv[2])
STUDY_NAME = str(sys.argv[3]).upper()
PARAM_TYPE = str(sys.argv[4]).upper()  # 'WI' or 'RI', decision layer only
EPSILON_DECISION = float(sys.argv[5])
EPSILON_EXTRACTOR = float(sys.argv[6])  # carry_module/unit_module init noise; CNN uses Flax's default init instead
OMEGA = float(sys.argv[7])  # extra Weber noise on top of the (evolving) CNN estimate
EPOCHS = int(sys.argv[8]) if len(sys.argv) > 8 else 5000
BATCH_SIZE = int(sys.argv[9]) if len(sys.argv) > 9 else 100
EPOCH_SIZE = int(sys.argv[10]) if len(sys.argv) > 10 else 1000
FIXED_VARIABILITY = len(sys.argv) > 11 and sys.argv[11].lower() in ['yes', 'true', '1']
TRAINING_DISTRIBUTION_TYPE = str(sys.argv[12]).lower() if len(sys.argv) > 12 else "none"
ALPHA_CURRICULUM = float(sys.argv[13]) if len(sys.argv) > 13 else 0.1
EARLY_STOP = len(sys.argv) > 14 and sys.argv[14].lower() in ['yes', 'true', '1']

LEARNING_RATE = 0.003
FINISH_TOLERANCE = 0.0
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 10
MODEL_TYPE = "vector"  # argmax has ~zero gradient -- would leave cnn/carry/unit at random init, see train_all_at_once.py's header note (same reasoning, now applies to the CNN too)

CARRY_STRUCTURE = [16]
UNIT_STRUCTURE = [128, 64]
carry_structure_static = _make_hashable(_parse_structure(CARRY_STRUCTURE))
unit_structure_static = _make_hashable(_parse_structure(UNIT_STRUCTURE))

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

DATASET_DIR = f"{CODE_DIR}/datasets/{NUMBER_SIZE}-digit"
MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
RAW_DIR = f"{MODULES_DIR}/all_at_once_perceptual/{NUMBER_SIZE}-digit/{STUDY_NAME}"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/epsilon_decision_{EPSILON_DECISION:.2f}_epsilon_extractor_{EPSILON_EXTRACTOR:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)


# --- Dataset loading (identical to train_decision_module_perceptual.py) ---
def _safe_load(name, path):
    try:
        return load_dataset(path)
    except Exception as e:
        print(f"[ERROR] Could not load {name} from {path}: {e}")
        return []


train_pairs = _safe_load('train_pairs', os.path.join(DATASET_DIR, "train_pairs_not_in_stimuli.txt"))
test_pairs = _safe_load('test_pairs', os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
if not test_pairs:
    raise RuntimeError(f"No pairs found in {DATASET_DIR}/stimuli_test_pairs.txt. "
                        "Run generate_arithmetic_datasets.py first.")


def load_category(filename):
    path = os.path.join(DATASET_DIR, filename)
    try:
        with open(path, "r") as f:
            content = f.read().strip()
            return eval(content) if content else []
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")
        return []


test_carry_small = set(load_category("test_carry_small.txt"))
test_carry_large = set(load_category("test_carry_large.txt"))
test_no_carry_small = set(load_category("test_no_carry_small.txt"))
test_no_carry_large = set(load_category("test_no_carry_large.txt"))

carry_mask_test = np.array([p in test_carry_small or p in test_carry_large for p in test_pairs])
small_mask_test = np.array([p in test_no_carry_small or p in test_carry_small for p in test_pairs])
large_mask_test = np.array([p in test_no_carry_large or p in test_carry_large for p in test_pairs])
totals = [len(test_no_carry_small), len(test_no_carry_large), len(test_carry_small), len(test_carry_large)]

_, y_test = generate_test_dataset(test_pairs, number_size=NUMBER_SIZE)

# --- MNIST images: TRAIN split for training, TEST split for the fixed eval set ---
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist()
pools_train = index_by_digit(x_train_mnist, y_train_mnist)
pools_test = index_by_digit(x_test_mnist, y_test_mnist)


def images_for_digits(digit_values, pools, seed):
    digit_values = np.asarray(digit_values).astype(int)
    batch_size, n_positions = digit_values.shape
    rng = np.random.default_rng(seed)
    image_shape = pools[0].shape[1:]
    images = np.empty((batch_size, n_positions) + image_shape, dtype=pools[0].dtype)
    flat_digits = digit_values.reshape(-1)
    flat_images = images.reshape(-1, *image_shape)
    for d in range(10):
        mask = flat_digits == d
        n_needed = int(mask.sum())
        if n_needed == 0:
            continue
        idxs = rng.integers(0, len(pools[d]), size=n_needed)
        flat_images[mask] = pools[d][idxs]
    return jnp.array(images)


def images_to_magnitudes(cnn_p, images, rng, omega):
    n_positions = images.shape[1]
    rngs = jax.random.split(rng, n_positions)
    mags = []
    for pos in range(n_positions):
        mag, _, _ = perceptual_magnitude(cnn_p, images[:, pos], cnn, omega=omega, rng=rngs[pos])
        mags.append(mag)
    return jnp.stack(mags, axis=1)


# --- Fresh, joint initialization: nothing pretrained, nothing frozen ---
cnn = DigitCNN(output_dim=10)
carry_model = ExtractorModel(structure=CARRY_STRUCTURE, output_dim=2)
unit_model = ExtractorModel(structure=UNIT_STRUCTURE, output_dim=10)

rng = jrandom.PRNGKey(hash((STUDY_NAME, EPSILON_DECISION, EPSILON_EXTRACTOR)) % (2 ** 31))
rng_cnn, rng_carry, rng_unit = jrandom.split(rng, 3)

# CNN gets Flax's default init (matches train_digit_recognizer.py -- there's
# no natural epsilon-scaled-Gaussian equivalent for conv filters the way
# there is for the small dense extractor/decision layers).
cnn_params = cnn.init(rng_cnn, pools_train[0][:1])['params']

carry_params_path = os.path.join(PARAMS_DIR, f"initial_params_carry_{timestamp}.json")
unit_params_path = os.path.join(PARAMS_DIR, f"initial_params_unit_{timestamp}.json")
carry_params = create_and_save_extractor_params(carry_model, rng_carry, (1, 2), carry_params_path, epsilon=EPSILON_EXTRACTOR)
unit_params = create_and_save_extractor_params(unit_model, rng_unit, (1, 2), unit_params_path, epsilon=EPSILON_EXTRACTOR)

decision_params = initialize_decision_params(
    PARAMS_DIR, epsilon=EPSILON_DECISION, param_type=PARAM_TYPE,
    model_type=MODEL_TYPE, timestamp=timestamp, number_size=NUMBER_SIZE,
)

all_params = {'decision': decision_params, 'unit': unit_params, 'carry': carry_params, 'cnn': cnn_params}
model_fn = decision_model_vector

tx = optax.adam(LEARNING_RATE)
opt_state = tx.init(all_params)


# --- Joint loss/update step: images -> (differentiable) CNN magnitude
# estimates -> compute_loss(), reused UNCHANGED, exactly the same MSE the
# frozen-CNN script uses. The only new code is that all_params['cnn'] is
# now part of the differentiated pytree instead of a closed-over constant. ---
def joint_loss_fn(all_params, images, y, rng):
    magnitudes = images_to_magnitudes(all_params['cnn'], images, rng, OMEGA)
    return compute_loss(all_params['decision'], magnitudes, y, all_params['unit'], all_params['carry'],
                         unit_structure_static, carry_structure_static, model_fn)


@jax.jit
def joint_train_step(all_params, opt_state, images, y, rng):
    loss, grads = jax.value_and_grad(joint_loss_fn)(all_params, images, y, rng)
    updates, opt_state = tx.update(grads, opt_state, all_params)
    all_params = optax.apply_updates(all_params, updates)
    return all_params, opt_state, loss


@jax.jit
def eval_magnitudes_fn(cnn_p, images, rng):
    # Unlike the frozen-CNN scripts, cnn_p is a live argument here (not
    # closed over) since it changes every training step.
    return images_to_magnitudes(cnn_p, images, rng, OMEGA)


def evaluate_module_perceptual(decision_params, x_eval, y_eval, unit_mod, carry_mod,
                                carry_mask=None, small_mask=None, large_mask=None, return_predictions=False):
    pred = model_fn(decision_params, x_eval, unit_mod, carry_mod, unit_structure_static, carry_structure_static)
    loss = compute_loss(decision_params, x_eval, y_eval, unit_mod, carry_mod,
                         unit_structure_static, carry_structure_static, model_fn)
    number_size = pred.shape[1] - 1
    pred_arr = jnp.round(pred).astype(int)
    y_arr = jnp.array(y_eval[:, :(number_size + 1)]).astype(int)
    powers = 10 ** jnp.arange(number_size, -1, -1)
    predictions = jnp.sum(pred_arr * powers, axis=1)
    targets = jnp.sum(y_arr * powers, axis=1)
    pred_correct = np.asarray(predictions == targets)
    pred_count = int(pred_correct.sum())

    if return_predictions:
        return pred_count, pred_count, float(loss), np.asarray(predictions), np.asarray(targets)

    tests = [0, 0, 0, 0]
    if carry_mask is not None and small_mask is not None and large_mask is not None:
        tests[0] = int(np.sum(pred_correct & small_mask & ~carry_mask))
        tests[1] = int(np.sum(pred_correct & large_mask & ~carry_mask))
        tests[2] = int(np.sum(pred_correct & small_mask & carry_mask))
        tests[3] = int(np.sum(pred_correct & large_mask & carry_mask))
    return pred_count, pred_count, float(loss), tests


# --- Fixed test IMAGES (drawn once, TEST split, fixed seed) -- but
# magnitudes are recomputed from the CURRENT (evolving) cnn params every
# evaluation call, since unlike the frozen-CNN scripts the CNN keeps changing. ---
x_test_digits, _ = generate_test_dataset(test_pairs, number_size=NUMBER_SIZE)
test_images = images_for_digits(x_test_digits, pools_test, seed=0)


def run_eval(all_params):
    x_eval = eval_magnitudes_fn(all_params['cnn'], test_images, jax.random.PRNGKey(0))
    return evaluate_module_perceptual(
        all_params['decision'], x_eval, y_test, all_params['unit'], all_params['carry'],
        carry_mask=carry_mask_test, small_mask=small_mask_test, large_mask=large_mask_test,
    )


# --- Config file ---
with open(os.path.join(SAVE_DIR, "config.txt"), "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Curriculum: all_at_once_perceptual\n")
    f.write(f"Study Name: {STUDY_NAME}\n")
    f.write(f"Number Size: {NUMBER_SIZE}\n")
    f.write(f"Model Type: {MODEL_TYPE} (argmax unsupported for all-at-once, see header note)\n")
    f.write(f"Decision Parameter Initialization Type: {PARAM_TYPE}\n")
    f.write(f"Epsilon (decision-layer init scale): {EPSILON_DECISION}\n")
    f.write(f"Epsilon (carry/unit extractor init scale; CNN uses Flax default init): {EPSILON_EXTRACTOR}\n")
    f.write(f"Weber fraction (Omega, extra noise on top of the CNN estimate): {OMEGA}\n")
    f.write(f"Fixed Variability: {'Yes' if FIXED_VARIABILITY else 'No'}\n")
    f.write(f"Distribution used for the training set: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Alpha for curriculum learning: {ALPHA_CURRICULUM}\n")
    f.write(f"Early Stop: {'Yes' if EARLY_STOP else 'No'}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epoch Size: {EPOCH_SIZE}\n")
    f.write(f"Unit Structure: {UNIT_STRUCTURE}\n")
    f.write(f"Carry Structure: {CARRY_STRUCTURE}\n")
    f.write(f"Training Pairs: {len(train_pairs)}\n")
    f.write(f"Test Pairs: {len(test_pairs)}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Training loop ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True
threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))
batches_per_epoch = max(1, EPOCH_SIZE // BATCH_SIZE)
master_rng = jax.random.PRNGKey(1)


def _log_row(epoch, loss, accuracy, pred_count, pred_count_test, tests):
    return {
        "epoch": epoch, "loss": float(loss), "accuracy": float(accuracy),
        "total_correct": pred_count, "test_correct": pred_count_test,
        "test_pairs_no_carry_small_total": totals[0], "test_pairs_no_carry_small_count": tests[0],
        "test_pairs_no_carry_small_accuracy": 100 * (tests[0] / totals[0]) if totals[0] > 0 else None,
        "test_pairs_no_carry_large_total": totals[1], "test_pairs_no_carry_large_count": tests[1],
        "test_pairs_no_carry_large_accuracy": 100 * (tests[1] / totals[1]) if totals[1] > 0 else None,
        "test_pairs_carry_small_total": totals[2], "test_pairs_carry_small_count": tests[2],
        "test_pairs_carry_small_accuracy": 100 * (tests[2] / totals[2]) if totals[2] > 0 else None,
        "test_pairs_carry_large_total": totals[3], "test_pairs_carry_large_count": tests[3],
        "test_pairs_carry_large_accuracy": 100 * (tests[3] / totals[3]) if totals[3] > 0 else None,
    }


try:
    pred_count, pred_count_test, loss, tests = run_eval(all_params)
    accuracy = pred_count / len(test_pairs) if len(test_pairs) > 0 else 0.0
    pd.DataFrame([_log_row(0, loss, accuracy, pred_count, pred_count_test, tests)]).to_csv(
        log_path, mode='a', index=False, header=first_write)
    first_write = False
    save_results_and_module(None, accuracy, all_params, SAVE_DIR, checkpoint_number=0)
    print(f"Checkpoint 0 (random init), accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"Warning: pre-training evaluation or checkpoint save failed: {e}")

for epoch in range(EPOCHS):
    for batch_idx in range(batches_per_epoch):
        seed = epoch * batches_per_epoch + batch_idx
        x_digits, y_train = generate_train_dataset(
            train_pairs, BATCH_SIZE, omega=0.0, distribution=TRAINING_DISTRIBUTION_TYPE,
            alpha=ALPHA_CURRICULUM, number_size=NUMBER_SIZE, seed=seed, fixed_variability=False,
        )
        images = images_for_digits(x_digits, pools_train, seed=seed)
        master_rng, step_rng = jax.random.split(master_rng)
        all_params, opt_state, train_loss = joint_train_step(all_params, opt_state, images, y_train, step_rng)

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        try:
            pred_count, pred_count_test, loss, tests = run_eval(all_params)
        except Exception as e:
            print(f"[ERROR] evaluation failed at epoch {epoch + 1}: {e}")
            pred_count, pred_count_test, loss, tests = 0, 0, float('nan'), [0, 0, 0, 0]
        accuracy = pred_count_test / len(test_pairs) if len(test_pairs) > 0 else 0.0
        pd.DataFrame([_log_row(epoch + 1, loss, accuracy, pred_count, pred_count_test, tests)]).to_csv(
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
        print(f"All-at-once perceptual model reached target accuracy at epoch {epoch + 1}.")
        if EARLY_STOP:
            last_epoch = epoch + 1
            last_metrics = _log_row(None, loss, accuracy, pred_count, pred_count_test, tests)
            try:
                save_results_and_module(None, accuracy, all_params, SAVE_DIR, checkpoint_number=last_epoch)
                print(f"Saved final checkpoint {last_epoch} after achieving target accuracy in {SAVE_DIR}")
            except Exception as e:
                print(f"[ERROR] Failed to save final checkpoint {last_epoch}: {e}")
            for fill_epoch in range(last_epoch + SHOW_EVERY_N_EPOCHS - (last_epoch % SHOW_EVERY_N_EPOCHS), EPOCHS + 1, SHOW_EVERY_N_EPOCHS):
                last_metrics["epoch"] = fill_epoch
                pd.DataFrame([last_metrics]).to_csv(log_path, mode='a', index=False, header=False)
            break
        # else: keep training the full EPOCHS budget, same as train_extractor_modules.py's default.

# --- Final Evaluation ---
try:
    x_eval = eval_magnitudes_fn(all_params['cnn'], test_images, jax.random.PRNGKey(0))
    final_pred_count, final_pred_count_test, final_loss, final_preds, targets = evaluate_module_perceptual(
        all_params['decision'], x_eval, y_test, all_params['unit'], all_params['carry'], return_predictions=True,
    )
    final_accuracy = final_pred_count_test / len(test_pairs) if len(test_pairs) > 0 else 0.0
except Exception as e:
    print(f"[ERROR] Final evaluation failed: {e}")
    final_preds, targets, final_accuracy = [], [], 0.0

results = []
for i in range(len(test_pairs)):
    x1, x2 = test_pairs[i]
    results.append({"x1": x1, "x2": x2, "y (true)": targets[i], "y (pred)": final_preds[i],
                     "correct": final_preds[i] == targets[i]})
df_results = pd.DataFrame(results)

save_results_and_module(df_results, final_accuracy, all_params, SAVE_DIR)
print(f"All-at-once perceptual training complete. Saved to {SAVE_DIR}")
print("Compare training_log.csv here against train_decision_module_perceptual.py (frozen CNN) and "
      "train_all_at_once.py (symbolic, no CNN at all) at matching OMEGA/EPSILON/NUMBER_SIZE to see "
      "how much the joint CNN training costs relative to those two.")
