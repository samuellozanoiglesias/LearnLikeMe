# USE: nohup python train_decision_module_perceptual.py cuenca 2 SEVENTH_STUDY WI vector 0.10 0.05 5000 No \
#        Decreasing_exponential 0.1 decision_only > logs_train_decision_perceptual.out 2>&1 &
#
# Perceptual counterpart to train_decision_module.py: by default, same
# step-by-step philosophy (only the decision layer trains; everything
# upstream is pretrained and FROZEN), but the "digit values" it feeds into
# the decision layer are no longer symbolic integers + synthetic Weber noise
# -- they are real MNIST images read through a DigitCNN
# (train_digit_recognizer.py) and fed into carry_extractor_perceptual /
# unit_extractor_perceptual checkpoints (train_extractor_modules_perceptual.py).
#
# Optional 12th positional argument, TRAIN_MODE, controls what's frozen:
#   - "decision_only" (default, original behavior): unit/carry extractors AND
#     the DigitCNN recognizer are all frozen; only the decision layer trains.
#   - "all": unit/carry extractors AND the DigitCNN are unfrozen and
#     fine-tuned jointly with the decision layer, end-to-end from raw MNIST
#     images. NOTE: this only actually moves the CNN/extractor weights if
#     compute_loss()/perceptual_magnitude() are differentiable w.r.t. them
#     (no stop_gradient / no hard argmax in the recognizer path) -- verify
#     against your library code before relying on it.
# Full 12-arg example (retrain everything -- extractors + CNN unfrozen too):
#   nohup python train_decision_module_perceptual.py cuenca 2 SEVENTH_STUDY WI vector 0.10 0.05 5000 No \
#       Decreasing_exponential 0.1 all > logs_train_decision_perceptual.out 2>&1 &
#
# WHERE THE FROZEN CNN COMES FROM: there's deliberately no separate
# "--recognizer-checkpoint" argument. Every checkpoint saved by
# train_extractor_modules_perceptual.py already bundles the CNN params it was
# trained against (dict keys 'extractor' and 'cnn' -- see that script's
# save_checkpoint()). This script loads the 'cnn' out of the unit_extractor's
# checkpoint and uses that as the canonical frozen recognizer, and checks
# that the carry_extractor's bundled 'cnn' is numerically identical -- if
# either extractor run had FREEZE_RECOGNIZER=No (fine-tuned the CNN), or the
# two were trained against different recognizer checkpoints, this print a
# loud warning, because it means "the" frozen CNN below is ambiguous.
#
# OMEGA plays the same dual role it already plays in train_decision_module.py:
# (1) it selects which omega_{OMEGA:.2f} perceptual-extractor run to load,
# and (2) it's passed straight through as the "extra Weber noise on top of
# the CNN estimate" during decision-level training too (matching how the
# symbolic script reuses OMEGA both to pick an extractor checkpoint AND to
# noise decision-level training batches).
#
# WHY THIS DOESN'T CALL evaluate_module()/generate_test_dataset() FOR x THE
# WAY THE SYMBOLIC SCRIPT DOES: evaluate_module()'s test-set/category
# matching reconstructs (a, b) from x via exact place-value decoding
# (x_test[:, :n] @ powers-of-ten) and requires EXACT equality against
# test_pairs/carry_set/etc. That's fine when x holds exact symbolic digit
# values, but here x holds continuous CNN magnitude ESTIMATES (e.g. 6.98
# instead of 7) -- the reconstruction would essentially never match, silently
# zeroing out the carry/small/large breakdown and the test-subset count. So:
# compute_loss()/update_params() (which don't do any such reconstruction) are
# reused completely unchanged; only evaluation gets a small perceptual-aware
# replacement (evaluate_module_perceptual, below) that looks up pair
# category membership directly from the known pair list instead of
# reconstructing it from x. As a consequence this script always evaluates on
# test_pairs/x_test directly (no x_val/all_pairs branch for NUMBER_SIZE<=2--
# see train_decision_module.py's NUMBER_SIZE<=2 branch -- since that branch
# needs exactly the reconstruction-based subset matching that's unreliable here).
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import pickle
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import jax
print(jax.devices())  # should only show CPU

import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import pandas as pd

from little_learner.modules.perceptual_module.digit_recognizer import DigitCNN, perceptual_magnitude
from little_learner.modules.perceptual_module.mnist_utils import load_mnist, index_by_digit
from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset, _make_hashable, _parse_structure,
    save_results_and_module, initialize_decision_params,
)
from little_learner.modules.decision_module.train_utils import (
    update_params, compute_loss, generate_train_dataset,
)
from little_learner.modules.decision_module.model import decision_model_argmax, decision_model_vector

# --- Config (identical CLI shape to train_decision_module.py -- no new args
# needed, since the frozen CNN is pulled from the extractor checkpoints) ---
CLUSTER = str(sys.argv[1]).lower()
NUMBER_SIZE = int(sys.argv[2])
STUDY_NAME = str(sys.argv[3]).upper()
PARAM_TYPE = str(sys.argv[4]).upper()  # 'WI' or 'RI', decision layer only
MODEL_TYPE = str(sys.argv[5]).lower()  # 'argmax' or 'vector' -- both fine here, nothing upstream needs gradients
EPSILON = float(sys.argv[6])  # decision-layer init noise
OMEGA = float(sys.argv[7])  # selects the omega_{OMEGA:.2f} perceptual extractor run + extra CNN-estimate noise
EPOCHS = int(sys.argv[8]) if len(sys.argv) > 8 else 5000
FIXED_VARIABILITY = len(sys.argv) > 9 and sys.argv[9].lower() in ['yes', 'true', '1']
TRAINING_DISTRIBUTION_TYPE = str(sys.argv[10]).lower() if len(sys.argv) > 10 else "none"
ALPHA_CURRICULUM = float(sys.argv[11]) if len(sys.argv) > 11 else 0.1
TRAIN_MODE = str(sys.argv[12]).lower() if len(sys.argv) > 12 else "decision_only"  # "decision_only" (freeze extractors + CNN, original behavior) or "all" (unfreeze + jointly train extractors + CNN too)
if TRAIN_MODE not in ("decision_only", "all"):
    raise ValueError("Invalid TRAIN_MODE. Choose 'decision_only' or 'all'.")

LEARNING_RATE = 0.003
BATCH_SIZE = 25
EPOCH_SIZE = 100
FINISH_TOLERANCE = 0.0
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 10

# Structures MUST match what train_extractor_modules_perceptual.py actually
# used (it hardcodes these, matching the symbolic train_extractor_modules.py).
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
RAW_DIR = f"{MODULES_DIR}/decision_module_perceptual/{NUMBER_SIZE}-digit/{STUDY_NAME}"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/{MODEL_TYPE}_version/epsilon_{EPSILON:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)


# --- Dataset loading (same files/convention as train_decision_module.py) ---
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


# generate_arithmetic_datasets.py's generate_test_categories() always builds
# these four files (unconditionally, for any number_size), so we load them
# directly rather than re-deriving carry/small/large membership ourselves.
test_carry_small = set(load_category("test_carry_small.txt"))
test_carry_large = set(load_category("test_carry_large.txt"))
test_no_carry_small = set(load_category("test_no_carry_small.txt"))
test_no_carry_large = set(load_category("test_no_carry_large.txt"))

carry_mask_test = np.array([p in test_carry_small or p in test_carry_large for p in test_pairs])
small_mask_test = np.array([p in test_no_carry_small or p in test_carry_small for p in test_pairs])
large_mask_test = np.array([p in test_no_carry_large or p in test_carry_large for p in test_pairs])
totals = [len(test_no_carry_small), len(test_no_carry_large), len(test_carry_small), len(test_carry_large)]

# y-target only (we discard the symbolic x -- we build our own from images).
_, y_test = generate_test_dataset(test_pairs, number_size=NUMBER_SIZE)

# --- Load pretrained, frozen perceptual extractors (+ the CNN bundled with them) ---
def load_extractor_module_perceptual(omega, modules_dir, model_type, study_name):
    """
    Load a {carry,unit}_extractor_perceptual checkpoint as saved by
    train_extractor_modules_perceptual.py's save_checkpoint(): a dict with
    keys 'extractor' (ExtractorModel params) and 'cnn' (the DigitCNN params
    that run trained against -- frozen unless FREEZE_RECOGNIZER=No was used).
    """
    base = os.path.join(modules_dir, f"{model_type}_perceptual", study_name, f"omega_{omega:.2f}")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No perceptual {model_type} runs found at {base}. "
                                 f"Run train_extractor_modules_perceptual.py for this omega first.")
    candidates = [os.path.join(base, name) for name in os.listdir(base)
                  if name.startswith("Training_") and os.path.isdir(os.path.join(base, name))]
    if not candidates:
        raise FileNotFoundError(f"No Training_* folders under {base}.")
    chosen = max(candidates, key=os.path.getmtime)
    model_path = os.path.join(chosen, "trained_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"trained_model.pkl not found in {chosen}.")
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    if 'extractor' not in ckpt or 'cnn' not in ckpt:
        raise KeyError(f"Expected keys 'extractor' and 'cnn' in {model_path}, found {list(ckpt.keys())}.")
    return ckpt['extractor'], ckpt['cnn'], chosen


carry_module, carry_cnn_params, carry_dir = load_extractor_module_perceptual(OMEGA, MODULES_DIR, 'carry_extractor', STUDY_NAME)
unit_module, unit_cnn_params, unit_dir = load_extractor_module_perceptual(OMEGA, MODULES_DIR, 'unit_extractor', STUDY_NAME)

cnn_mismatch = not jax.tree_util.tree_all(
    jax.tree_util.tree_map(lambda a, b: bool(jnp.allclose(a, b)), carry_cnn_params, unit_cnn_params)
)
if cnn_mismatch:
    print("[WARNING] carry_extractor's and unit_extractor's bundled CNN params differ -- "
          "they were either trained against different recognizer checkpoints, or one/both "
          "runs used FREEZE_RECOGNIZER=No and fine-tuned the CNN differently. Using the "
          "unit_extractor's CNN as the canonical frozen recognizer below; if that's not what "
          "you want, retrain the extractors against a single shared, frozen recognizer checkpoint.")
cnn_params = unit_cnn_params
cnn = DigitCNN(output_dim=10)

# --- MNIST images: TRAIN split for training batches, TEST split for the
# fixed evaluation set (never mixed -- see generate_mnist_test_pairs.py). ---
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist()
pools_train = index_by_digit(x_train_mnist, y_train_mnist)
pools_test = index_by_digit(x_test_mnist, y_test_mnist)


def images_for_digits(digit_values, pools, seed):
    """
    digit_values: array-like (batch, n_positions) of integer digit values (0-9).
    pools: dict digit -> array of MNIST images for that digit.
    Returns a jnp.array (batch, n_positions, *image_shape): one randomly
    drawn image per digit value, with replacement, fresh every call -- the
    "data augmentation" role Weber noise played for the symbolic model.
    """
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
    """
    images: (batch, n_positions, H, W[, C]) MNIST images, one per digit
    position (number_size digits of the first addend, then number_size of
    the second -- the exact x-layout decision_model_vector/_argmax expect).
    Returns (batch, n_positions) magnitude estimates: a drop-in replacement
    for the noisy digit-value array the symbolic pipeline builds, so it can
    be fed straight into the UNCHANGED decision_model_vector/compute_loss.
    """
    n_positions = images.shape[1]
    rngs = jax.random.split(rng, n_positions)
    mags = []
    for pos in range(n_positions):
        mag, _, _ = perceptual_magnitude(cnn_p, images[:, pos], cnn, omega=omega, rng=rngs[pos])
        mags.append(mag)
    return jnp.stack(mags, axis=1)


@jax.jit
def magnitudes_fn(cnn_p, images, rng):
    # cnn_p is now an explicit (traced) argument rather than a closed-over
    # Python constant. With TRAIN_MODE="decision_only" cnn_params never
    # changes, so this behaves exactly like before. With TRAIN_MODE="all" the
    # CNN is fine-tuned, and a closed-over cnn_params would get baked into
    # the jit cache at first call and silently ignore all later updates --
    # passing it explicitly avoids that trap. OMEGA/cnn are still closed over
    # since they never change during a run.
    return images_to_magnitudes(cnn_p, images, rng, OMEGA)


# --- Perceptual-aware evaluation (see header note: evaluate_module()'s
# exact-value pair reconstruction from x doesn't work with continuous CNN
# magnitude estimates, so category/subset matching is done from the known
# pair list + precomputed masks instead). Loss/prediction logic is the same
# as evaluate_module()'s. ---
def evaluate_module_perceptual(decision_params, x_eval, y_eval, unit_mod, carry_mod, model_fn,
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


# --- Fixed test set: images drawn once (fixed seed, TEST split), magnitudes
# computed once (CNN is frozen, so they never change across the run). ---
x_test_digits, _ = generate_test_dataset(test_pairs, number_size=NUMBER_SIZE)  # digit values, just to build images from
test_images = images_for_digits(x_test_digits, pools_test, seed=0)

def compute_x_test(cnn_p):
    return magnitudes_fn(cnn_p, test_images, jax.random.PRNGKey(0))

# Initial test-set magnitudes. If TRAIN_MODE="all" this is recomputed with
# the current (fine-tuned) cnn_params before every evaluation below, since a
# moving CNN means the "same" test images produce different magnitude
# estimates as training progresses; if TRAIN_MODE="decision_only" the CNN
# never changes so this fixed x_test is reused as-is (matches original
# behavior, and avoids recomputing it every epoch).
x_test = compute_x_test(cnn_params)

# --- Decision-layer params: fresh, WI/RI init (unchanged from the symbolic script) ---
decision_params = initialize_decision_params(
    PARAMS_DIR, epsilon=EPSILON, param_type=PARAM_TYPE, model_type=MODEL_TYPE,
    timestamp=timestamp, number_size=NUMBER_SIZE,
)

if MODEL_TYPE == "vector":
    model_fn = decision_model_vector
elif MODEL_TYPE == "argmax":
    model_fn = decision_model_argmax
else:
    raise ValueError("Invalid model type. Choose 'argmax' or 'vector'.")

# --- TRAIN_MODE wiring ---
# "decision_only": unchanged behavior -- unit_module/carry_module/cnn_params
#   are only ever read (used to turn images into decision-model inputs),
#   never updated. Training batches are built by first computing magnitudes
#   with the frozen CNN (magnitudes_fn), then calling update_params on the
#   decision params only, exactly like the original script.
# "all": unit_module/carry_module/cnn_params are folded into one trainable
#   pytree with the decision params, and gradients are taken end-to-end from
#   raw MNIST images through the CNN, both extractors, and the decision
#   model in a single jax.grad call. NOTE: this only moves the CNN/extractor
#   weights if compute_loss()/perceptual_magnitude() don't stop_gradient
#   anywhere on that path (they have no reason to today, since nothing
#   upstream ever differentiated through them) -- verify against your
#   library code if you rely on this.
if TRAIN_MODE == "all":
    trainable = {"decision": decision_params, "unit": unit_module, "carry": carry_module, "cnn": cnn_params}

    def _joint_loss(trainable_params, images, y, rng):
        x = images_to_magnitudes(trainable_params["cnn"], images, rng, OMEGA)
        return compute_loss(
            trainable_params["decision"], x, y,
            trainable_params["unit"], trainable_params["carry"],
            unit_structure_static, carry_structure_static, model_fn
        )

    _joint_grad_fn = jax.jit(jax.grad(_joint_loss))

    def train_step(trainable_params, images, y, rng):
        grads = _joint_grad_fn(trainable_params, images, y, rng)
        return jax.tree_util.tree_map(
            lambda p, g: p - LEARNING_RATE * g, trainable_params, grads
        )
else:
    trainable = {"decision": decision_params, "unit": unit_module, "carry": carry_module, "cnn": cnn_params}

    def train_step(trainable_params, images, y, rng):
        # CNN stays frozen: compute magnitudes with it first, outside any
        # gradient tape, exactly like the original script.
        x = magnitudes_fn(trainable_params["cnn"], images, rng)
        new_decision = update_params(
            trainable_params["decision"], x, y,
            trainable_params["unit"], trainable_params["carry"], LEARNING_RATE,
            model_fn=model_fn, unit_structure=unit_structure_static, carry_structure=carry_structure_static,
        )
        trainable_params["decision"] = new_decision
        return trainable_params

def _save_finetuned_modules(save_dir, checkpoint_number=None):
    """Only meaningful in TRAIN_MODE='all': persist the fine-tuned extractor
    and CNN params alongside the decision-module checkpoint, since
    save_results_and_module() only knows how to save the decision params."""
    if TRAIN_MODE != "all":
        return
    suffix = f"_checkpoint_{checkpoint_number}" if checkpoint_number is not None else "_final"
    with open(os.path.join(save_dir, f"unit_extractor_finetuned{suffix}.pkl"), "wb") as f:
        pickle.dump(trainable["unit"], f)
    with open(os.path.join(save_dir, f"carry_extractor_finetuned{suffix}.pkl"), "wb") as f:
        pickle.dump(trainable["carry"], f)
    with open(os.path.join(save_dir, f"cnn_finetuned{suffix}.pkl"), "wb") as f:
        pickle.dump(trainable["cnn"], f)

# --- Config file ---
with open(os.path.join(SAVE_DIR, "config.txt"), "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Module Name: decision_module_perceptual\n")
    f.write(f"Study Name: {STUDY_NAME}\n")
    f.write(f"Model Type (Argmax or Vector): {MODEL_TYPE}\n")
    f.write(f"Number Size: {NUMBER_SIZE}\n")
    f.write(f"Parameter Initialization Type: {PARAM_TYPE}\n")
    f.write(f"Noise Factor for Initialization Parameters (Epsilon): {EPSILON}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epoch Size: {EPOCH_SIZE}\n")
    f.write(f"Weber fraction (Omega, selects extractor run + extra CNN-estimate noise): {OMEGA}\n")
    f.write(f"Fixed Variability: {'Yes' if FIXED_VARIABILITY else 'No'}\n")
    f.write(f"Unit Extractor (perceptual) imported: {unit_dir}\n")
    f.write(f"Carry Extractor (perceptual) imported: {carry_dir}\n")
    f.write(f"CNN bundled-params mismatch between carry/unit extractors: {'Yes -- see warning above' if cnn_mismatch else 'No'}\n")
    f.write(f"Train Mode (decision_only = extractors+CNN frozen, all = extractors+CNN jointly fine-tuned): {TRAIN_MODE}\n")
    f.write(f"Distribution used for the training set: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Alpha for curriculum learning: {ALPHA_CURRICULUM}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"Show Every N Epochs: {SHOW_EVERY_N_EPOCHS}\n")
    f.write(f"Checkpoint Every: {CHECKPOINT_EVERY}\n")
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
    pred_count, pred_count_test, loss, tests = evaluate_module_perceptual(
        trainable["decision"], x_test, y_test, trainable["unit"], trainable["carry"], model_fn,
        carry_mask=carry_mask_test, small_mask=small_mask_test, large_mask=large_mask_test,
    )
    accuracy = pred_count / len(test_pairs) if len(test_pairs) > 0 else 0.0
    pd.DataFrame([_log_row(0, loss, accuracy, pred_count, pred_count_test, tests)]).to_csv(
        log_path, mode='a', index=False, header=first_write)
    first_write = False
    save_results_and_module(None, accuracy, trainable["decision"], SAVE_DIR, checkpoint_number=0)
    _save_finetuned_modules(SAVE_DIR, checkpoint_number=0)
    print(f"Saved pre-training checkpoint 0 in {SAVE_DIR}")
except Exception as e:
    print(f"Warning: pre-training evaluation or checkpoint save failed: {e}")

for epoch in range(EPOCHS):
    for batch_idx in range(batches_per_epoch):
        seed = epoch * batches_per_epoch + batch_idx
        # Reuse generate_train_dataset with omega=0.0 to get the curriculum-
        # sampled pairs' EXACT digit values + correct sum target, then swap
        # those exact values for real MNIST images + the frozen CNN's own
        # magnitude estimate (with OMEGA applied as extra noise on top).
        x_digits, y_train = generate_train_dataset(
            train_pairs, BATCH_SIZE, omega=0.0, distribution=TRAINING_DISTRIBUTION_TYPE,
            alpha=ALPHA_CURRICULUM, number_size=NUMBER_SIZE, seed=seed, fixed_variability=False,
        )
        images = images_for_digits(x_digits, pools_train, seed=seed)
        master_rng, step_rng = jax.random.split(master_rng)
        # Update parameters (decision-only, or decision+extractors+CNN
        # jointly per TRAIN_MODE). In "decision_only" mode, train_step
        # computes magnitudes with the frozen CNN internally, same as before.
        trainable = train_step(trainable, images, y_train, step_rng)

    decision_params, unit_module, carry_module, cnn_params = (
        trainable["decision"], trainable["unit"], trainable["carry"], trainable["cnn"]
    )

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        try:
            if TRAIN_MODE == "all":
                # CNN just moved -- the fixed test images now map to
                # different magnitude estimates, so recompute x_test.
                x_test = compute_x_test(cnn_params)
            pred_count, pred_count_test, loss, tests = evaluate_module_perceptual(
                decision_params, x_test, y_test, unit_module, carry_module, model_fn,
                carry_mask=carry_mask_test, small_mask=small_mask_test, large_mask=large_mask_test,
            )
        except Exception as e:
            print(f"[ERROR] evaluate_module_perceptual failed at epoch {epoch + 1}: {e}")
            pred_count, pred_count_test, loss, tests = 0, 0, float('nan'), [0, 0, 0, 0]
        accuracy = pred_count_test / len(test_pairs) if len(test_pairs) > 0 else 0.0
        pd.DataFrame([_log_row(epoch + 1, loss, accuracy, pred_count, pred_count_test, tests)]).to_csv(
            log_path, mode='a', index=False, header=first_write)
        first_write = False
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        try:
            save_results_and_module(None, accuracy, decision_params, SAVE_DIR, checkpoint_number=epoch + 1)
            _save_finetuned_modules(SAVE_DIR, checkpoint_number=epoch + 1)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint {epoch + 1}: {e}")

    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        last_epoch = epoch + 1
        last_metrics = _log_row(None, loss, accuracy, pred_count, pred_count_test, tests)
        for fill_epoch in range(last_epoch + SHOW_EVERY_N_EPOCHS - (last_epoch % SHOW_EVERY_N_EPOCHS), EPOCHS + 1, SHOW_EVERY_N_EPOCHS):
            last_metrics["epoch"] = fill_epoch
            pd.DataFrame([last_metrics]).to_csv(log_path, mode='a', index=False, header=False)
        break

# --- Final Evaluation ---
try:
    if TRAIN_MODE == "all":
        x_test = compute_x_test(cnn_params)
    final_pred_count, final_pred_count_test, final_loss, final_preds, targets = evaluate_module_perceptual(
        decision_params, x_test, y_test, unit_module, carry_module, model_fn, return_predictions=True,
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

save_results_and_module(df_results, final_accuracy, decision_params, SAVE_DIR)
_save_finetuned_modules(SAVE_DIR)
print('Training complete.')