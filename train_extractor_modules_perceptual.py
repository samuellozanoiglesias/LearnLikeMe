# USE: 
# nohup python train_extractor_modules_perceptual.py cuenca unit_extractor NEW_STUDY 0.05 /path/to/digit_recognizer/trained_model.pkl 0.10 Yes 500 25 Decreasing_exponential 0.1 Yes > logs_train_extractor_mnist.out 2>&1 &
#
# Example path: /data/samuel_lozano/LearnLikeMe/unit_extractor_perceptual/NEW_STUDY/omega_0.10/Training_2026-07-19_17-38-37/
#
# Perceptual counterpart to train_extractor_modules.py: instead of a symbolic
# digit value plus synthetic Weber noise, each single-digit input is now a real
# MNIST image, read through DigitCNN (perceptual_models.py). The CNN's
# softmax-weighted expected value is a differentiable magnitude estimate that
# already carries whatever confusion the *image* creates; OMEGA controls how
# much *additional* Weber/ANS noise (if any) sits on top of that estimate --
# set OMEGA=0.0 to run the "pure perception, no extra noise" condition and
# compare it against OMEGA>0 runs and against the original symbolic-noise model.
#
# By default the recognizer is FROZEN (matches the step-by-step philosophy:
# learn to read digits first, learn arithmetic on top of that fixed skill).
# Set FREEZE_RECOGNIZER=No to fine-tune it jointly with the extractor.
#
# EPSILON controls the extractor's initial-parameter noise, exactly like in
# train_extractor_modules.py -- and, like there, the extractor's initial
# params are created with create_and_save_initial_params/load_initial_params
# so runs are reproducible and comparable across the symbolic and perceptual
# pipelines. When the recognizer is frozen (default), the CNN just turns each
# batch of images into (mag_a, mag_b) features that stand in for the noisy
# symbolic x_batch -- so the extractor itself is trained with the *same*
# load_train_state / train_step / evaluate / compute_loss used in
# train_extractor_modules.py, not a separate hand-rolled copy. Only the joint
# fine-tuning case (FREEZE_RECOGNIZER=No) needs custom code, since gradients
# there must flow back through the CNN as well.
#

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import pickle
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import jax
print(jax.devices())  # should only show CPU

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training import train_state

from little_learner.modules.perceptual_module.digit_recognizer import DigitCNN, perceptual_magnitude
from little_learner.modules.perceptual_module.mnist_utils import load_mnist, index_by_digit, generate_mnist_batch
from little_learner.modules.extractor_modules.models import ExtractorModel
from little_learner.modules.extractor_modules.utils import (
    load_dataset, create_and_save_initial_params, load_initial_params, one_hot_encode
)
from little_learner.modules.extractor_modules.train_utils import (
    load_train_state,
    evaluate as evaluate_extractor,
    train_step as train_step_extractor,
    compute_loss as compute_loss_extractor,
)

# --- Config ---
CLUSTER = str(sys.argv[1]).lower()
MODULE_NAME = sys.argv[2].lower()  # unit_extractor or carry_extractor
STUDY_NAME = str(sys.argv[3]).upper()
EPSILON = float(sys.argv[4]) if sys.argv[4] != "None" else None  # Noise factor for EXTRACTOR parameter initialization
RECOGNIZER_CHECKPOINT = str(sys.argv[5])  # path to a trained_model*.pkl from train_digit_recognizer.py
FREEZE_RECOGNIZER = (len(sys.argv) <= 6) or sys.argv[6].lower() in ['yes', 'true', '1']
OMEGA = float(sys.argv[7]) if len(sys.argv) > 6 else 0.0  # EXTRA Weber noise on top of the CNN estimate
EARLY_STOP = len(sys.argv) > 8 and sys.argv[8].lower() in ['yes', 'true', '1']
TRAINING_DISTRIBUTION_TYPE = str(sys.argv[9]).lower() if len(sys.argv) > 10 else "none"  # Curriculum learning distribution
ALPHA_CURRICULUM = float(sys.argv[10]) if len(sys.argv) > 11 else 0.1  # Only used if TRAINING_DISTRIBUTION_TYPE is "decreasing_exponential"

# --- Training Parameters ---
LEARNING_RATE = 0.003
PARAMS_FILE = None  # Set to None to create new params, or provide a path to load existing params
EPOCH_SIZE = 100  # Number of examples per epoch

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

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}_perceptual/{STUDY_NAME}"
SAVE_DIR = f"{RAW_DIR}/omega_{OMEGA:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

DATASET_DIR = f"{CODE_DIR}/datasets"

# --- Model Hyperparameters ---
if MODULE_NAME == "carry_extractor":
    num_classes = 2
    FINISH_TOLERANCE = 0.00  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 500  # Carry model uses 500 epochs
    BATCH_SIZE = 25
    SHOW_EVERY_N_EPOCHS = 1  # Show accuracy every 1 epochs
    CHECKPOINT_EVERY = 1  # Save checkpoint every 1 epochs
    structure = [16]  # Carry model hidden layer sizes
    output_dim = 2  # Carry model output dimension (carry or no carry)

elif MODULE_NAME == "unit_extractor":
    num_classes = 10
    FINISH_TOLERANCE = 0.00  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 5000  # Unit model uses 5000 epochs
    BATCH_SIZE = 25
    SHOW_EVERY_N_EPOCHS = 5  # Show accuracy every 5 epochs
    CHECKPOINT_EVERY = 10  # Save checkpoint every 10 epochs
    structure = [128, 64]  # Unit model hidden layer sizes
    output_dim = 10  # Unit model output dimension (0-9)

else:
    raise ValueError("Invalid module name. Choose 'carry_extractor' or 'unit_extractor'.")


# --- Data: single-digit pairs (same source and loader as the symbolic pipeline) ---
single_digit_pairs = load_dataset(os.path.join(DATASET_DIR, "single_digit_additions.txt"))
all_train_pairs = [(a, b) for a in range(10) for b in range(10)]

# --- MNIST images: TRAIN split for training, TEST split for the held-out
# evaluation that gets logged/plotted during training. Mixing these would let
# the logged "accuracy" curve reflect performance on images from the same
# distribution the model is being fit to, rather than genuine generalization
# -- the same TRAIN/TEST separation generate_mnist_test_pairs.py's docstring
# insists on for the decision-level test set applies here too.
(x_train, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist()
pools_train = index_by_digit(x_train, y_train_mnist)
pools_test = index_by_digit(x_test_mnist, y_test_mnist)

# --- Models ---
cnn = DigitCNN(output_dim=10)
with open(RECOGNIZER_CHECKPOINT, "rb") as f:
    cnn_params = pickle.load(f)

extractor = ExtractorModel(structure=structure, output_dim=output_dim)

# --- Extractor initial parameters: same epsilon-noise / save-and-load
# convention as train_extractor_modules.py, so a given epsilon means the same
# thing and produces a comparable init in both pipelines. ---
if PARAMS_FILE is not None:
    PARAMS_FILE = os.path.join(PARAMS_DIR, PARAMS_FILE)
    extractor_initial_params = load_initial_params(PARAMS_FILE)
else:
    PARAMS_FILE = os.path.join(PARAMS_DIR, f"initial_params_{timestamp}.json")
    rng = jax.random.PRNGKey(42)  # same seed as train_extractor_modules.py
    extractor_initial_params = create_and_save_initial_params(
        extractor, rng, (1, 2), PARAMS_FILE, epsilon=EPSILON
    )

# --- Train state ---
if FREEZE_RECOGNIZER:
    # CNN is fixed -> the extractor is trained exactly like in
    # train_extractor_modules.py: same load_train_state, and later the same
    # train_step/evaluate/compute_loss, just fed CNN-derived (mag_a, mag_b)
    # features instead of a noisy symbolic x_batch.
    state = load_train_state(extractor, LEARNING_RATE, extractor_initial_params)
else:
    # Joint fine-tuning needs gradients through the CNN too, so both sets of
    # params share one optimizer state. This path keeps its own loss/train_step
    # below, since train_utils' versions assume a single model.
    trainable_params = {'extractor': extractor_initial_params, 'cnn': cnn_params}
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=None, params=trainable_params, tx=tx)


def compute_features(cnn_p, images_a, images_b, rng_key):
    rng_a, rng_b = jax.random.split(rng_key)
    mag_a, _, _ = perceptual_magnitude(cnn_p, images_a, cnn, omega=OMEGA, rng=rng_a)
    mag_b, _, _ = perceptual_magnitude(cnn_p, images_b, cnn, omega=OMEGA, rng=rng_b)
    return jnp.stack([mag_a, mag_b], axis=-1)  # (batch, 2), same shape ExtractorModel expects


# --- Joint (non-frozen) loss/train_step/evaluate. Unavoidably custom, since
# gradients must flow through the CNN as well as the extractor; the y_one_hot
# convention still matches train_utils (one-hot computed outside the grad). ---
def joint_loss_fn(params, images_a, images_b, y_one_hot, rng_key):
    features = compute_features(params['cnn'], images_a, images_b, rng_key)
    logits = extractor.apply({'params': params['extractor']}, features)
    return jnp.mean(optax.softmax_cross_entropy(logits, y_one_hot))


def joint_train_step(state, images_a, images_b, y_one_hot, rng_key):
    grads = jax.grad(joint_loss_fn)(state.params, images_a, images_b, y_one_hot, rng_key)
    return state.apply_gradients(grads=grads)


def joint_evaluate(params, images_a, images_b, y_one_hot, rng_key):
    features = compute_features(params['cnn'], images_a, images_b, rng_key)
    logits = extractor.apply({'params': params['extractor']}, features)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == jnp.argmax(y_one_hot, axis=-1))
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y_one_hot))
    return float(accuracy), float(loss)


def save_checkpoint(state, save_dir, checkpoint_number=None):
    fname = "trained_model.pkl" if checkpoint_number is None else f"trained_model_checkpoint_{checkpoint_number}.pkl"
    if FREEZE_RECOGNIZER:
        extractor_params, cnn_p = state.params, cnn_params
    else:
        extractor_params, cnn_p = state.params['extractor'], state.params['cnn']
    with open(os.path.join(save_dir, fname), "wb") as f:
        pickle.dump({'extractor': extractor_params, 'cnn': cnn_p}, f)


# --- Config file ---
with open(os.path.join(SAVE_DIR, "config.txt"), "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Module Name: {MODULE_NAME}_perceptual\n")
    f.write(f"Recognizer Checkpoint: {RECOGNIZER_CHECKPOINT}\n")
    f.write(f"Recognizer Frozen: {'Yes' if FREEZE_RECOGNIZER else 'No'}\n")
    f.write(f"Extra Weber Noise (Omega, on top of the CNN estimate): {OMEGA}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Structure: {structure}\n")
    f.write(f"Noise Factor for Extractor Initialization Parameters (Epsilon): {EPSILON}\n")
    f.write(f"Extractor Parameters File: {PARAMS_FILE}\n")
    f.write(f"Training Distribution Type: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Alpha for curriculum learning: {ALPHA_CURRICULUM}\n")
    f.write(f"Early Stop: {'Yes' if EARLY_STOP else 'No'}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Training loop ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True
batches_per_epoch = max(1, EPOCH_SIZE // BATCH_SIZE)
master_rng = jax.random.PRNGKey(1)
threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))
accuracy, loss = None, None  # populated on first eval, checked each epoch for early stop

for epoch in range(EPOCHS):
    for batch_idx in range(batches_per_epoch):
        seed = epoch * batches_per_epoch + batch_idx
        images_a, images_b, y = generate_mnist_batch(
            all_train_pairs, BATCH_SIZE, pools_train, MODULE_NAME, seed,
            distribution=TRAINING_DISTRIBUTION_TYPE, alpha=ALPHA_CURRICULUM
        )
        master_rng, step_rng = jax.random.split(master_rng)
        y_one_hot = jnp.array(one_hot_encode(y, num_classes=num_classes), dtype=jnp.float32)

        if FREEZE_RECOGNIZER:
            # CNN forward pass only (no grad needed) -> features become the
            # extractor's x_batch, then reuse the exact train_step used by
            # train_extractor_modules.py.
            x_batch = compute_features(cnn_params, images_a, images_b, step_rng)
            state, grads = train_step_extractor(state, x_batch, y_one_hot)
        else:
            state = joint_train_step(state, images_a, images_b, y_one_hot, step_rng)

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        # Fixed seed + TEST-split pools, no curriculum: a genuinely held-out,
        # reproducible evaluation set, not a fresh sample from the same pool
        # training draws from.
        images_a, images_b, y = generate_mnist_batch(single_digit_pairs, len(single_digit_pairs), pools_test,
                                                       MODULE_NAME, seed=999999)
        master_rng, eval_rng = jax.random.split(master_rng)
        y_val_one_hot = jnp.array(one_hot_encode(y, num_classes=num_classes), dtype=jnp.float32)

        if FREEZE_RECOGNIZER:
            x_val = compute_features(cnn_params, images_a, images_b, eval_rng)
            accuracy = evaluate_extractor(extractor, state.params, x_val, y_val_one_hot)
            loss = compute_loss_extractor(extractor, state.params, x_val, y_val_one_hot)
        else:
            accuracy, loss = joint_evaluate(state.params, images_a, images_b, y_val_one_hot, eval_rng)

        pd.DataFrame([{"epoch": epoch + 1, "loss": float(loss), "accuracy": float(accuracy)}]).to_csv(
            log_path, mode='a', index=False, header=first_write)
        first_write = False
        print(f"Epoch {epoch + 1}, Loss: {float(loss):.4f}, Accuracy: {float(accuracy):.4f}")

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(state, SAVE_DIR, checkpoint_number=epoch + 1)

    # --- Early stopping check (mirrors train_extractor_modules.py) ---
    if accuracy is not None:
        accuracy_dec = Decimal(str(float(accuracy))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        if accuracy_dec >= threshold:
            print(f"All combinations have been learned correctly! Epoch: {epoch + 1}.")
            if EARLY_STOP:
                last_epoch = epoch + 1
                last_metrics = {"epoch": None, "loss": float(loss), "accuracy": float(accuracy)}

                try:
                    save_checkpoint(state, SAVE_DIR, checkpoint_number=last_epoch)
                    print(f"Saved final checkpoint {last_epoch} after achieving target accuracy in {SAVE_DIR}")
                except Exception as e:
                    print(f"[ERROR] Failed to save final checkpoint {last_epoch}: {e}")

                # Fill remaining epochs with last metrics
                for fill_epoch in range(last_epoch + SHOW_EVERY_N_EPOCHS - (last_epoch % SHOW_EVERY_N_EPOCHS),
                                         EPOCHS + 1, SHOW_EVERY_N_EPOCHS):
                    last_metrics["epoch"] = fill_epoch
                    pd.DataFrame([last_metrics]).to_csv(log_path, mode='a', index=False, header=False)

                break

save_checkpoint(state, SAVE_DIR)
print(f"Perceptual {MODULE_NAME} trained. Saved to {SAVE_DIR}")