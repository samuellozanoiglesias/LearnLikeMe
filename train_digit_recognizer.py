# USE: nohup python train_digit_recognizer.py cuenca NEW_STUDY 20 128 > logs_train_recognizer.out 2>&1 &
#
# Stage 0 of the (extended) step-by-step curriculum: learn to read a
# handwritten digit off a 28x28 MNIST image before any arithmetic is
# involved. carry_extractor / unit_extractor (train_extractor_modules.py)
# stay stage 1, decision_module (train_decision_module.py) stays stage 2.
# This mirrors the existing curriculum philosophy -- just adds a perceptual
# step before the componential-magnitude steps.
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import pickle
from datetime import datetime

import jax
print(jax.devices())  # should only show CPU

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training import train_state

from little_learner.modules.perceptual_module.digit_recognizer import DigitCNN
from little_learner.modules.perceptual_module.mnist_utils import load_mnist

# --- Config ---
CLUSTER = str(sys.argv[1]).lower()  # cuenca, brigit or local
STUDY_NAME = str(sys.argv[2]).upper()
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BATCH_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 128

LEARNING_RATE = 1e-3
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 5

# --- Paths (same conventions as the other training scripts) ---
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

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/digit_recognizer/{STUDY_NAME}"
SAVE_DIR = f"{RAW_DIR}/Training_{timestamp}"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Data ---
(x_train, y_train), (x_test, y_test) = load_mnist()
x_train, y_train = jnp.array(x_train), jnp.array(y_train)
x_test, y_test = jnp.array(x_test), jnp.array(y_test)

model = DigitCNN(output_dim=10)
rng = jax.random.PRNGKey(0)
params = model.init(rng, x_train[:1])['params']
tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def loss_fn(params, x, y):
    logits = model.apply({'params': params}, x)
    one_hot = jax.nn.one_hot(y, 10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


@jax.jit
def train_step(state, x, y):
    grads = jax.grad(loss_fn)(state.params, x, y)
    return state.apply_gradients(grads=grads)


@jax.jit
def eval_step(params, x, y):
    logits = model.apply({'params': params}, x)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == y)
    loss = loss_fn(params, x, y)
    return accuracy, loss


def save_checkpoint(state, save_dir, checkpoint_number=None):
    fname = "trained_model.pkl" if checkpoint_number is None else f"trained_model_checkpoint_{checkpoint_number}.pkl"
    with open(os.path.join(save_dir, fname), "wb") as f:
        pickle.dump(state.params, f)


# --- Config file ---
with open(os.path.join(SAVE_DIR, "config.txt"), "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Module Name: digit_recognizer\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Checkpoint Every: {CHECKPOINT_EVERY}\n")
    f.write(f"Train Images: {x_train.shape[0]}\n")
    f.write(f"Test Images: {x_test.shape[0]}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Training loop ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True
n_train = x_train.shape[0]
steps_per_epoch = max(1, n_train // BATCH_SIZE)

for epoch in range(EPOCHS):
    perm = np.random.default_rng(epoch).permutation(n_train)
    for step in range(steps_per_epoch):
        batch_idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        state = train_step(state, x_train[batch_idx], y_train[batch_idx])

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        accuracy, loss = eval_step(state.params, x_test, y_test)
        pd.DataFrame([{
            "epoch": epoch + 1, "loss": float(loss), "accuracy": float(accuracy),
        }]).to_csv(log_path, mode='a', index=False, header=first_write)
        first_write = False
        print(f"Epoch {epoch + 1}, Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(state, SAVE_DIR, checkpoint_number=epoch + 1)

save_checkpoint(state, SAVE_DIR)  # final trained_model.pkl, no frozen extra noise needed --
print(f"Digit recognizer trained. Saved to {SAVE_DIR}")
print("Feed this checkpoint's params into perceptual_models.perceptual_magnitude() as cnn_params "
      "when training the carry/unit extractors on images (train_extractor_modules_perceptual.py) "
      "or when building the cached MNIST test set (generate_mnist_test_pairs.py).")
