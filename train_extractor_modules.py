import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import os
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from little_learner.modules.extractor_modules.utils import (
    create_and_save_initial_params, load_initial_params, generate_unit_data, generate_carry_data, one_hot_encode, save_results_and_model
)
from little_learner.modules.extractor_modules.models import UnitLSTMModel, CarryLSTMModel
from little_learner.modules.extractor_modules.train_utils import (
    load_train_state, evaluate, train_step, get_predictions, compute_loss
)

# --- Config ---
CLUSTER = "cuenca" # Cuenca, Brigit or Local
MODULE_NAME = sys.argv[1]  # unit_extractor or carry_over_extractor
TRAINING_DATA_TYPE = "gaussian" # gaussian (not exact numbers) or default (exact numbers)

# --- Training Parameters ---
LEARNING_RATE = 0.1
PARAMS_FILE = None  # Set to None to create new params, or provide a path to load existing params
OMEGA = float(sys.argv[2])  # Weber fraction (~0.2) for gaussian noise, if applicable

# --- Paths ---
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")
RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}" 
SAVE_DIR = f"{RAW_DIR}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/Initial_Parameters"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Data Preparation ---
if MODULE_NAME == "carry_over_extractor":
    x, y = generate_carry_data(training_data_type=TRAINING_DATA_TYPE, omega=OMEGA)
    y_one_hot = one_hot_encode(y, num_classes=2)
    model = CarryLSTMModel()
    FINISH_TOLERANCE = 0.01  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 2000  # Carry model uses 2000 epochs
    BATCH_SIZE = 10  # Carry model uses batch size of 10
    SHOW_EVERY_N_EPOCHS = 10  # Show accuracy every 10 epochs
    CHECKPOINT_EVERY = 500  # Save checkpoint every 500 epochs

elif MODULE_NAME == "unit_extractor":
    x, y = generate_unit_data(training_data_type=TRAINING_DATA_TYPE, omega=OMEGA)
    y_one_hot = one_hot_encode(y, num_classes=10)
    model = UnitLSTMModel()
    FINISH_TOLERANCE = 0.1  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 1000000  # Unit model uses 1000000 epochs
    BATCH_SIZE = 100  # Unit model uses batch size of 100
    SHOW_EVERY_N_EPOCHS = 1000  # Show accuracy every 1000 epochs
    CHECKPOINT_EVERY = 100000 # Save checkpoint every 100000 epochs

else:
    raise ValueError("Invalid module name. Choose 'carry_over_extractor' or 'unit_extractor'.")

if TRAINING_DATA_TYPE == "default":
    FINISH_TOLERANCE = 0.0  # No tolerance for exact numbers

x = jnp.array(x, dtype=jnp.float32)
y_one_hot = jnp.array(y_one_hot, dtype=jnp.float32)

x_train = x[:, None, :]
x_val   = x[:, None, :]
y_train = y_one_hot
y_val   = y_one_hot

# --- Save Config File ---
config_path = os.path.join(SAVE_DIR, "config.txt")
with open(config_path, "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Module Name: {MODULE_NAME}\n")
    f.write(f"Training Data Type: {TRAINING_DATA_TYPE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Parameters File: {PARAMS_FILE if PARAMS_FILE else 'New parameters generated'}\n")
    f.write(f"Weber fraction: {OMEGA}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"Data Shape (x): {x.shape}\n")
    f.write(f"Data Shape (y): {y.shape}\n")
    f.write(f"Show Every N Epochs: {SHOW_EVERY_N_EPOCHS}\n")
    f.write(f"Checkpoint Every: {CHECKPOINT_EVERY}\n")
    f.write(f"Model: {model.__class__.__name__}\n")
    f.write(f"JAX Devices: {jax.devices()}\n")

# --- Model & State ---
if PARAMS_FILE is not None:
    PARAMS_FILE = os.path.join(PARAMS_DIR, PARAMS_FILE)
    initial_params = load_initial_params(PARAMS_FILE)
else:
    PARAMS_FILE = os.path.join(PARAMS_DIR, f"initial_params_{timestamp}.json")
    rng = random.PRNGKey(10)
    input_shape = (1, x_train.shape[1], x_train.shape[2])  # (batch_size, sequence_length, features)
    initial_params = create_and_save_initial_params(model, rng, input_shape, PARAMS_FILE)

state = load_train_state(model, LEARNING_RATE, initial_params)

# --- Training Loop ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True
threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))

for epoch in range(EPOCHS):
    for i in range(0, len(x_train), BATCH_SIZE):
        x_batch = x_train[i:i + BATCH_SIZE]
        y_batch = y_train[i:i + BATCH_SIZE]
        state = train_step(model, state, x_batch, y_batch)
    accuracy = evaluate(model, state.params, x_val, y_val)
    loss = compute_loss(model, state.params, x_val, y_val)

    pd.DataFrame([{
        "epoch": epoch + 1,
        "loss": float(loss),
        "accuracy": float(accuracy)
    }]).to_csv(log_path, mode='a', index=False, header=first_write)
    first_write = False

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.params")
        save_results_and_model(df_results=None, final_accuracy=None, model_params=state.params, save_dir=SAVE_DIR, checkpoint_number=epoch + 1)
    
    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        print("All combinations have been learned correctly! Stopping training.")
        break

# --- Final Evaluation ---
final_accuracy = evaluate(model, state.params, x_val, y_val)
print(f"Final accuracy: {final_accuracy:.4f}")

# --- Results Table ---
preds, true_labels = get_predictions(model, state, x_val, y_val)
results = []
for i in range(len(x_val)):
    x1 = x_val[i, 0, 0].item()
    x2 = x_val[i, 0, 1].item()
    y_true = true_labels[i].item()
    y_pred = preds[i].item()
    results.append({"x1": x1, "x2": x2, "y (real)": y_true, "pred": y_pred})
df_results = pd.DataFrame(results)
print(df_results)

# --- Save Everything ---
save_results_and_model(df_results, final_accuracy, state.params, SAVE_DIR)