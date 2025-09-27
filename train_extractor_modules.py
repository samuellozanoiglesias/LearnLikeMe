# USE: nohup python train_extractor_modules.py unit_extractor 0.05 > logs_train_extractor.out 2>&1 &
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
print(jax.devices())  # should only show CPU

import jax.numpy as jnp
from jax import random
import pandas as pd
import os
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from little_learner.modules.extractor_modules.utils import (
    load_dataset, generate_test_dataset,
    create_and_save_initial_params, load_initial_params, generate_train_data, one_hot_encode, save_results_and_module
)
from little_learner.modules.extractor_modules.models import ExtractorModel
from little_learner.modules.extractor_modules.train_utils import (
    load_train_state, evaluate, train_step, get_predictions, compute_loss
)

# --- Config ---
CLUSTER = "cuenca" # Cuenca, Brigit or Local
MODULE_NAME = sys.argv[1].lower()  # unit_extractor or carry_extractor
OMEGA = float(sys.argv[2])  # Weber fraction (~0.2) for gaussian noise, if applicable

# --- Training Parameters ---
LEARNING_RATE = 0.005
PARAMS_FILE = None  # Set to None to create new params, or provide a path to load existing params

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

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}" 
SAVE_DIR = f"{RAW_DIR}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Model Hyperparameters ---
if MODULE_NAME == "carry_extractor":
    num_classes = 2
    FINISH_TOLERANCE = 0.00  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 500  # Carry model uses 500 epochs
    BATCH_SIZE = 50  # Carry model uses batch size of 50
    SHOW_EVERY_N_EPOCHS = 1  # Show accuracy every 1 epochs
    CHECKPOINT_EVERY = 10  # Save checkpoint every 10 epochs
    structure = [16]  # Carry model hidden layer sizes
    output_dim = 2  # Carry model output dimension (carry or no carry)

elif MODULE_NAME == "unit_extractor":
    num_classes = 10
    FINISH_TOLERANCE = 0.00  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 5000  # Unit model uses 6000 epochs
    BATCH_SIZE = 50  # Unit model uses batch size of 10
    SHOW_EVERY_N_EPOCHS = 5  # Show accuracy every 5 epochs
    CHECKPOINT_EVERY = 200  # Save checkpoint every 200 epochs
    structure = [128, 64]  # Unit model hidden layer sizes
    output_dim = 10  # Unit model output dimension (0-9)

else:
    raise ValueError("Invalid module name. Choose 'carry_extractor' or 'unit_extractor'.")


# --- Data Preparation ---
DATASET_DIR = f"{CODE_DIR}/datasets"
single_digit_pairs = load_dataset(os.path.join(DATASET_DIR, "single_digit_additions.txt"))
x, y = generate_train_data(omega=OMEGA, module_name=MODULE_NAME)
x_val, y_val = generate_test_dataset(single_digit_pairs, MODULE_NAME)
x = jnp.array(x, dtype=jnp.float32)
y_one_hot = jnp.array(one_hot_encode(y, num_classes=num_classes), dtype=jnp.float32)
y_val = jnp.array(one_hot_encode(y_val, num_classes=num_classes), dtype=jnp.float32)

x_train = x
y_train = y_one_hot

model = ExtractorModel(structure=structure, output_dim=output_dim)

# --- Save Config File ---
config_path = os.path.join(SAVE_DIR, "config.txt")
with open(config_path, "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Module Name: {MODULE_NAME}\n")
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
    f.write(f"Structure: {structure}\n")
    f.write(f"Output Dim: {output_dim}\n")
    
# --- Model & State ---
if PARAMS_FILE is not None:
    PARAMS_FILE = os.path.join(PARAMS_DIR, PARAMS_FILE)
    initial_params = load_initial_params(PARAMS_FILE)
else:
    PARAMS_FILE = os.path.join(PARAMS_DIR, f"initial_params_{timestamp}.json")
    rng = random.PRNGKey(42)
    input_shape = (1, x_train.shape[1])  # (batch_size, sequence_length, features)
    initial_params = create_and_save_initial_params(model, rng, input_shape, PARAMS_FILE)

state = load_train_state(model, LEARNING_RATE, initial_params)

# --- Training Loop ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True

# Pre-training evaluation: save metrics and a checkpoint for epoch 0
try:
    accuracy = evaluate(model, state.params, x_val, y_val)
    loss = compute_loss(model, state.params, x_val, y_val)

    # Write epoch 0 row to training log
    pd.DataFrame([{
        "epoch": 0,
        "loss": float(loss),
        "accuracy": float(accuracy),
    }]).to_csv(log_path, mode='a', index=False, header=first_write)
    first_write = False

    # Save checkpoint 0 (initial parameters)
    save_results_and_module(df_results=None, final_accuracy=None, model_params=state.params, save_dir=SAVE_DIR, checkpoint_number=0)
    print(f"Saved pre-training checkpoint 0 in {SAVE_DIR}")
except Exception as e:
    print(f"Warning: pre-training evaluation or checkpoint save failed: {e}")


threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))

for epoch in range(EPOCHS):
    for i in range(0, len(x_train), BATCH_SIZE):
        x_batch = x_train[i:i + BATCH_SIZE]
        y_batch = y_train[i:i + BATCH_SIZE]
        state, grads = train_step(state, x_batch, y_batch)
        #print("Epoch", epoch, "some grad example:", jnp.mean(grads['Dense_0']['kernel']))

    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        accuracy = evaluate(model, state.params, x_val, y_val)
        loss = compute_loss(model, state.params, x_val, y_val)

        pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": float(loss),
            "accuracy": float(accuracy)
        }]).to_csv(log_path, mode='a', index=False, header=first_write)
        first_write = False
        
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.params")
        save_results_and_module(df_results=None, final_accuracy=None, model_params=state.params, save_dir=SAVE_DIR, checkpoint_number=epoch + 1)
    
    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        print(f"All combinations have been learned correctly! Epoch: {epoch + 1}.")
        #break

# --- Final Evaluation ---
final_accuracy = evaluate(model, state.params, x_val, y_val)
print(f"Final accuracy: {final_accuracy:.4f}")

# --- Results Table ---
preds, true_labels = get_predictions(model, state, x_val, y_val)
results = []
for i in range(len(x_val)):
    x1, x2 = x_val[i]
    y_true = int(true_labels[i])
    y_pred = int(preds[i])
    results.append({"x1": x1, "x2": x2, "y (real)": y_true, "pred": y_pred})
df_results = pd.DataFrame(results)
print(df_results)

# --- Save Everything ---
save_results_and_module(df_results, final_accuracy, state.params, SAVE_DIR)