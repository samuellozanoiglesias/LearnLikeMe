# USE: nohup python train_extractor_modules.py unit_extractor THIRD_STUDY-NO_AVERAGED_OMEGA 0.05 No Yes Decreasing_exponential 0.1 > logs_train_extractor.out 2>&1 &
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
    create_and_save_initial_params, load_initial_params, generate_batch_data, one_hot_encode, save_results_and_module
)
from little_learner.modules.extractor_modules.models import ExtractorModel
from little_learner.modules.extractor_modules.train_utils import (
    load_train_state, evaluate, train_step, get_predictions, compute_loss
)

# --- Config ---
CLUSTER = "cuenca" # Cuenca, Brigit or Local
MODULE_NAME = sys.argv[1].lower()  # unit_extractor or carry_extractor
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
OMEGA = float(sys.argv[3])  # Weber fraction (~0.2) for gaussian noise, if applicable
FIXED_VARIABILITY = len(sys.argv) > 4 and sys.argv[4].lower() in ['yes', 'true', '1']  # Fixed variability flag (Yes/No)
EARLY_STOP = len(sys.argv) > 5 and sys.argv[5].lower() in ['yes', 'true', '1']  # Early stopping flag (Yes/No)
TRAINING_DISTRIBUTION_TYPE = str(sys.argv[6]).lower() if len(sys.argv) > 6 else "none"  # Use curriculum learning for training (Decreasing_exponential or Balanced)
ALPHA_CURRICULUM = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1  # Only used if TRAINING_DISTRIBUTION_TYPE is "decreasing_exponential"

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

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}/{STUDY_NAME}" 
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
    BATCH_SIZE = 25 
    SHOW_EVERY_N_EPOCHS = 1  # Show accuracy every 1 epochs
    CHECKPOINT_EVERY = 10  # Save checkpoint every 10 epochs
    structure = [16]  # Carry model hidden layer sizes
    output_dim = 2  # Carry model output dimension (carry or no carry)

elif MODULE_NAME == "unit_extractor":
    num_classes = 10
    FINISH_TOLERANCE = 0.00  # Tolerance for stopping training when accuracy reaches 1.0
    EPOCHS = 5000  # Unit model uses 5000 epochs
    BATCH_SIZE = 25
    SHOW_EVERY_N_EPOCHS = 5  # Show accuracy every 5 epochs
    CHECKPOINT_EVERY = 200  # Save checkpoint every 200 epochs
    structure = [128, 64]  # Unit model hidden layer sizes
    output_dim = 10  # Unit model output dimension (0-9)

else:
    raise ValueError("Invalid module name. Choose 'carry_extractor' or 'unit_extractor'.")


# --- Data Preparation ---
DATASET_DIR = f"{CODE_DIR}/datasets"
single_digit_pairs = load_dataset(os.path.join(DATASET_DIR, "single_digit_additions.txt"))

# Generate all possible training pairs for curriculum learning
all_train_pairs = [(a, b) for a in range(10) for b in range(10)]

# Generate validation data
x_val, y_val = generate_test_dataset(single_digit_pairs, MODULE_NAME)
y_val = jnp.array(one_hot_encode(y_val, num_classes=num_classes), dtype=jnp.float32)

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
    f.write(f"Fixed Variability: {'Yes' if FIXED_VARIABILITY else 'No'}\n")
    f.write(f"Early Stop: {'Yes' if EARLY_STOP else 'No'}\n")
    f.write(f"Training Distribution Type: {TRAINING_DISTRIBUTION_TYPE}\n")
    f.write(f"Alpha for curriculum learning: {ALPHA_CURRICULUM}\n")
    f.write(f"Finish Tolerance: {FINISH_TOLERANCE}\n")
    f.write(f"Epoch Size: {EPOCH_SIZE}\n")
    f.write(f"Batches per epoch: {max(1, EPOCH_SIZE // BATCH_SIZE)}\n")
    f.write(f"Training pairs available: {len(all_train_pairs)}\n")
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
    input_shape = (1, 2)  # (batch_size, 2 features for a and b)
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

# Calculate number of batches per epoch based on EPOCH_SIZE
batches_per_epoch = max(1, EPOCH_SIZE // BATCH_SIZE)  # At least 1 batch per epoch

# Initialize df_results for potential early stopping
df_results = None

for epoch in range(EPOCHS):
    # Generate training batches dynamically with curriculum learning
    for batch_idx in range(batches_per_epoch):
        # Generate batch with curriculum learning
        x_batch, y_batch = generate_batch_data(
            all_train_pairs, BATCH_SIZE, omega=OMEGA, 
            seed=epoch * batches_per_epoch + batch_idx,  # Unique seed per batch
            module_name=MODULE_NAME, fixed_variability=FIXED_VARIABILITY,
            distribution=TRAINING_DISTRIBUTION_TYPE, alpha=ALPHA_CURRICULUM
        )
        
        # Convert to proper format
        x_batch = jnp.array(x_batch, dtype=jnp.float32)
        y_batch_one_hot = jnp.array(one_hot_encode(y_batch, num_classes=num_classes), dtype=jnp.float32)
        
        state, grads = train_step(state, x_batch, y_batch_one_hot)
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
        if EARLY_STOP:        
            # Save last epoch info for filling remaining epochs
            last_epoch = epoch + 1
            last_metrics = {
                "epoch": None,  # will be set in loop
                "loss": float(loss),
                "accuracy": float(accuracy),
            }

            # Generate final results table for early stopping
            try:
                preds, true_labels = get_predictions(model, state, x_val, y_val)
                results = []
                for i in range(len(x_val)):
                    x1, x2 = x_val[i]
                    y_true = int(true_labels[i])
                    y_pred = int(preds[i])
                    results.append({"x1": x1, "x2": x2, "y (real)": y_true, "pred": y_pred})
                df_results = pd.DataFrame(results)
            except Exception as e:
                print(f"[ERROR] Failed to generate results table during early stopping: {e}")
                df_results = None

            # Save final checkpoint with current parameters
            try:
                save_results_and_module(df_results=df_results, final_accuracy=accuracy, model_params=state.params, save_dir=SAVE_DIR, checkpoint_number=last_epoch)
                print(f"Saved final checkpoint {last_epoch} after achieving target accuracy in {SAVE_DIR}")
            except Exception as e:
                print(f"[ERROR] Failed to save final checkpoint {last_epoch}: {e}")

            # Fill remaining epochs with last metrics
            for fill_epoch in range(last_epoch + SHOW_EVERY_N_EPOCHS - (last_epoch % SHOW_EVERY_N_EPOCHS), EPOCHS + 1, SHOW_EVERY_N_EPOCHS):
                last_metrics["epoch"] = fill_epoch
                pd.DataFrame([last_metrics]).to_csv(log_path, mode='a', index=False, header=False)

            break

# --- Final Evaluation (only if training completed without early stopping) ---
if df_results is None:  # Only run if we didn't break early
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
else:
    print("Training stopped early due to achieving target accuracy.")
    print(df_results)
print('Training complete.')