import jax
import jax.numpy as jnp
import pandas as pd
import os
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from little_learner.modules.decision_module.utils import (
    load_dataset, generate_train_dataset, generate_test_dataset,
    save_results_and_module, load_extractor_module, create_and_save_decision_params, load_initial_params
)
from little_learner.modules.decision_module.train_utils import (
    evaluate_module, update_params
)

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
EPSILON = float(sys.argv[1])  # Noise factor for parameter initialization
PARAM_TYPE = str(sys.argv[2]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)

# --- Training Parameters ---
OMEGA_UNIT = 0.05  # Omega value for loading pre-trained unit_extractor
OMEGA_CARRY = 0.05  # Omega value for loading pre-trained carry_over_extractor
LEARNING_RATE = 0.01
EPOCHS = 10000
BATCH_SIZE = 50
FINISH_TOLERANCE = 0.0  # Tolerance for stopping training when accuracy reaches 1.0
SHOW_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY = 2000
PARAMS_FILE = None  # Set to None to create new params, or provide a path to load existing params
TRAINING_DISTRIBUTION_TYPE = "Decreasing_exponential"  # Use curriculum learning for training

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

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/epsilon_{EPSILON:.2f}/Training_{timestamp}"
PARAMS_DIR = f"{RAW_DIR}/initial_parameters"
MODELS_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
DATASET_DIR = f"datasets"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Data Preparation ---
# Load datasets and extractors
train_pairs = load_dataset(os.path.join(DATASET_DIR, "train_pairs_not_in_stimuli.txt"))
test_pairs = load_dataset(os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
carry_module = load_extractor_module(OMEGA_CARRY, MODELS_DIR, model_type='carry_over_extractor')
unit_module = load_extractor_module(OMEGA_UNIT, MODELS_DIR, model_type='unit_extractor')

# Generate initial test dataset
x_test, y_test = generate_test_dataset(test_pairs)

# --- Save Config File ---
config_path = os.path.join(SAVE_DIR, "config.txt")
with open(config_path, "w") as f:
    f.write(f"Training ID: {timestamp}\n")
    f.write(f"Cluster Directory: {CLUSTER if CLUSTER else ''}\n")
    f.write(f"Module Name: decision_module\n")
    f.write(f"Parameter type (Wise initialization or Random initialization): {PARAM_TYPE}\n")
    f.write(f"Noise for initialization parameters: {EPSILON}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Parameters File: {PARAMS_FILE if PARAMS_FILE else 'New parameters generated'}\n")
    f.write(f"Weber fraction for Unit Extractor (Omega): {OMEGA_UNIT}\n")
    f.write(f"Weber fraction for Carry Extractor (Omega): {OMEGA_CARRY}\n")
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
    params = create_and_save_decision_params(PARAMS_DIR, epsilon=EPSILON, param_type=PARAM_TYPE, timestamp=timestamp)

# --- Training Loop ---
log_path = os.path.join(SAVE_DIR, "training_log.csv")
first_write = True
threshold = Decimal('1.0') - Decimal(str(FINISH_TOLERANCE))

for epoch in range(EPOCHS):
    # Generate training batch with curriculum learning
    x_train, y_train = generate_train_dataset(train_pairs, BATCH_SIZE, distribution=TRAINING_DISTRIBUTION_TYPE)
    
    # Update parameters
    params = update_params(params, x_train, y_train, unit_module, carry_module, LEARNING_RATE)
    
    if (epoch + 1) % SHOW_EVERY_N_EPOCHS == 0 or epoch == 0:
        # Evaluate on test set
        pred_count, pred_count_test, loss = evaluate_module(
            params, x_test, y_test, unit_module, carry_module, test_pairs
        )
        
        accuracy = pred_count_test / len(test_pairs)
        
        # Log results
        pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "total_correct": pred_count,
            "test_correct": pred_count_test
        }]).to_csv(log_path, mode='a', index=False, header=first_write)
        first_write = False
        
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
              f"Correct: {pred_count}/{len(x_test)}, Test Correct: {pred_count_test}/{len(test_pairs)}")
        
    # Save periodic checkpoint
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        save_results_and_module(None, accuracy, params, SAVE_DIR, checkpoint_number=epoch + 1)
    
    # Early stopping check
    accuracy_dec = Decimal(str(accuracy)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if accuracy_dec >= threshold:
        print(f"Reached target accuracy of {threshold}! Stopping training.")
        break

# --- Final Evaluation ---
final_pred_count, final_pred_count_test, final_loss = evaluate_module(
    params, x_test, y_test, unit_module, carry_module, test_pairs
)
final_accuracy = final_pred_count_test / len(test_pairs)
print(f"\nTraining completed:")
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Accuracy: {final_accuracy:.4f}")
print(f"Total Correct: {final_pred_count}/{len(x_test)}")
print(f"Test Set Correct: {final_pred_count_test}/{len(test_pairs)}")

# --- Results Table ---
# Get predictions for all test examples
pred_count, pred_count_test, loss, final_preds = evaluate_module(params, x_test, y_test, unit_module, carry_module, test_pairs, return_predictions=True)
results = []
for i in range(len(test_pairs)):
    x1, x2 = test_pairs[i]
    y_true = y_test[i, 0] * 10 + y_test[i, 1]
    y_pred = final_preds[i].item()
    results.append({
        "x1": x1, 
        "x2": x2, 
        "y (true)": y_true, 
        "y (pred)": y_pred,
        "correct": y_pred == y_true
    })
df_results = pd.DataFrame(results)

# --- Save Final Model ---
save_results_and_module(df_results, final_accuracy, params, SAVE_DIR)
