import jax
import jax.numpy as jnp
import pandas as pd
import os
import sys
from datetime import datetime

from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset,
    load_extractor_module, load_decision_module
)
from little_learner.modules.decision_module.test_utils import (
    predictions, parse_config
)

# --- Config ---
CLUSTER = "lenovo"  # Cuenca, Brigit, Local or Lenovo
EPSILON = float(sys.argv[1])  # Noise factor for parameter initialization
PARAM_TYPE = sys.argv[2]  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)

# --- Paths ---
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if CLUSTER.lower() == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER.lower() == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER.lower() == "lenovo":
    CLUSTER_DIR = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
elif CLUSTER.lower() == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', 'local' or 'lenovo'.")

MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
FOLDER_DIR = f"{MODULES_DIR}/decision_module/{PARAM_TYPE}/epsilon_{EPSILON:.2f}"
SAVE_DIR = f"{CLUSTER_DIR}/Tests"
DATASET_DIR = f"datasets"

# --- Data Preparation ---
# Load dataset
test_pairs = load_dataset(os.path.join(DATASET_DIR, "stimuli_test_pairs.txt"))
x_test, y_test = generate_test_dataset(test_pairs)

os.makedirs(SAVE_DIR, exist_ok=True) 
results_file = os.path.join(SAVE_DIR, f"Tests_{PARAM_TYPE}_{EPSILON}.csv")
all_tests = [] 

for foldername in os.listdir(FOLDER_DIR):
    MODEL_DIR = os.path.join(FOLDER_DIR, foldername)

    cfg_path = os.path.join(MODEL_DIR, "config.txt")
    if not os.path.isfile(cfg_path):
        print(f"[WARN] No config.txt in {MODEL_DIR}, skipping.")
        continue

    cfg = parse_config(cfg_path)
    training_id = cfg["training_id"]
    epochs      = cfg["epochs"]
    batch_size  = cfg["batch_size"]
    omega_unit  = cfg["omega_unit"]
    omega_carry = cfg["omega_carry"]
    total_trainings = epochs * batch_size

    filepath = os.path.join(SAVE_DIR, f"Test_{PARAM_TYPE}_{EPSILON}_{training_id}.csv")
    
    unit_module = load_extractor_module(omega_unit, MODULES_DIR, model_type='unit_extractor')
    carry_module = load_extractor_module(omega_carry, MODULES_DIR, model_type='carry_over_extractor')
    decision_module = load_decision_module(MODEL_DIR)
    test_data = predictions(decision_module, unit_module, carry_module, x_test, y_test, CLUSTER_DIR)

    test_data["training_id"] = training_id
    test_data["total_trainings"] = total_trainings
    test_data["epsilon"] = EPSILON
    test_data["param_type"] = PARAM_TYPE
    test_data["timestamp"] = timestamp

    test_data.to_csv(filepath, index=False)
    print(f"Test obtained for {training_id}.")

    all_tests.append(test_data)

if all_tests:
    final_df = pd.concat(all_tests, ignore_index=True)
    results_file = os.path.join(SAVE_DIR, f"Tests_{PARAM_TYPE}_{EPSILON:.2f}.xlsx")
    final_df.to_excel(results_file, index=False)
    print(f"File {results_file} created.")
else:
    print("No models processed, nothing to write.")