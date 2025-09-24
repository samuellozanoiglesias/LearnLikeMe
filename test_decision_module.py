# USE: nohup python test_decision_module.py 0.01 WI > logs_test_decision.out 2>&1 &
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
#print(jax.devices())  # should only show CPU

import jax.numpy as jnp
import pandas as pd
import pickle
import re
import glob
import sys

from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset,
    load_extractor_module, load_decision_module, load_initial_params
)
from little_learner.modules.decision_module.test_utils import (
    predictions, parse_config
)

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit, Local or Lenovo
EPSILON = float(sys.argv[1])  # Noise factor for parameter initialization
PARAM_TYPE = str(sys.argv[2]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)

# --- Paths ---
if CLUSTER.lower() == "cuenca":
    CLUSTER_DIR = ""
    CODE_DIR = f"/home/samuel_lozano/LearnLikeMe"
elif CLUSTER.lower() == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
elif CLUSTER.lower() == "lenovo":
    CLUSTER_DIR = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
elif CLUSTER.lower() == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
FOLDER_DIR = f"{MODULES_DIR}/decision_module/{PARAM_TYPE}/epsilon_{EPSILON:.2f}"
SAVE_DIR = f"{MODULES_DIR}/decision_module/{PARAM_TYPE}/tests"
SAVE_DIR_CHECKPOINTS = f"{SAVE_DIR}/epsilon_{EPSILON:.2f}"
DATASET_DIR = f"{CODE_DIR}/datasets"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_CHECKPOINTS, exist_ok=True)

# --- Data Preparation ---
# Load dataset
all_pairs = load_dataset(os.path.join(DATASET_DIR, "all_valid_additions.txt"))
x_test, y_test = generate_test_dataset(all_pairs)

all_tests = pd.DataFrame()

for foldername in os.listdir(FOLDER_DIR):
    MODEL_DIR = os.path.join(FOLDER_DIR, foldername)

    cfg_path = os.path.join(MODEL_DIR, "config.txt")
    if not os.path.isfile(cfg_path):
        print(f"[WARN] No config.txt in {MODEL_DIR}, skipping.")
        continue

    cfg = parse_config(cfg_path)
    training_id = cfg["training_id"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    omega = cfg.get("omega")
    total_trainings = epochs * batch_size

    filepath = os.path.join(SAVE_DIR_CHECKPOINTS, f"test_{PARAM_TYPE}_{EPSILON}_{training_id}.csv")
    
    unit_module, _ = load_extractor_module(omega, MODULES_DIR, model_type='unit_extractor')
    carry_module, _ = load_extractor_module(omega, MODULES_DIR, model_type='carry_over_extractor')
    
    # Run tests for all checkpoints in the MODEL_DIR (search recursively).
    # We sort by numeric checkpoint id so ordering is natural (1,2,3,...)
    processed_any = False
    checkpoint_pattern = os.path.join(MODEL_DIR, "**", "trained_model_checkpoint_*.pkl")
    checkpoint_paths = glob.glob(checkpoint_pattern, recursive=True)

    # Extract numeric id for sorting; use 0 when not found to keep them at front
    def _ckpt_num(path):
        m = re.search(r"trained_model_checkpoint_(\d+)\.pkl$", path)
        return int(m.group(1)) if m else 0

    checkpoint_paths = sorted(checkpoint_paths, key=_ckpt_num)

    # collect DataFrames for this training
    training_dfs = []
    ckpt_id = 0

    for cp in checkpoint_paths:
        m = re.search(r"trained_model_checkpoint_(\d+)\.pkl$", cp)
        ckpt_id = m.group(1) if m else os.path.basename(cp)

        print(f"Found checkpoint: {ckpt_id} -> {cp}")

        # Load the decision module from the directory that actually contains the checkpoint
        decision_module = load_decision_module(MODEL_DIR, ckpt_id)

        test_data_cp = predictions(decision_module, unit_module, carry_module, x_test, y_test, CODE_DIR)
        test_data_cp["epsilon"] = EPSILON
        test_data_cp["param_type"] = PARAM_TYPE
        test_data_cp["omega"] = omega
        test_data_cp["training_id"] = training_id
        test_data_cp["total_trainings"] = total_trainings
        test_data_cp["checkpoint"] = ckpt_id
        training_dfs.append(test_data_cp)
        processed_any = True    

    # If checkpoints are missing or do not reach the epochs in config, also
    # evaluate the final trained model (trained_model.pkl) and place its
    # results at the last logged epoch and at the configured final epoch.
    if ckpt_id < epochs:
        trained_model_path = os.path.join(MODEL_DIR, "trained_model.pkl")


        # read training_log to find last logged epoch (if present)
        training_log_path = os.path.join(MODEL_DIR, "training_log.csv")
        last_logged_epoch = None
        if os.path.exists(training_log_path):
            try:
                log_df = pd.read_csv(training_log_path)
                if 'epoch' in log_df.columns:
                   last_logged_epoch = int(log_df['epoch'].max())
            except Exception:
                last_logged_epoch = None

        # If trained_model.pkl exists, use it to generate predictions to fill missing slots
        if os.path.exists(trained_model_path):
            with open(trained_model_path, 'rb') as f:
                trained_model = pickle.load(f)

            decision_module = load_decision_module(MODEL_DIR)
            trained_df = predictions(decision_module, unit_module, carry_module, x_test, y_test, CODE_DIR)
            trained_df['epsilon'] = EPSILON
            trained_df['param_type'] = PARAM_TYPE
            trained_df['omega'] = omega
            trained_df['training_id'] = training_id
            trained_df['total_trainings'] = total_trainings

            # Insert at last_logged_epoch if available and missing
            if last_logged_epoch is not None:
                df_copy = trained_df.copy()
                df_copy['checkpoint'] = str(last_logged_epoch)
                training_dfs.append(df_copy)
                processed_any = True

            # Insert at target_epoch (final configured epochs) if missing and different
            if epochs and (last_logged_epoch is None or epochs != last_logged_epoch):
                df_copy2 = trained_df.copy()
                df_copy2['checkpoint'] = str(epochs)
                training_dfs.append(df_copy2)
                processed_any = True

    # Build a per-training CSV including checkpoint 0 (initial parameters) and all checkpoints
    metadata_keys = ["epsilon", "param_type", "omega", "training_id", "total_trainings", "checkpoint"]

    # Attempt to create checkpoint-0 row from initial parameters
    initial_params_dir = os.path.join(MODULES_DIR, 'decision_module', 'initial_parameters')
    init_pattern = os.path.join(initial_params_dir, f"trainable_model_{PARAM_TYPE}_{EPSILON}_{training_id}.json")
    init_paths = glob.glob(init_pattern)
    init_df = None

    if init_paths:
        # pick most recent
        init_path = max(init_paths, key=os.path.getmtime)
        try:
            init_params = load_initial_params(init_path)
            init_pred = predictions(init_params, unit_module, carry_module, x_test, y_test, CODE_DIR)
            init_pred['epsilon'] = EPSILON
            init_pred['param_type'] = PARAM_TYPE
            init_pred['omega'] = omega
            init_pred['training_id'] = training_id
            init_pred['total_trainings'] = total_trainings
            init_pred['checkpoint'] = '0'
            init_df = init_pred
        except Exception as e:
            print(f"Warning: failed to load/run initial params from {init_path}: {e}")

    # If no checkpoints and no init params, warn
    if not processed_any and init_df is None:
        print(f"[WARN] No checkpoints and no initial params found for {MODEL_DIR}.")

    # Determine columns base for building checkpoint-0 row
    if training_dfs:
        first_df = training_dfs[0]
        cols = list(first_df.columns)
    elif init_df is not None:
        cols = list(init_df.columns)
    else:
        cols = metadata_keys

    # Build checkpoint-0 dataframe: prefer init_df if available, otherwise build zero row
    if init_df is not None:
        # reindex to match cols and fill missing with zeros
        zero_df = init_df.reindex(columns=cols).fillna(0)
    else:
        zero_row = {}
        for col in cols:
            if col in metadata_keys:
                if col == 'epsilon':
                    zero_row[col] = EPSILON
                elif col == 'param_type':
                    zero_row[col] = PARAM_TYPE
                elif col == 'omega':
                    zero_row[col] = omega
                elif col == 'training_id':
                    zero_row[col] = training_id
                elif col == 'total_trainings':
                    zero_row[col] = total_trainings
                elif col == 'checkpoint':
                    zero_row[col] = '0'
            else:
                if col.endswith('_accuracy'):
                    zero_row[col] = 0.0
                elif col.endswith('_count'):
                    zero_row[col] = 0
                elif col.endswith('_total'):
                    # keep same totals as in first checkpoint
                    zero_row[col] = first_df[col].iloc[0] if col in first_df.columns else 0
                else:
                    zero_row[col] = 0

        zero_df = pd.DataFrame([zero_row], columns=cols)

    # Sort training_dfs by numeric checkpoint id to keep order
    def _df_ckpt_num(d):
        try:
            return int(d['checkpoint'].iloc[0])
        except Exception:
            return 0

    training_dfs_sorted = sorted(training_dfs, key=_df_ckpt_num)

    # Concatenate zero row + all checkpoints for this training
    combined = pd.concat([zero_df] + training_dfs_sorted, ignore_index=True, sort=False)

    # Reorder columns: metadata first, then the rest preserving original order
    cols_order = metadata_keys + [c for c in combined.columns if c not in metadata_keys]
    combined = combined.reindex(columns=cols_order)

    # Save per-training CSV
    combined.to_csv(filepath, index=False)
    print(f"Test file with all checkpoints created for {training_id} at {filepath}.")

    # Append to global aggregator DataFrame
    if all_tests.empty:
        all_tests = combined.copy()
    else:
        all_tests = pd.concat([all_tests, combined], ignore_index=True, sort=False)
        

if not all_tests.empty:
    results_path = os.path.join(SAVE_DIR, f"tests_{PARAM_TYPE}_{EPSILON:.2f}.csv")
    all_tests.to_csv(results_path, index=False)
    print(f"File {results_path} created.")
else:
    print("No models processed, nothing to write.")