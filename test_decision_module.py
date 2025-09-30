# USE: nohup python test_decision_module.py 2 FOURTH_STUDY WI argmax 0.10 > logs_test_decision.out 2>&1 &
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
print(jax.devices())  # should only show CPU

import jax.numpy as jnp
import pandas as pd
import pickle
import re
import glob
import sys

from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset,
    load_extractor_module, load_decision_module, load_initial_params,
    _parse_structure
)
from little_learner.modules.decision_module.test_utils import (
    predictions, parse_config
)
from little_learner.modules.decision_module.model import decision_model_argmax, decision_model_vector

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit, Local or Lenovo
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower() # "argmax" for argmax outputs, "vector" for probability vector outputs
EPSILON = float(sys.argv[5])  # Noise factor for parameter initialization

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

DATASET_DIR = f"{CODE_DIR}/datasets/{NUMBER_SIZE}-digit"
MODULES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe"
RAW_DIR = f"{MODULES_DIR}/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}"
FOLDER_DIR = f"{RAW_DIR}/{PARAM_TYPE}/{MODEL_TYPE}_version/epsilon_{EPSILON:.2f}"
SAVE_DIR = f"{RAW_DIR}/{PARAM_TYPE}/{MODEL_TYPE}_version/tests"
SAVE_DIR_CHECKPOINTS = f"{SAVE_DIR}/epsilon_{EPSILON:.2f}"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_CHECKPOINTS, exist_ok=True)

# --- Data Preparation ---
# Load dataset
all_pairs = load_dataset(os.path.join(DATASET_DIR, "all_valid_additions.txt"))
x_test, y_test = generate_test_dataset(all_pairs)

# Select model function
if MODEL_TYPE == "vector":
    model_fn = decision_model_vector
elif MODEL_TYPE == "argmax":
    model_fn = decision_model_argmax
else:
    raise ValueError("Invalid model type. Choose 'argmax' or 'vector'.")

# --- Run tests ---
all_tests = pd.DataFrame()

for foldername in os.listdir(FOLDER_DIR):
    MODEL_DIR = os.path.join(FOLDER_DIR, foldername)

    cfg_path = os.path.join(MODEL_DIR, "config.txt")
    if not os.path.isfile(cfg_path):
        print(f"[WARN] No config.txt in {MODEL_DIR}, skipping.")
        continue

    cfg = parse_config(cfg_path)
    omega = cfg.get("omega")
    training_id = cfg["training_id"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    checkpoint_every = cfg["checkpoint_every"]
    total_trainings = epochs * batch_size

    filepath = os.path.join(SAVE_DIR_CHECKPOINTS, f"test_{PARAM_TYPE}_{EPSILON}_{training_id}.csv")
    
    carry_module, carry_dir, carry_structure = load_extractor_module(omega, MODULES_DIR, model_type='carry_extractor', study_name=STUDY_NAME)
    unit_module, unit_dir, unit_structure = load_extractor_module(omega, MODULES_DIR, model_type='unit_extractor', study_name=STUDY_NAME)

    carry_structure = _parse_structure(carry_structure)
    unit_structure = _parse_structure(unit_structure)
    
    # Run tests for all checkpoints in the MODEL_DIR (search recursively).
    # We sort by numeric checkpoint id so ordering is natural (1,2,3,...)
    processed_any = False
    checkpoint_pattern = os.path.join(MODEL_DIR, "**", "trained_model_checkpoint_*.pkl")
    checkpoint_paths = glob.glob(checkpoint_pattern, recursive=True)
    expected_checkpoints = [str(e) for e in range(checkpoint_every, epochs + 1, checkpoint_every)]

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

        test_data_cp = predictions(decision_module, unit_module, carry_module, x_test, y_test, CODE_DIR, 
                                   unit_structure=unit_structure, carry_structure=carry_structure,
                                   model_fn=model_fn, dataset_dir=DATASET_DIR)
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
    if int(ckpt_id) < epochs:
        trained_model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
        present_checkpoints = set(str(df['checkpoint'].iloc[0]) for df in training_dfs)

        # If trained_model.pkl exists, use it to generate predictions to fill missing slots
        if os.path.exists(trained_model_path):
            with open(trained_model_path, 'rb') as f:
                trained_model = pickle.load(f)

            decision_module = load_decision_module(MODEL_DIR)
            trained_df = predictions(decision_module, unit_module, carry_module, x_test, y_test, CODE_DIR, 
                                     unit_structure=unit_structure, carry_structure=carry_structure,
                                     model_fn=model_fn, dataset_dir=DATASET_DIR)
            trained_df['epsilon'] = EPSILON
            trained_df['param_type'] = PARAM_TYPE
            trained_df['omega'] = omega
            trained_df['training_id'] = training_id
            trained_df['total_trainings'] = total_trainings

            for ckpt in expected_checkpoints:
                if ckpt not in present_checkpoints:
                    df_copy = trained_df.copy()
                    df_copy['checkpoint'] = ckpt
                    training_dfs.append(df_copy)
                    processed_any = True

            # Insert at final configured epoch if missing
            if epochs and (str(epochs) not in present_checkpoints):
                df_copy2 = trained_df.copy()
                df_copy2['checkpoint'] = str(epochs)
                training_dfs.append(df_copy2)
                processed_any = True

    # Build a per-training CSV including checkpoint 0 (initial parameters) and all checkpoints
    metadata_keys = ["epsilon", "param_type", "omega", "training_id", "total_trainings", "checkpoint"]

    # Check if checkpoint=0 was already processed
    checkpoint_0_exists = any(
        (df['checkpoint'].iloc[0] == '0' or df['checkpoint'].iloc[0] == 0)
        for df in training_dfs
    )

    # Attempt to create checkpoint-0 row from initial parameters only if not present
    zero_df = None
    if not checkpoint_0_exists:
        initial_params_dir = os.path.join(RAW_DIR, 'initial_parameters')
        init_pattern = os.path.join(initial_params_dir, f"trainable_model_{PARAM_TYPE}_{EPSILON}_{training_id}.json")
        init_paths = glob.glob(init_pattern)
        init_df = None

        if init_paths:
            # pick most recent
            init_path = max(init_paths, key=os.path.getmtime)
            try:
                init_params = load_initial_params(init_path)
                init_pred = predictions(init_params, unit_module, carry_module, x_test, y_test, CODE_DIR, 
                                        unit_structure=unit_structure, carry_structure=carry_structure,
                                        model_fn=model_fn, dataset_dir=DATASET_DIR)
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

        if init_df is not None:
            zero_df = init_df.reindex(columns=cols).fillna(0)
            zero_df = zero_df.infer_objects(copy=False)

    # Sort training_dfs by numeric checkpoint id to keep order
    def _df_ckpt_num(d):
        try:
            return int(d['checkpoint'].iloc[0])
        except Exception:
            return 0

    training_dfs_sorted = sorted(training_dfs, key=_df_ckpt_num)

    # Filter out empty or all-NA DataFrames before concatenation
    def _is_valid_df(df):
        return df is not None and not df.empty and not df.isna().all(axis=None)

    valid_dfs = [df for df in training_dfs_sorted if _is_valid_df(df)]

    if zero_df is not None and _is_valid_df(zero_df):
        combined = pd.concat([zero_df] + valid_dfs, ignore_index=True, sort=False)
    else:
        combined = pd.concat(valid_dfs, ignore_index=True, sort=False)

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
    # Sort by omega, then by checkpoint (as integer)
    def _ckpt_int(val):
        try:
            return int(val)
        except Exception:
            return 0
    all_tests_sorted = all_tests.copy()
    if 'omega' in all_tests_sorted.columns and 'checkpoint' in all_tests_sorted.columns:
        all_tests_sorted = all_tests_sorted.sort_values(
            by=['omega', 'checkpoint'],
            key=lambda col: col if col.name != 'checkpoint' else col.map(_ckpt_int),
            ascending=[True, True]
        )
    results_path = os.path.join(SAVE_DIR, f"tests_{PARAM_TYPE}_{EPSILON:.2f}.csv")
    all_tests_sorted.to_csv(results_path, index=False)
    print(f"File {results_path} created.")
else:
    print("No models processed, nothing to write.")