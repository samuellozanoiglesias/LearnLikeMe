# USE: nohup python paper_figure_3d_decision.py 2 STUDY WI argmax > logs_paper_error_decision.out 2>&1 &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
STUDY_NAME = str(sys.argv[2]).upper()  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
PARAM_TYPE = str(sys.argv[3]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[4]).lower()  # 'argmax' or 'vector' version of the decision module

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

FIGURES_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{STUDY_NAME}"
RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{STUDY_NAME}/{PARAM_TYPE}/{MODEL_TYPE}_version"

def analyze_multidigit_module(raw_dir, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        return
    
    # Read CSV with low_memory=False to avoid dtype warning
    combined_logs = pd.read_csv(combined_logs_path, low_memory=False)
    print(f"Combined logs loaded from: {combined_logs_path}")
    print(f"Initial number of rows in combined logs: {len(combined_logs)}")
    
    # Clean and convert data types
    combined_logs['epoch'] = pd.to_numeric(combined_logs['epoch'], errors='coerce')
    combined_logs['accuracy'] = pd.to_numeric(combined_logs['accuracy'], errors='coerce')
    combined_logs['omega'] = pd.to_numeric(combined_logs['omega'], errors='coerce')
    
    # Remove rows with NaN values in critical columns
    valid_combined_logs = combined_logs[combined_logs['epoch'].notna() & combined_logs['accuracy'].notna()]
    if not valid_combined_logs.empty:
        last_epoch_acc = (
            valid_combined_logs.groupby(["omega", "param_init_type", "epsilon"])[["epoch", "accuracy"]]
            .apply(lambda g: g.loc[g["epoch"].idxmax(), "accuracy"] if not g.empty else np.nan)
            .reset_index(name="accuracy")
        )
        last_epoch_acc['error'] = 100 - last_epoch_acc['accuracy'] * 100
        # Remove any rows where accuracy is NaN
        last_epoch_acc = last_epoch_acc[last_epoch_acc['accuracy'].notna()]
    else:
        print("No valid epoch/accuracy data found for final analysis")
        last_epoch_acc = pd.DataFrame()
    
    print(f"After cleaning: {len(combined_logs)} rows remaining")
    
    # Create a DataFrame that EXCLUDES epsilon = 0
    last_epoch_acc = last_epoch_acc[last_epoch_acc['epsilon'] != 0].copy()
    
    # Prepare figure and 3D axes
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter the raw points first
    ax.scatter(last_epoch_acc['omega'], last_epoch_acc['epsilon'], last_epoch_acc['error'], c=last_epoch_acc['error'], cmap='viridis_r', depthshade=True)
    
    # Try to build a regular grid (omega x epsilon) for surface plotting
    try:
        omegas = np.array(sorted(last_epoch_acc['omega'].dropna().unique()))
        epss = np.array(sorted(last_epoch_acc['epsilon'].dropna().unique()))

        if len(omegas) > 1 and len(epss) > 1:
            # pivot to grid of accuracies
            pivot = last_epoch_acc.pivot_table(index='omega', columns='epsilon', values='error')
            # ensure full index/columns in sorted order
            pivot = pivot.reindex(index=omegas, columns=epss)

            # Interpolate missing values along each axis when possible
            pivot = pivot.sort_index()
            pivot = pivot.interpolate(axis=0, limit_direction='both').interpolate(axis=1, limit_direction='both')
            # if still NaNs at edges, fill with nearest
            pivot = pivot.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
            pivot = pivot.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

            Z = pivot.values
            X, Y = np.meshgrid(omegas, epss, indexing='ij')

            # plot surface (transpose or orientation should match pivot shape)
            surf = ax.plot_surface(X, Y, Z, cmap='viridis_r', edgecolor='none', alpha=0.7)
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, ticks=[0, 20, 40, 60, 80, 100])
            cbar.ax.tick_params(labelsize=26)
        else:
            # Not enough distinct omegas/epsilons for a surface — keep scatter only
            pass
    except Exception as e:
        print(f"Could not draw surface: {e}")
    
    ax.set_xlim(last_epoch_acc['omega'].max(), last_epoch_acc['omega'].min())
    ax.set_ylim(0.01, last_epoch_acc['epsilon'].max())
    ax.set_zlim(0, 101)

    ax.set_xlabel('Magnitude Noise (ω)', fontsize=30)
    ax.set_ylabel('Initialization (ε)', fontsize=30)
    ax.set_zlabel('Averaged Error (%)', fontsize=30)

    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    ax.tick_params(axis='z', labelsize=26)

    ax.xaxis.labelpad = 20 # Aumenta el espacio entre el eje X y su label
    ax.yaxis.labelpad = 20 # Aumenta el espacio entre el eje Y y su label
    ax.zaxis.labelpad = 20 # Aumenta el espacio entre el eje Z y su label

    output_path = os.path.join(figures_dir, f"3d_accuracy_vs_omega_epsilon.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")

analyze_multidigit_module(RAW_DIR, FIGURES_DIR)