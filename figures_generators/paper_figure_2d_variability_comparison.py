# USE: nohup python paper_figure_2d_variability_comparison.py 2 WI argmax STUDY1 STUDY2 > logs_paper_2d_variability_comparison.out 2>&1 &

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
import seaborn as sns
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Config ---
CLUSTER = "brigit"  # Cuenca, Brigit or Local
NUMBER_SIZE = int(sys.argv[1])  # Number of digits in the numbers to be added (2 for two-digit addition)
PARAM_TYPE = str(sys.argv[2]).upper()  # Parameter type for initialization ('WI' for wise initialization or 'RI' for random initialization)
MODEL_TYPE = str(sys.argv[3]).lower()  # 'argmax' or 'vector' version of the decision module
STUDY_NAME_1 = str(sys.argv[4]).upper()  # Name of the first study
STUDY_NAME_2 = str(sys.argv[5]).upper() if len(sys.argv) > 5 else None  # Name of the second study

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

def get_study_dirs(study_name):
    """Get the figures and raw directories for a given study."""
    figures_dir = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/{study_name}"
    raw_dir = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/decision_module/{NUMBER_SIZE}-digit/{study_name}/{PARAM_TYPE}/{MODEL_TYPE}_version"
    return figures_dir, raw_dir

def analyze_multidigit_module(raw_dir, figures_dir, return_data=False):
    os.makedirs(figures_dir, exist_ok=True)

    combined_logs_path = os.path.join(raw_dir, "combined_logs.csv")
    if not os.path.exists(combined_logs_path):
        print(f"No combined_logs.csv found in {raw_dir}")
        if return_data:
            return None
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

    # Custom colormap
    custom_colors = [
    '#7F6BB3',  # violeta sobrio
    '#6A66AA',  # violeta azulado suave
    '#5067A8',  # azul empolvado
    '#3D7FA8',  # azul petróleo suave
    '#3693A3',  # cian desaturado
    '#319882',  # verde agua apagado
    '#3C8E57',  # verde sobrio y profesional
    '#6E9B44',  # verde oliva suave
    '#A8A63A',  # amarillo oliva (serio, no brillante)
    '#C4B635'   # amarillo apagado, tono científico
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom', custom_colors[::-1])  # Reverse for _r effect

    # Scatter the raw points first
    ax.scatter(last_epoch_acc['omega'], last_epoch_acc['epsilon'], last_epoch_acc['error'], c=last_epoch_acc['error'], cmap=custom_cmap, depthshade=True)
    
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
            surf = ax.plot_surface(X, Y, Z, cmap=custom_cmap, edgecolor='none', alpha=0.9)
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
    
    # --- 2D Plot: Averaged Error vs Omega (averaged over all epsilons) ---
    if not last_epoch_acc.empty:
        # Group by omega and calculate mean and std of error across all epsilon values
        omega_stats = last_epoch_acc.groupby('omega')['error'].agg(['mean', 'std']).reset_index()
        omega_stats = omega_stats.sort_values('omega')
        
        fig2, ax2 = plt.subplots(figsize=(12, 8))
                
        # Normalize omega values to [0, 1] for colormap
        norm = Normalize(vmin=omega_stats['omega'].min(), vmax=omega_stats['omega'].max())
        colors = custom_cmap(norm(omega_stats['omega']))
        
        # Plot line segments with gradient colors
        for i in range(len(omega_stats) - 1):
            ax2.plot(omega_stats['omega'].iloc[i:i+2], omega_stats['mean'].iloc[i:i+2],
                    linewidth=3, color=colors[i], alpha=0.9)
        
        # Plot markers with gradient colors
        scatter = ax2.scatter(omega_stats['omega'], omega_stats['mean'], 
                             c=omega_stats['omega'], cmap=custom_cmap, 
                             s=120, zorder=5, edgecolors='white', linewidth=1.5)
        
        # Add shaded regions with gradient colors for each segment
        for i in range(len(omega_stats)):
            if i < len(omega_stats) - 1:
                # Create gradient fill between consecutive points
                ax2.fill_between(omega_stats['omega'].iloc[i:i+2], 
                                omega_stats['mean'].iloc[i:i+2] - omega_stats['std'].iloc[i:i+2], 
                                omega_stats['mean'].iloc[i:i+2] + omega_stats['std'].iloc[i:i+2],
                                alpha=0.25, color=colors[i])
        
        ax2.set_xlabel('Magnitude Noise (ω)', fontsize=24)
        ax2.set_ylabel('Averaged Error (%)', fontsize=24)
        ax2.tick_params(axis='both', labelsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter, ax=ax2, pad=0.02)
        cbar2.set_label('Magnitude Noise (ω)', fontsize=20)
        cbar2.ax.tick_params(labelsize=18)
        
        output_path_2d = os.path.join(figures_dir, f"2d_error_vs_omega_averaged.png")
        plt.savefig(output_path_2d, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"2D figure saved to: {output_path_2d}")
    
    if return_data:
        return omega_stats if not last_epoch_acc.empty else None

def create_comparison_plot(study1_name, study2_name, omega_stats1, omega_stats2, comparison_dir):
    """Create a 2D comparison plot with both studies."""
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define colors for the two studies
    color1 = '#25ae66'  # Light green for study 1
    color2 = '#0f6741'  # Dark green for study 2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Study 1
    if omega_stats1 is not None and not omega_stats1.empty:
        omega_stats1 = omega_stats1.sort_values('omega')
        
        # Plot line with solid style
        ax.plot(omega_stats1['omega'], omega_stats1['mean'],
               linewidth=3, color=color1, alpha=0.9,
               label='Fixed Variability')
        
        # Plot markers
        ax.scatter(omega_stats1['omega'], omega_stats1['mean'], 
                  c=color1, s=120, zorder=5, edgecolors='white', linewidth=1.5)
        
        # Add shaded region
        ax.fill_between(omega_stats1['omega'], 
                       omega_stats1['mean'] - omega_stats1['std'], 
                       omega_stats1['mean'] + omega_stats1['std'],
                       alpha=0.2, color=color1)
    
    # Plot Study 2
    if omega_stats2 is not None and not omega_stats2.empty:
        omega_stats2 = omega_stats2.sort_values('omega')
        
        # Plot line with solid style
        ax.plot(omega_stats2['omega'], omega_stats2['mean'],
               linewidth=3, color=color2, alpha=0.9,
               label='Increasing Variability')
        
        # Plot markers
        ax.scatter(omega_stats2['omega'], omega_stats2['mean'], 
                  c=color2, s=120, zorder=5, edgecolors='white', linewidth=1.5)
        
        # Add shaded region
        ax.fill_between(omega_stats2['omega'], 
                       omega_stats2['mean'] - omega_stats2['std'], 
                       omega_stats2['mean'] + omega_stats2['std'],
                       alpha=0.2, color=color2)
    
    ax.set_xlabel('Magnitude Noise (ω)', fontsize=32)
    ax.set_ylabel('Averaged Error (%)', fontsize=32)
    ax.tick_params(axis='both', labelsize=28)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=30, loc='lower right')
    
    output_path = os.path.join(comparison_dir, f"2d_comparison_{study1_name}_vs_{study2_name}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Comparison figure saved to: {output_path}")

# Main execution
print(f"Processing Study 1: {STUDY_NAME_1}")
figures_dir_1, raw_dir_1 = get_study_dirs(STUDY_NAME_1)
omega_stats_1 = analyze_multidigit_module(raw_dir_1, figures_dir_1, return_data=True)

if STUDY_NAME_2:
    print(f"\nProcessing Study 2: {STUDY_NAME_2}")
    figures_dir_2, raw_dir_2 = get_study_dirs(STUDY_NAME_2)
    omega_stats_2 = analyze_multidigit_module(raw_dir_2, figures_dir_2, return_data=True)

    # Create comparison plot
    comparison_dir = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/figures_paper/comparison_{STUDY_NAME_1}_vs_{STUDY_NAME_2}"
    print(f"\nCreating comparison plot...")
    create_comparison_plot(STUDY_NAME_1, STUDY_NAME_2, omega_stats_1, omega_stats_2, comparison_dir)