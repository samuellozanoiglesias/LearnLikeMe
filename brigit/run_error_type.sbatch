#!/bin/bash
#SBATCH --job-name=ERROR-TYPE
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-%x-%j.out
#SBATCH --error=error-%x-%j.err

################################################################################
# Script to generate error distance video for a specific omega value          #
################################################################################

# === PARAMETERS ===
cluster=brigit
number_size=2
study_name=19_STUDY-FIXED_EXP_DECAY_0.05-OMEGA_0.10
param_type=RI                    # Options: RI (random init), WI (wise init)
model_type=argmax                # Options: argmax, vector
omega_value=0.10                # Omega value to analyze
checkpoint=600

PYTHON_SCRIPT="figures_generators/paper_figure_error_type.py"

# Use consistent decimal formatting
export LC_NUMERIC=C

# Ensure logs folder exists
mkdir -p logs

# Change to the project directory
cd /mnt/lustre/home/samuloza/LearnLikeMe

echo "Starting error distance video generation..."
echo "Parameters: cluster=$cluster, number_size=$number_size, study=$study_name"
echo "           param_type=$param_type, model_type=$model_type, omega=$omega_value", "checkpoint=$checkpoint"

# Run the Python script
python3 "$PYTHON_SCRIPT" "$number_size" "$study_name" "$param_type" "$model_type" "$omega_value" "$checkpoint"

echo "Error distance video generation completed for omega=$omega_value"
