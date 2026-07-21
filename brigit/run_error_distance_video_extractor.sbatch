#!/bin/bash
#SBATCH --job-name=ERROR-DISTANCE-VIDEO
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
study_type=unit
study_name=20_STUDY-INCREASING_EXP_DECAY_0.00
omega_value=0.06                # Omega value to analyze

PYTHON_SCRIPT="figures_generators/paper_figure_error_distance_extractor_video.py"

# Use consistent decimal formatting
export LC_NUMERIC=C

# Ensure logs folder exists
mkdir -p logs

# Change to the project directory
cd /mnt/lustre/home/samuloza/LearnLikeMe

echo "Starting error distance video generation..."
echo "Parameters: cluster=$cluster, study=$study_name"
echo "           study_type=$study_type, omega=$omega_value"

# Run the Python script
python3 "$PYTHON_SCRIPT" "$study_type" "$study_name" "$omega_value"

echo "Error distance video generation completed for omega=$omega_value"
