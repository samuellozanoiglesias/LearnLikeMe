#!/bin/bash
#SBATCH --job-name=FIXED-DECISION_RLLIB-RI
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

################################################################################
# Script to run all omega-epsilon combinations in parallel inside a single job #
################################################################################

# === PARAMETERS ===
cluster=brigit
number_size=2
study_name=19_STUDY-FIXED_EXP_DECAY_0.00-OMEGA_0.10
param_type=RI
model_type=argmax # Options: argmax, vector
epochs=2000
batch_size=100
epoch_size=1000
fixed_variability=Yes
training_distribution_type=decreasing_exponential # Options: none, decreasing_exponential, balanced
alpha_curriculum=0.00

MAX_PARALLEL=24   # Number of parallel python processes (use all allocated CPUs)
PYTHON_SCRIPT="../train_decision_module.py"

# Use consistent decimal formatting
export LC_NUMERIC=C

# Ensure logs folder exists
mkdir -p logs

# Define omega and epsilon ranges
mapfile -t omegas < <(seq 0.10 0.01 0.10)
mapfile -t epsilons < <(seq 0.00 0.50 10.00)

total_tasks=$(( ${#omegas[@]} * ${#epsilons[@]} ))
current=0

wait_for_slot() {
    local running
    while true; do
        running=$(jobs -r | wc -l)
        [[ $running -lt $MAX_PARALLEL ]] && break
        sleep 5
    done
}

echo "Launching $total_tasks tasks with up to $MAX_PARALLEL running at the same time"

for epsilon in "${epsilons[@]}"; do
    epsilon_fmt=$(printf "%.2f" $epsilon)
    for omega in "${omegas[@]}"; do
        omega_fmt=$(printf "%.2f" $omega)
        
        current=$((current + 1))
        percent=$((current * 100 / total_tasks))
        logfile="logs/${number_size}_${study_name}_${param_type}_${model_type}_omega-${omega_fmt}_epsilon-${epsilon_fmt}.out"

        wait_for_slot

        echo "[$current/$total_tasks] Starting omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}%)"
        python3 "$PYTHON_SCRIPT" "$cluster" "$number_size" "$study_name" "$param_type" "$model_type" "$epsilon" "$omega" "$epochs" "$batch_size" "$epoch_size" "$fixed_variability" "$training_distribution_type" "$alpha_curriculum" \
            > "$logfile" 2>&1 &
        
        sleep 30
    done
done

# Wait for all to finish
wait
echo "All $total_tasks tasks completed."
