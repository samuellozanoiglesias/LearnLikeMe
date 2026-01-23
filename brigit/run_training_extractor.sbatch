#!/bin/bash
#SBATCH --job-name=CARRY_EXTRACTOR_RLLIB
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

################################################################################
# Script to run all omega extractor experiments in parallel inside one SLURM job
################################################################################

# === PARAMETERS ===
cluster=brigit
module_name=unit_extractor       # "unit_extractor" or "carry_extractor"
study_name="21_STUDY-INCREASING_EXP_DECAY_0.00"
fixed_variability=No            # "Yes" or "No"
early_stop=No                    # "Yes" or "No"
training_distribution_type="decreasing_exponential" 
alpha_curriculum=0.00            # Only used if decreasing_exponential

MAX_PARALLEL=24   # Number of concurrent processes (= allocated CPUs)
PYTHON_SCRIPT="../train_extractor_modules.py"

# Use consistent decimal formatting
export LC_NUMERIC=C

# Ensure logs folder exists
mkdir -p logs
# Precompute omegas (0.01 to 0.10)
mapfile -t omegas < <(seq 0.01 0.01 0.01)
#mapfile -t epsilons < <(seq 0.00 0.50 10.00)
epsilons=(None)

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

echo "Launching $total_tasks extractor tasks with up to $MAX_PARALLEL running at the same time"

for epsilon in "${epsilons[@]}"; do
    if [ "$epsilon" = "None" ]; then
        epsilon_fmt="None"
    else
        epsilon_fmt=$(printf "%.2f" $epsilon)
    fi
    for omega in "${omegas[@]}"; do
        omega_fmt=$(printf "%.2f" $omega)
        
        current=$((current + 1))
        percent=$((current * 100 / total_tasks))
        logfile="logs/${module_name}_${study_name}_${param_type}_${model_type}_omega-${omega_fmt}_epsilon-${epsilon_fmt}_${fixed_variability}_${early_stop}_${training_distribution_type}_${alpha_curriculum}.out"

        wait_for_slot

        echo "[$current/$total_tasks] Starting omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}%)"
        python3 "$PYTHON_SCRIPT" "$cluster" "$module_name" "$study_name" "$epsilon" "$omega" "$fixed_variability" "$early_stop" "$training_distribution_type" "$alpha_curriculum" \
            > "$logfile" 2>&1 &
        
        sleep 30
    done
done

# Wait for all to finish
wait
echo "Completed all $total_tasks extractor module tasks."