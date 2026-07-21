#!/bin/bash
# Use: nohup bash automatic-train_decision_module.sh > logs_decision.out 2>&1 &

cluster=cuenca
number_size=2  # Number of digits in the numbers to be added (2 for two-digit addition)
study_name=NEW_STUDY  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
param_type=RI  # "WI" for wise initialization or "RI" for random initialization
model_type=argmax  # "argmax" or "vector" or "straight_through"
epochs=10000  # Number of training epochs
batch_size=25  # Batch size for training
epoch_size=100 # Number of examples per epoch
fixed_variability=Yes  # "Yes" or "No" to indicate if fixed variability is used
training_distribution_type=Decreasing_exponential  # "Decreasing_exponential" or "Balanced"
alpha_curriculum=0.05  # Only used if training_distribution_type is "Decreasing_exponential"
training_mode=decision_only # "decision_only" (freeze extractors, original behavior) or "all" (unfreeze + jointly train extractors too)

MAX_PARALLEL=10  # Maximum number of parallel simulations

# Forzar punto decimal para seq
export LC_NUMERIC=C

# Carpeta padre donde está el script Python
PYTHON_SCRIPT="../train_decision_module.py"

# Crear carpeta logs si no existe
mkdir -p logs

# Build arrays of omegas and epsilons and compute totals
init_omegas=0.10
end_omegas=0.10
step_omegas=0.05

init_epsilons=0.50
end_epsilons=0.50
step_epsilons=0.05

mapfile -t omegas < <(seq $init_omegas $step_omegas $end_omegas)
mapfile -t epsilons < <(seq $init_epsilons $step_epsilons $end_epsilons)
omega_count=${#omegas[@]}
epsilon_count=${#epsilons[@]}
total_tasks=$(( omega_count * epsilon_count ))

# Function to wait for available slot (global across all script instances)
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
}

# Launch each individual task with slot control
for e_idx in "${!epsilons[@]}"; do
    epsilon="${epsilons[$e_idx]}"
    epsilon_fmt=$(printf "%.2f" $epsilon)

    for o_idx in "${!omegas[@]}"; do
        omega="${omegas[$o_idx]}"
        omega_fmt=$(printf "%.2f" $omega)
        current=$(( o_idx * epsilon_count + e_idx + 1 ))
        percent=$(( current * 100 / total_tasks ))

        # Wait for an available slot before each individual task
        wait_for_slot

        sleep 5  # Small delay to stagger starts

        echo "Running task $current/$total_tasks — omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}% complete overall)"

        nohup python3 "$PYTHON_SCRIPT" "$cluster" "$number_size" "$study_name" "$param_type" "$model_type" "$epsilon" "$omega" "$epochs" "$batch_size" "$epoch_size" "$fixed_variability" "$training_distribution_type" "$alpha_curriculum" "$training_mode" > "logs/decision_module_${number_size}_${study_name}_${param_type}_${model_type}_omega-${omega_fmt}_epsilon-${epsilon_fmt}_epochs-${epochs}_batch-${batch_size}_epochsize-${epoch_size}_fixed-${fixed_variability}_dist-${training_distribution_type}_alpha-${alpha_curriculum}_training_mode-${training_mode}.out" 2>&1 &

        echo "Started task $current/$total_tasks — omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}% complete overall)"
    done
done

# Esperar a que terminen todos los procesos en paralelo
wait
echo "All omega-epsilon experiments finished! (100% complete)"