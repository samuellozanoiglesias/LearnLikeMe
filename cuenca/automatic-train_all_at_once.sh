#!/bin/bash
# Use: nohup bash automatic-train_all_at_once.sh > logs_all_at_once.out 2>&1 &

cluster=cuenca
number_size=2  # Number of digits in the numbers to be added (2 for two-digit addition)
study_name=NEW_STUDY  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
param_type=RI  # "WI" for wise initialization or "RI" for random initialization (decision layer only)
model_type=argmax  # "vector" or "argmax" or "straight_through"
epochs=10000  # Number of training epochs
batch_size=25  # Batch size for training
epoch_size=100 # Number of examples per epoch
fixed_variability=Yes  # "Yes" or "No" to indicate if fixed variability is used
training_distribution_type=Decreasing_exponential  # "Decreasing_exponential" or "Balanced"
alpha_curriculum=0.05  # Only used if training_distribution_type is "Decreasing_exponential"
early_stop=No  # "Yes" to stop as soon as target accuracy is hit, "No" to keep training the full epoch budget

MAX_PARALLEL=10  # Maximum number of parallel simulations

# Forzar punto decimal para seq
export LC_NUMERIC=C

# Carpeta padre donde está el script Python
PYTHON_SCRIPT="../train_all_at_once.py"

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

# NOTE: train_all_at_once.py takes TWO epsilons -- one for the decision layer
# (epsilon_decision) and one for carry_module/unit_module init
# (epsilon_extractor), since unlike train_decision_module.py it doesn't load
# pretrained extractors. This sweep uses the same swept value for both, so
# each (omega, epsilon) task still means "one config" like in the
# step-by-step launcher. If you want to sweep them independently, add a
# second epsilon array/loop and swap $epsilon_fmt for two variables below.

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
    epsilon_decision=$epsilon_fmt
    epsilon_extractor=$epsilon_fmt

    for o_idx in "${!omegas[@]}"; do
        omega="${omegas[$o_idx]}"
        omega_fmt=$(printf "%.2f" $omega)
        current=$(( o_idx * epsilon_count + e_idx + 1 ))
        percent=$(( current * 100 / total_tasks ))

        # Wait for an available slot before each individual task
        wait_for_slot

        sleep 5  # Small delay to stagger starts

        echo "Running task $current/$total_tasks — omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}% complete overall)"

        nohup python3 "$PYTHON_SCRIPT" "$cluster" "$number_size" "$study_name" "$param_type" "$model_type" "$epsilon_decision" "$epsilon_extractor" "$omega" "$epochs" "$batch_size" "$epoch_size" "$fixed_variability" "$training_distribution_type" "$alpha_curriculum" "$early_stop" > "logs/all_at_once_${number_size}_${study_name}_${param_type}_${model_type}_omega-${omega_fmt}_epsilonDec-${epsilon_decision}_epsilonExt-${epsilon_extractor}_epochs-${epochs}_batch-${batch_size}_epochsize-${epoch_size}_fixed-${fixed_variability}_dist-${training_distribution_type}_alpha-${alpha_curriculum}_earlystop-${early_stop}.out" 2>&1 &

        echo "Started task $current/$total_tasks — omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}% complete overall)"
    done
done

# Esperar a que terminen todos los procesos en paralelo
wait
echo "All omega-epsilon experiments finished! (100% complete)"