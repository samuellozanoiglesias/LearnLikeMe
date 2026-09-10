#!/bin/bash
# Use: nohup bash automatic-train_extractor_modules.sh > logs_extractor.out 2>&1 &

cluster=cuenca
module_name=carry_extractor  # "unit_extractor" or "carry_extractor"
study_name=19_STUDY-FIXED_EXP_DECAY_0.05-OMEGA_0.10  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
fixed_variability=Yes  # "Yes" or "No" to use fixed/increasing variability for the std of inputs
early_stop=No  # "Yes" or "No" to use early stopping during training
training_distribution_type=decreasing_exponential  # "decreasing_exponential" or "balanced"
alpha_curriculum=0.05  # Only used if training_distribution_type is "decreasing_exponential"
    
# Forzar punto decimal para seq
export LC_NUMERIC=C

# Carpeta padre donde está el script Python
PYTHON_SCRIPT="../train_extractor_modules.py"

# Crear carpeta logs si no existe
mkdir -p logs

# Build array of omegas and compute total
init_omegas=0.1
end_omegas=0.1
step_omegas=0.05

init_epsilons=0.5
end_epsilons=0.5
step_epsilons=0.05

init_seeds=100
end_seeds=120
step_seeds=1

mapfile -t omegas < <(seq $init_omegas $step_omegas $end_omegas)
mapfile -t epsilons < <(seq $init_epsilons $step_epsilons $end_epsilons)
mapfile -t seeds < <(seq $init_seeds $step_seeds $end_seeds)
omega_count=${#omegas[@]}
epsilon_count=${#epsilons[@]}
seed_count=${#seeds[@]}
total_tasks=$(( omega_count * epsilon_count * seed_count ))

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

        for s_idx in "${!seeds[@]}"; do
            seed="${seeds[$s_idx]}"
            seed_fmt=$(printf "%03d" $seed)
            current=$(( o_idx * epsilon_count * seed_count + e_idx * seed_count + s_idx + 1 ))
            percent=$(( current * 100 / total_tasks ))

            # Wait for an available slot before each individual task
            wait_for_slot

            sleep 5  # Small delay to stagger starts

            echo "Running task $current/$total_tasks — omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}% complete overall)"

            nohup python3 "$PYTHON_SCRIPT" "$cluster" "$module_name" "$study_name" "$epsilon" "$omega" "$fixed_variability" "$early_stop" "$training_distribution_type" "$alpha_curriculum" "$seed" > "logs/${module_name}_${study_name}_${epsilon_fmt}_${omega_fmt}_${fixed_variability}_${early_stop}_${training_distribution_type}_${alpha_curriculum}_${seed_fmt}.out" 2>&1 &

            echo "Started task $current/$total_tasks — omega=$omega_fmt, epsilon=$epsilon_fmt (${percent}% complete overall)"
        done
    done
done

# Esperar a que terminen todos los procesos en paralelo
wait
echo "All omega-epsilon experiments finished! (100% complete)"