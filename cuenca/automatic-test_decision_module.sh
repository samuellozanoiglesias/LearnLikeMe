#!/bin/bash
# Use: nohup bash automatic-test_decision_module.sh > logs_tests_decision.out &

number_size=2  # Number of digits in the numbers to be added (2 for two-digit addition)
study_name="EIGHTH_STUDY-BALANCED"  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
param_type="WI"  # "WI" for wise initialization or "RI" for random initialization
model_type="argmax"  # "argmax" or "vector" version of the decision module

# Forzar punto decimal para seq
export LC_NUMERIC=C

# Python script to run (test runner)
PYTHON_SCRIPT="../test_decision_module.py"

# Crear carpeta logs si no existe
mkdir -p logs

# Build array of epsilons and compute totals
mapfile -t epsilons < <(seq 0.00 0.05 0.50)
epsilon_count=${#epsilons[@]}
total_tasks=${epsilon_count}

# Lanzar cada epsilon en paralelo (one background job per epsilon)
for e_idx in "${!epsilons[@]}"; do
    epsilon="${epsilons[$e_idx]}"
    epsilon_fmt=$(printf "%.2f" $epsilon)

    (
        current=$(( e_idx + 1 ))
        percent=$(( current * 100 / total_tasks ))
        echo "Running task $current/$total_tasks — epsilon=$epsilon_fmt (${percent}% complete overall)"

        nohup python3 "$PYTHON_SCRIPT" "$number_size" "$study_name" "$param_type" "$model_type" "$epsilon" > "logs/tests-${number_size}_${study_name}_${param_type}_${model_type}_epsilon-${epsilon_fmt}.out" 2>&1

        echo "Finished task $current/$total_tasks — epsilon=$epsilon_fmt (${percent}% complete overall)"
    ) &
done

# Esperar a que terminen todos los procesos en paralelo
wait
echo "All epsilon experiments finished! (100% complete)"