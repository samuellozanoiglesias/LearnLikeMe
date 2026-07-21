#!/bin/bash
# Use: nohup bash automatic-train_extractor_modules_perceptual.sh > logs_extractor.out 2>&1 &

cluster=cuenca
module_name=unit_extractor  # "unit_extractor" or "carry_extractor"
study_name=NEW_STUDY  # Name of the study ('FIRST_STUDY', 'SECOND_STUDY', 'THIRD_STUDY-NO_AVERAGED_OMEGA'...)
epsilon=1.0  # Epsilon value for the training
# Path to the pre-trained digit recognizer model
digit_recognizer_model_path=/data/samuel_lozano/LearnLikeMe/digit_recognizer/NEW_STUDY/Training_2026-07-19_17-38-37/trained_model.pkl
freeze_recognizer=Yes  # "Yes" or "No" to freeze the digit recognizer during training  
early_stop=No  # "Yes" or "No" to use early stopping during training
training_distribution_type=decreasing_exponential  # "decreasing_exponential" or "balanced"
alpha_curriculum=0.05  # Only used if training_distribution_type is "decreasing_exponential"
    
# Forzar punto decimal para seq
export LC_NUMERIC=C

# Carpeta padre donde está el script Python
PYTHON_SCRIPT="../train_extractor_modules_perceptual.py"

# Crear carpeta logs si no existe
mkdir -p logs

# Build array of omegas and compute total
init_omegas=0.1
end_omegas=0.1
step_omegas=0.05
mapfile -t omegas < <(seq $init_omegas $step_omegas $end_omegas)
total=${#omegas[@]}

# Launch each individual task
for i in "${!omegas[@]}"; do
    omega="${omegas[$i]}"
    omega_fmt=$(printf "%.2f" $omega)
    idx=$((i + 1))
    percent=$(( idx * 100 / total ))
    echo "Starting omega = $omega_fmt ($idx/$total) - ${percent}% completed so far"

    # nohup para que continúe corriendo si se cierra la terminal
    nohup python3 "$PYTHON_SCRIPT" "$cluster" "$module_name" "$study_name" "$epsilon" "$digit_recognizer_model_path"  "$freeze_recognizer" "$omega" "$early_stop" "$training_distribution_type" "$alpha_curriculum" > "logs/perceptual_${module_name}_${study_name}_${omega_fmt}_${freeze_recognizer}_${early_stop}_${training_distribution_type}_${alpha_curriculum}.out" 2>&1 &
    sleep 5
    
    echo "Finished omega = $omega_fmt ($idx/$total) - ${percent}% completed"
done