#!/bin/bash

# Set locale to use period as decimal separator
export LC_NUMERIC="C"

# Array to store epsilon values from 0.01 to 0.25 in steps of 0.02
epsilons=($(seq 0.01 0.02 0.25))

# Loop through each epsilon value
for epsilon in "${epsilons[@]}"; do
    echo "Starting training with epsilon = $epsilon"
    python train_decision_module.py "$epsilon"
    echo "Completed training with epsilon = $epsilon"
    echo "-----------------------------------"
done

echo "All training runs completed!"
