import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# --- Config ---
CLUSTER = "cuenca"  # Cuenca, Brigit or Local
MODULE_NAME = "decision_module"

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

RAW_DIR = f"{CLUSTER_DIR}/data/samuel_lozano/LearnLikeMe/{MODULE_NAME}"

def extract_metrics_from_test_file(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Extract all metrics using regex
        patterns = {
            'total_predictions': r'Out of (\d+), (\d+) predictions correct',
            'train_predictions': r'Out of (\d+), (\d+) train predictions correct',
            'carry_over_predictions': r'Out of (\d+), (\d+) carry-over predictions correct',
            'carry_over_train_predictions': r'Out of (\d+), (\d+) carry-over train predictions correct',
            'small_predictions': r'Out of (\d+), (\d+) small predictions correct',
            'small_train_predictions': r'Out of (\d+), (\d+) small train predictions correct',
            'carry_over_small_predictions': r'Out of (\d+), (\d+) carry-over small predictions correct',
            'carry_over_small_train_predictions': r'Out of (\d+), (\d+) carry-over small train predictions correct',
            'large_predictions': r'Out of (\d+), (\d+) large predictions correct',
            'large_train_predictions': r'Out of (\d+), (\d+) large train predictions correct',
            'carry_over_large_predictions': r'Out of (\d+), (\d+) carry-over large predictions correct',
            'carry_over_large_train_predictions': r'Out of (\d+), (\d+) carry-over large train predictions correct'
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                total = int(match.group(1))
                correct = int(match.group(2))
                metrics[f'{metric_name}_total'] = total
                metrics[f'{metric_name}_correct'] = correct
                metrics[f'{metric_name}_accuracy'] = correct / total if total > 0 else 0
                
    return metrics

def analyze_decision_module(raw_dir):
    figures_dir = os.path.join(raw_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Find all test result directories
    test_dirs = []
    for root, dirs, files in os.walk(raw_dir):
        if "Tests" in dirs:
            test_dirs.append(os.path.join(root, "Tests"))

    all_results = []

    # Process each test directory
    for test_dir in test_dirs:
        for file in os.listdir(test_dir):
            if file.startswith("Tests_AP_"):
                # Extract epsilon and training date from filename
                match = re.match(r'Tests_AP_([\d.]+)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.txt', file)
                if match:
                    epsilon = float(match.group(1))
                    training_date = match.group(2)
                    
                    # Extract metrics from the test file
                    metrics = extract_metrics_from_test_file(os.path.join(test_dir, file))
                    metrics['epsilon'] = epsilon
                    metrics['training_date'] = training_date
                    all_results.append(metrics)

    if not all_results:
        print("No test results found to analyze")
        return

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Create visualization plots
    plt.style.use('seaborn')

    # 1. Overall accuracy vs epsilon
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='epsilon', y='total_predictions_accuracy', marker='o')
    plt.title('Overall Accuracy vs Epsilon')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(figures_dir, 'overall_accuracy_vs_epsilon.png'))
    plt.close()

    # 2. Training vs Test accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='epsilon', y='train_predictions_accuracy', 
                marker='o', label='Training Set')
    sns.lineplot(data=results_df, x='epsilon', y='total_predictions_accuracy', 
                marker='o', label='Full Set')
    plt.title('Training vs Overall Accuracy')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'training_vs_overall_accuracy.png'))
    plt.close()

    # 3. Small vs Large Problem Size accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='epsilon', y='small_predictions_accuracy', 
                marker='o', label='Small Problems')
    sns.lineplot(data=results_df, x='epsilon', y='large_predictions_accuracy', 
                marker='o', label='Large Problems')
    plt.title('Small vs Large Problem Size Accuracy')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'small_vs_large_accuracy.png'))
    plt.close()

    # 4. Carry-over analysis
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='epsilon', y='carry_over_predictions_accuracy', 
                marker='o', label='With Carry-over')
    sns.lineplot(data=results_df, x='epsilon', y='total_predictions_accuracy', 
                marker='o', label='Overall')
    plt.title('Carry-over vs Overall Accuracy')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'carryover_vs_overall_accuracy.png'))
    plt.close()

    # 5. Comprehensive analysis of problem types
    plt.figure(figsize=(12, 8))
    metrics = ['total_predictions_accuracy', 'carry_over_predictions_accuracy',
              'small_predictions_accuracy', 'large_predictions_accuracy',
              'carry_over_small_predictions_accuracy', 'carry_over_large_predictions_accuracy']
    labels = ['Overall', 'Carry-over', 'Small', 'Large', 
              'Carry-over Small', 'Carry-over Large']
    
    for metric, label in zip(metrics, labels):
        sns.lineplot(data=results_df, x='epsilon', y=metric, 
                    marker='o', label=label)
    plt.title('Comprehensive Analysis of Problem Types')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'comprehensive_analysis.png'))
    plt.close()

    # Save the processed results
    results_df.to_csv(os.path.join(figures_dir, 'analysis_results.csv'), index=False)
    print(f"Analysis complete. Results and figures saved in: {figures_dir}")

if __name__ == "__main__":
    analyze_decision_module(RAW_DIR)
