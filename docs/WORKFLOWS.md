# LearnLikeMe Workflows Guide

This document provides step-by-step workflows for common tasks in the LearnLikeMe framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Training a Complete Model](#training-a-complete-model)
3. [Running Parameter Sweeps](#running-parameter-sweeps)
4. [Analyzing Results](#analyzing-results)
5. [Reproducing Paper Results](#reproducing-paper-results)
6. [Custom Experiments](#custom-experiments)
7. [Troubleshooting Workflows](#troubleshooting-workflows)

---

## Getting Started

### Workflow 1: Initial Setup

**Goal**: Install dependencies and verify installation.

**Steps**:

1. **Clone the repository**:
```bash
git clone https://github.com/samuellozanoiglesias/LearnLikeMe.git
cd LearnLikeMe
```

2. **Create Python environment**:
```bash
# Using conda (recommended)
conda create -n learnlikeme python=3.10.18
conda activate learnlikeme

# Or using venv
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install little_learner package**:
```bash
cd little_learner
pip install -e .
cd ..
```

5. **Verify installation**:
```bash
python -c "import jax; import little_learner; print('Success!')"
```

**Expected Output**: `Success!`

**Time Required**: 10-15 minutes

---

### Workflow 2: Generate Your First Dataset

**Goal**: Create datasets for 2-digit addition.

**Steps**:

1. **Run dataset generator**:
```bash
python generate_arithmetic_datasets.py 2
```

2. **Verify output**:
```bash
ls datasets/2-digit/
```

**Expected Files**:
- `all_valid_additions.txt`
- `train_pairs_not_in_stimuli.txt`
- `stimuli_test_pairs.txt`
- `carry_additions.txt`
- `small_additions.txt`
- `large_additions.txt`
- `test_carry_small.txt`
- `test_carry_large.txt`
- `test_no_carry_small.txt`
- `test_no_carry_large.txt`

3. **Inspect a dataset**:
```bash
head datasets/2-digit/all_valid_additions.txt
```

**Expected Output**: List of tuples like `[(10, 10, 20), (10, 11, 21), ...]`

**Time Required**: 1-2 minutes

---

## Training a Complete Model

### Workflow 3: Train Your First Extractor Module

**Goal**: Train a unit extractor module.

**Steps**:

1. **Start training** (use `local` for your machine):
```bash
python train_extractor_modules.py local unit_extractor MY_FIRST_STUDY 0.50 0.05 No Yes Balanced 0.1
```

2. **Monitor progress**:
The script will print updates like:
```
Epoch 1/5000 - Accuracy: 0.12
Epoch 5/5000 - Accuracy: 0.45
...
```

3. **Verify outputs**:
```bash
# Find your training directory
find . -type d -name "Training_*" | head -1

# Check outputs (replace with your actual path)
ls -la data/samuel_lozano/LearnLikeMe/unit_extractor/MY_FIRST_STUDY/Training_*/
```

**Expected Files**:
- `training_log.csv`
- `training_results.csv`
- `config.txt`
- `module_checkpoint_200.pkl`
- `module_checkpoint_400.pkl`
- ...
- `module_final.pkl`

4. **Check convergence**:
```bash
# View last 10 lines of training log
tail -10 data/samuel_lozano/LearnLikeMe/unit_extractor/MY_FIRST_STUDY/Training_*/training_log.csv
```

**Expected**: Accuracy approaching 1.0 (100%)

**Time Required**: 30-60 minutes (CPU)

---

### Workflow 4: Train Both Extractor Modules

**Goal**: Train unit and carry extractors with the same parameters.

**Steps**:

1. **Train unit extractor**:
```bash
python train_extractor_modules.py local unit_extractor EXPERIMENT_A 0.50 0.05 No Yes Balanced 0.1
```

2. **Train carry extractor** (in parallel or after unit completes):
```bash
python train_extractor_modules.py local carry_extractor EXPERIMENT_A 0.50 0.05 No Yes Balanced 0.1
```

3. **Verify both completed**:
```bash
# Check unit extractor
ls data/samuel_lozano/LearnLikeMe/unit_extractor/EXPERIMENT_A/Training_*/module_final.pkl

# Check carry extractor
ls data/samuel_lozano/LearnLikeMe/carry_extractor/EXPERIMENT_A/Training_*/module_final.pkl
```

**Both should exist**

**Time Required**: 
- Unit: 30-60 minutes
- Carry: 5-10 minutes
- Can run in parallel

---

### Workflow 5: Train Decision Module

**Goal**: Train decision module using pre-trained extractors.

**Prerequisites**:
- Completed Workflow 4 (both extractors trained)
- Dataset generated (Workflow 2)

**Steps**:

1. **Identify extractor omega**: Check what omega your extractors were trained with (usually 0.05 in our example)

2. **Start decision training**:
```bash
python train_decision_module.py local 2 EXPERIMENT_A WI argmax 0.10 0.05 5000 100 1000 No Balanced 0.1
```

**Parameter Explanation**:
- `local`: Your machine
- `2`: 2-digit problems
- `EXPERIMENT_A`: Study name (match extractors)
- `WI`: Wise initialization
- `argmax`: Model type
- `0.10`: Epsilon (initialization noise)
- `0.05`: Omega (must match extractor training)
- `5000`: Epochs
- `100`: Batch size
- `1000`: Examples per epoch
- `No`: Not fixed variability
- `Balanced`: Curriculum type
- `0.1`: Alpha (unused for Balanced, but required)

3. **Monitor training**:
```
Epoch 1/5000 - Accuracy: 0.15
Epoch 10/5000 - Accuracy: 0.42
...
Checkpoint saved at epoch 200
...
```

4. **Check outputs**:
```bash
# Find training directory
find . -path "*/decision_module/2-digit/EXPERIMENT_A/WI/argmax_version/epsilon_0.10/Training_*" -type d

# List files
ls -la <path_from_above>/
```

**Expected Files**:
- `training_log.csv`
- `training_results.csv`
- `config.txt`
- `module_checkpoint_200.pkl`
- `module_checkpoint_400.pkl`
- ...

**Time Required**: 2-4 hours (CPU, 5000 epochs)

---

### Workflow 6: Test Trained Model

**Goal**: Evaluate decision module on test set.

**Prerequisites**:
- Completed Workflow 5 (decision module trained)

**Steps**:

1. **Run testing**:
```bash
python test_decision_module.py 2 EXPERIMENT_A WI argmax 0.10
```

2. **Monitor progress**:
```
Found training run: Training_2025-12-11_10-30-45
Loading checkpoint: epoch 200...
Testing checkpoint 200/5000...
Loading checkpoint: epoch 400...
...
Saved results to: tests/test_WI_0.10_<training_id>.csv
```

3. **Check results**:
```bash
# Find test results
find . -path "*/decision_module/2-digit/EXPERIMENT_A/WI/argmax_version/tests/test_WI_0.10_*.csv"

# View results
head -20 <path_to_test_csv>
```

**Expected Columns**:
- `checkpoint`: Epoch number
- `training_examples`: Total examples seen
- `overall_accuracy`: Overall test accuracy
- `carry_accuracy`: Accuracy on carry problems
- `no_carry_accuracy`: Accuracy on no-carry problems
- `small_accuracy`, `large_accuracy`, etc.

**Time Required**: 15-30 minutes

---

## Analyzing Results

### Workflow 7: Analyze Training Dynamics

**Goal**: Generate visualizations and statistics from training runs.

**Steps**:

1. **Run training analysis**:
```bash
python analyze_training_decision_module.py 2 GRID_EXPERIMENT_1 WI argmax
```

2. **Wait for processing**:
```
Processing training run: Training_2025-12-11_10-30-45
Found 15 training runs
Generating figures...
Saving consolidated data...
Analysis complete!
```

3. **Check outputs**:
```bash
# Navigate to figures directory
cd data/samuel_lozano/LearnLikeMe/decision_module/2-digit/GRID_EXPERIMENT_1/WI/argmax_version/figures/

# List generated figures
ls -lh
```

**Expected Files**:
- `errors_by_distance_omega_0_05_epsilon_0_00.png`
- `errors_by_distance_omega_0_05_epsilon_0_05.png`
- ...
- `learning_curves_omega_0_05.png`
- ...
- `consolidated_training_results.csv`

4. **View a figure**:
```bash
# Open with image viewer (adjust command for your system)
xdg-open errors_by_distance_omega_0_05_epsilon_0_10.png
```

**Time Required**: 5-15 minutes

---

### Workflow 8: Analyze Test Results

**Goal**: Generate test performance visualizations.

**Steps**:

1. **Run test analysis**:
```bash
python analyze_test_decision_module.py 2 GRID_EXPERIMENT_1 WI argmax
```

2. **Wait for processing**:
```
Found test files: 15
Aggregating results...
Generating accuracy plots...
Analysis complete!
```

3. **Check outputs**:
```bash
cd data/samuel_lozano/LearnLikeMe/decision_module/2-digit/GRID_EXPERIMENT_1/WI/argmax_version/figures_tests/

ls -lh
```

**Expected Files**:
- `overall_accuracy_by_epsilon.png`
- `carry_accuracy_by_epsilon.png`
- `no_carry_accuracy_by_epsilon.png`
- `small_accuracy_by_epsilon.png`
- `large_accuracy_by_epsilon.png`
- `tests_WI_aggregated.csv`

4. **Examine aggregated data**:
```bash
# View CSV in terminal
column -t -s, tests_WI_aggregated.csv | less -S

# Or open in spreadsheet software
```

**Time Required**: 5-10 minutes

---

### Workflow 9: Statistical Analysis (ANOVA-ready)

**Goal**: Prepare data for statistical analysis in R or Python.

**Steps**:

1. **Collect consolidated CSVs**:
```bash
# Training data
find . -name "consolidated_training_results.csv" -path "*/GRID_EXPERIMENT_1/*"

# Test data
find . -name "tests_WI_aggregated.csv" -path "*/GRID_EXPERIMENT_1/*"
```

2. **Copy to analysis directory**:
```bash
mkdir -p analysis_data
cp <path_to_consolidated_training_results.csv> analysis_data/
cp <path_to_tests_WI_aggregated.csv> analysis_data/
```

3. **Load in Python**:
```python
import pandas as pd

# Load data
train_df = pd.read_csv('analysis_data/consolidated_training_results.csv')
test_df = pd.read_csv('analysis_data/tests_WI_aggregated.csv')

# Example: Fit ANOVA model
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA on test accuracy
model = ols('overall_accuracy ~ C(epsilon) + C(omega) + C(epsilon):C(omega)', 
            data=test_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

4. **Or load in R**:
```R
# Load data
train_df <- read.csv('analysis_data/consolidated_training_results.csv')
test_df <- read.csv('analysis_data/tests_WI_aggregated.csv')

# ANOVA
model <- aov(overall_accuracy ~ epsilon * omega, data=test_df)
summary(model)

# Post-hoc tests
TukeyHSD(model)
```

**Time Required**: Varies by analysis complexity

---

## Reproducing Paper Results

### Workflow 10: Replicate CogSci 2025 Results

**Goal**: Reproduce the main results from the paper.

**Steps**:

1. **Generate 2-digit datasets**:
```bash
python generate_arithmetic_datasets.py 2
```

2. **Train extractors** (from paper configuration):
```bash
# Unit extractor with omega=0.05
python train_extractor_modules.py local unit_extractor COGSCI_2025_REPLICATION 0.50 0.05 No Yes Balanced 0.1

# Carry extractor with omega=0.05
python train_extractor_modules.py local carry_extractor COGSCI_2025_REPLICATION 0.50 0.05 No Yes Balanced 0.1
```

3. **Train decision modules** (key configurations from paper):
```bash
# WI argmax with exponential decay curriculum
python train_decision_module.py local 2 COGSCI_2025_REPLICATION WI argmax 0.10 0.05 10000 100 1000 No Decreasing_exponential 0.15

# WI argmax with balanced curriculum
python train_decision_module.py local 2 COGSCI_2025_REPLICATION WI argmax 0.10 0.05 10000 100 1000 No Balanced 0.1
```

4. **Test models**:
```bash
python test_decision_module.py 2 COGSCI_2025_REPLICATION WI argmax 0.10
```

5. **Analyze results**:
```bash
python analyze_training_decision_module.py 2 COGSCI_2025_REPLICATION WI argmax
python analyze_test_decision_module.py 2 COGSCI_2025_REPLICATION WI argmax
```

6. **Generate paper figures**:
```bash
cd figures_generators
python paper_figure_effects.py
python paper_anova_effects.py
python paper_figure_error_distance.py
```

7. **Compare with paper**:
- Check Figure 2 (problem-size effect)
- Check Figure 3 (distance effect)
- Check Table 1 (accuracy by category)
- Check supplementary materials for full statistics

**Time Required**: 1-2 days (includes training time)

---

## Custom Experiments

### Workflow 11: Train with Custom Curriculum

**Goal**: Implement and test a new curriculum learning strategy.

**Steps**:

1. **Modify curriculum function** in `little_learner/modules/decision_module/train_utils.py`:
```python
def generate_train_dataset(..., curriculum_type, ...):
    # ... existing code ...
    
    elif curriculum_type == "my_custom_curriculum":
        # Your custom logic here
        # Example: Focus on carry problems early
        if epoch < 1000:
            # 80% carry, 20% no-carry
            carry_sample = random.sample(carry_pairs, int(epoch_size * 0.8))
            no_carry_sample = random.sample(
                list(set(train_pairs) - set(carry_pairs)), 
                int(epoch_size * 0.2)
            )
            epoch_data = carry_sample + no_carry_sample
        else:
            # Gradual transition to balanced
            carry_ratio = 0.8 - (epoch - 1000) / 4000 * 0.3
            carry_sample = random.sample(carry_pairs, int(epoch_size * carry_ratio))
            no_carry_sample = random.sample(
                list(set(train_pairs) - set(carry_pairs)),
                int(epoch_size * (1 - carry_ratio))
            )
            epoch_data = carry_sample + no_carry_sample
        
        random.shuffle(epoch_data)
        return epoch_data
```

2. **Train with custom curriculum**:
```bash
python train_decision_module.py local 2 CUSTOM_CURRICULUM_TEST WI argmax 0.10 0.05 5000 100 1000 No my_custom_curriculum 0.0
```

3. **Compare with baseline**:
```bash
# Train baseline
python train_decision_module.py local 2 BASELINE_TEST WI argmax 0.10 0.05 5000 100 1000 No Balanced 0.0

# Test both
python test_decision_module.py 2 CUSTOM_CURRICULUM_TEST WI argmax 0.10
python test_decision_module.py 2 BASELINE_TEST WI argmax 0.10

# Analyze both
python analyze_test_decision_module.py 2 CUSTOM_CURRICULUM_TEST WI argmax
python analyze_test_decision_module.py 2 BASELINE_TEST WI argmax
```

4. **Compare results**:
```python
import pandas as pd
import matplotlib.pyplot as plt

custom = pd.read_csv('path/to/custom/tests_WI_aggregated.csv')
baseline = pd.read_csv('path/to/baseline/tests_WI_aggregated.csv')

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(custom['checkpoint'], custom['overall_accuracy'], label='Custom Curriculum')
plt.plot(baseline['checkpoint'], baseline['overall_accuracy'], label='Baseline')
plt.xlabel('Training Examples')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig('curriculum_comparison.png')
```

**Time Required**: Variable (includes development and testing)

---

### Workflow 12: Train on Custom Problem Set

**Goal**: Train models on your own arithmetic problems.

**Steps**:

1. **Create custom dataset file**:
```python
# custom_problems.py
problems = []
for a in range(20, 80):  # Custom range
    for b in range(20, 80):
        if a + b < 150:  # Custom constraint
            problems.append((a, b, a + b))

# Save
with open('datasets/2-digit/custom_train_set.txt', 'w') as f:
    f.write(str(problems))
```

2. **Modify training script** to load custom dataset:
```python
# In train_decision_module.py, replace:
# train_pairs = load_dataset(os.path.join(DATASET_DIR, "train_pairs_not_in_stimuli.txt"))
# with:
train_pairs = load_dataset(os.path.join(DATASET_DIR, "custom_train_set.txt"))
```

3. **Train with custom data**:
```bash
python train_decision_module.py local 2 CUSTOM_DATA_TEST WI argmax 0.10 0.05 5000 100 1000 No Balanced 0.1
```

4. **Create custom test set**:
```python
# custom_test.py
test_problems = [(a, b, a+b) for a in range(15, 25) for b in range(15, 25)]

with open('datasets/2-digit/custom_test_set.txt', 'w') as f:
    f.write(str(test_problems))
```

5. **Modify test script** to use custom test set

6. **Test and analyze as usual**

**Time Required**: Variable

---

## Troubleshooting Workflows

### Workflow 13: Debugging Training Issues

**Problem**: Model not converging or low accuracy.

**Steps**:

1. **Check dataset integrity**:
```bash
python -c "
import ast
with open('datasets/2-digit/all_valid_additions.txt', 'r') as f:
    data = ast.literal_eval(f.read())
    print(f'Loaded {len(data)} problems')
    print(f'First 5: {data[:5]}')
    print(f'Last 5: {data[-5:]}')
"
```

2. **Verify extractors are loaded correctly**:
```bash
# Add debug prints to train_decision_module.py
# After loading extractors:
print(f"Carry extractor loaded from: {carry_dir}")
print(f"Unit extractor loaded from: {unit_dir}")
print(f"Carry structure: {carry_structure}")
print(f"Unit structure: {unit_structure}")
```

3. **Check extractor accuracy**:
```python
# Test extractors independently
from little_learner.modules.extractor_modules.utils import load_dataset, generate_test_dataset
from little_learner.modules.extractor_modules.train_utils import evaluate

# Load test data
pairs = load_dataset('datasets/single_digit_additions.txt')
x_test, y_test = generate_test_dataset(pairs, 'unit_extractor')

# Load model and evaluate
# ... (load model code)
accuracy = evaluate(params, x_test, y_test, model)
print(f"Unit extractor accuracy: {accuracy}")
```

4. **Reduce learning rate if unstable**:
```python
# In train script, try:
LEARNING_RATE = 0.001  # Instead of 0.003
```

5. **Check for NaN values**:
```python
# Add to training loop:
if jnp.isnan(loss):
    print(f"NaN loss at epoch {epoch}")
    print(f"Batch: {batch}")
    break
```

---

### Workflow 14: Recovering from Interrupted Training

**Problem**: Training was interrupted (power loss, killed process, etc.)

**Steps**:

1. **Find latest checkpoint**:
```bash
find . -path "*/Training_*/module_checkpoint_*.pkl" | sort | tail -1
```

2. **Check checkpoint epoch**:
```bash
# Example: module_checkpoint_2000.pkl means trained for 2000 epochs
```

3. **Resume training** (modify train script):
```python
# In train_decision_module.py, add:
RESUME_FROM_CHECKPOINT = "/path/to/module_checkpoint_2000.pkl"
START_EPOCH = 2000

if RESUME_FROM_CHECKPOINT:
    with open(RESUME_FROM_CHECKPOINT, 'rb') as f:
        params = pickle.load(f)
    print(f"Resumed from epoch {START_EPOCH}")
else:
    params = initialize_decision_params(...)

# In training loop:
for epoch in range(START_EPOCH, EPOCHS):
    # ... training code ...
```

4. **Continue training**:
```bash
python train_decision_module.py local 2 EXPERIMENT_A WI argmax 0.10 0.05 5000 100 1000 No Balanced 0.1
```

---

### Workflow 15: Memory Optimization

**Problem**: Out of memory errors during training.

**Steps**:

1. **Reduce batch size**:
```bash
# Instead of batch_size=100:
python train_decision_module.py local 2 EXP WI argmax 0.10 0.05 5000 50 1000 No Balanced 0.1
#                                                                          ^^ reduced
```

2. **Reduce epoch size**:
```bash
# Instead of epoch_size=1000:
python train_decision_module.py local 2 EXP WI argmax 0.10 0.05 5000 100 500 No Balanced 0.1
#                                                                          ^^^ reduced
```

3. **Use CPU instead of GPU**:
```bash
# Ensure this line is in train script:
os.environ["JAX_PLATFORM_NAME"] = "cpu"
```

4. **Clear JAX cache periodically**:
```python
# Add to training loop every N epochs:
if epoch % 100 == 0:
    jax.clear_caches()
```

5. **Monitor memory usage**:
```bash
# In another terminal:
watch -n 1 'ps aux | grep python | grep train'
```

---

## Summary Checklist

### Before Training:
- [ ] Environment activated
- [ ] Dependencies installed
- [ ] `little_learner` package installed
- [ ] Datasets generated
- [ ] Sufficient disk space available

### Training Extractors:
- [ ] Study name chosen and documented
- [ ] Parameters documented
- [ ] Training logs monitored
- [ ] Checkpoints verified
- [ ] Final accuracy checked (>95% for good extractors)

### Training Decision Module:
- [ ] Extractors trained with same study name
- [ ] Omega parameter matches extractor training
- [ ] Sufficient training time allocated
- [ ] Checkpoints saving correctly
- [ ] Backup created for long runs

### Testing:
- [ ] Study name matches training
- [ ] All parameters match
- [ ] Test completes without errors
- [ ] Results CSV created

### Analysis:
- [ ] Figures generated successfully
- [ ] Consolidated CSVs created
- [ ] Results make sense (sanity check)
- [ ] Key findings documented

---

For more information:
- [ARCHITECTURE.md](ARCHITECTURE.md): System design
- [MODULES.md](MODULES.md): Detailed module specs
- [API.md](API.md): Function reference
