# LearnLikeMe Modules Documentation

This document provides detailed specifications for all modules, scripts, and components in the LearnLikeMe framework.

## Table of Contents

1. [Core Python Scripts](#core-python-scripts)
2. [Little Learner Package](#little-learner-package)
3. [Batch Processing Scripts](#batch-processing-scripts)
4. [Dataset Files](#dataset-files)
5. [Figure Generation Scripts](#figure-generation-scripts)
6. [Jupyter Notebooks](#jupyter-notebooks)

---

## Core Python Scripts

### `generate_arithmetic_datasets.py`

**Purpose**: Generate structured datasets for multi-digit arithmetic problems.

**Usage**:
```bash
python generate_arithmetic_datasets.py <number_size>
```

**Parameters**:
- `number_size` (int): Number of digits in problems (e.g., 2 for two-digit)

**Outputs** (in `datasets/{n}-digit/`):
- `all_valid_additions.txt`: All possible addition problems
- `train_pairs_not_in_stimuli.txt`: Training set (excludes test problems)
- `stimuli_test_pairs.txt`: Test set
- `carry_additions.txt`: Problems requiring carry operations
- `small_additions.txt`: Problems with operands below threshold
- `large_additions.txt`: Problems with operands above threshold
- `test_carry_small.txt`: Test set intersection (carry ∩ small)
- `test_carry_large.txt`: Test set intersection (carry ∩ large)
- `test_no_carry_small.txt`: Test set intersection (no-carry ∩ small)
- `test_no_carry_large.txt`: Test set intersection (no-carry ∩ large)

**Example**:
```bash
# Generate 2-digit datasets
python generate_arithmetic_datasets.py 2

# Output structure:
# datasets/2-digit/all_valid_additions.txt
# datasets/2-digit/train_pairs_not_in_stimuli.txt
# ...
```

**Key Functions**:
```python
class ArithmeticDatasetGenerator:
    def __init__(self, number_size, output_dir)
    def generate_all_valid_additions(self) → List[Tuple]
    def generate_train_pairs(self) → List[Tuple]
    def generate_test_pairs(self) → List[Tuple]
    def generate_carry_additions(self) → List[Tuple]
    def categorize_by_size(self) → (List[Tuple], List[Tuple])
    def generate_test_categories(self) → None
```

---

### `train_extractor_modules.py`

**Purpose**: Train unit extractor or carry extractor modules on single-digit operations.

**Usage**:
```bash
python train_extractor_modules.py <cluster> <module_name> <study_name> <epsilon> <omega> <fixed_variability> <early_stop> <training_distribution_type> <alpha_curriculum>
```

**Parameters**:
- `cluster` (str): System configuration ('cuenca', 'brigit', 'local')
- `module_name` (str): 'unit_extractor' or 'carry_extractor'
- `study_name` (str): Experiment identifier (e.g., 'FIRST_STUDY')
- `epsilon` (float): Initialization noise (0.0-1.0)
- `omega` (float): Weber fraction for input noise (0.0-1.0)
- `fixed_variability` (str): 'Yes' or 'No' for fixed/increasing std
- `early_stop` (str): 'Yes' or 'No' for early stopping
- `training_distribution_type` (str): 'Decreasing_exponential', 'Balanced', or 'none'
- `alpha_curriculum` (float): Curriculum decay rate (0.0-1.0)

**Training Configuration**:

**Unit Extractor**:
- Hidden layers: [128, 64]
- Output: 10 classes (digits 0-9)
- Epochs: 5000
- Batch size: 25
- Checkpoint every: 200 epochs
- Learning rate: 0.003

**Carry Extractor**:
- Hidden layers: [16]
- Output: 2 classes (carry/no-carry)
- Epochs: 500
- Batch size: 25
- Checkpoint every: 10 epochs
- Learning rate: 0.003

**Outputs** (in `data/{module_name}/{study_name}/Training_{timestamp}/`):
- `training_log.csv`: Epoch-by-epoch accuracy and loss
- `training_results.csv`: Individual predictions per example
- `config.txt`: Complete configuration
- `module_checkpoint_{epoch}.pkl`: Saved parameters at checkpoints
- `module_final.pkl`: Final trained parameters

**Example**:
```bash
# Train unit extractor with balanced curriculum
python train_extractor_modules.py cuenca unit_extractor EXPERIMENT_1 0.50 0.05 No Yes Balanced 0.1

# Train carry extractor with exponential decay curriculum
python train_extractor_modules.py local carry_extractor EXPERIMENT_1 0.50 0.05 No Yes Decreasing_exponential 0.15
```

---

### `train_decision_module.py`

**Purpose**: Train the decision module that integrates extractor outputs to solve multi-digit arithmetic.

**Usage**:
```bash
python train_decision_module.py <cluster> <number_size> <study_name> <param_type> <model_type> <epsilon> <omega> <epochs> <batch_size> <epoch_size> <fixed_variability> <training_distribution_type> <alpha_curriculum>
```

**Parameters**:
- `cluster` (str): System configuration ('cuenca', 'brigit', 'local')
- `number_size` (int): Number of digits (2, 3, etc.)
- `study_name` (str): Experiment identifier
- `param_type` (str): 'WI' (Wise Initialization) or 'RI' (Random Initialization)
- `model_type` (str): 'argmax' or 'vector'
- `epsilon` (float): Initialization noise (0.0-1.0)
- `omega` (float): Extractor module version (Weber fraction used in extractor training)
- `epochs` (int): Number of training epochs (default: 5000)
- `batch_size` (int): Batch size (default: 100)
- `epoch_size` (int): Examples per epoch (default: 1000)
- `fixed_variability` (str): 'Yes' or 'No'
- `training_distribution_type` (str): 'Decreasing_exponential', 'Balanced', or 'none'
- `alpha_curriculum` (float): Curriculum decay rate

**Training Configuration**:
- Learning rate: 0.003
- Checkpoint frequency: Every 200 epochs
- Show progress: Every 1 epoch

**Outputs** (in `decision_module/{n}-digit/{study_name}/{param_type}/{model_type}_version/epsilon_{epsilon}/Training_{timestamp}/`):
- `training_log.csv`: Epoch-by-epoch metrics
- `training_results.csv`: Individual predictions
- `config.txt`: Complete configuration
- `module_checkpoint_{epoch}.pkl`: Model checkpoints
- `decision_params_initial.pkl`: Initial parameters

**Example**:
```bash
# Train 2-digit decision module with wise initialization
python train_decision_module.py cuenca 2 STUDY_1 WI argmax 0.10 0.05 5000 100 1000 No Decreasing_exponential 0.15

# Train 3-digit decision module with random initialization
python train_decision_module.py local 3 STUDY_1 RI vector 0.20 0.10 10000 100 1000 Yes Balanced 0.1
```

**Key Features**:
- Loads pre-trained extractors (frozen during training)
- Supports curriculum learning strategies
- Regular checkpointing for analysis
- Tracks accuracy by problem category

---

### `test_decision_module.py`

**Purpose**: Evaluate trained decision modules on comprehensive test sets.

**Usage**:
```bash
python test_decision_module.py <number_size> <study_name> <param_type> <model_type> <epsilon>
```

**Parameters**:
- `number_size` (int): Number of digits
- `study_name` (str): Experiment identifier
- `param_type` (str): 'WI' or 'RI'
- `model_type` (str): 'argmax' or 'vector'
- `epsilon` (float): Initialization noise value

**Testing Process**:
1. Locates all training runs matching parameters
2. For each training run:
   - Loads all checkpoints
   - Evaluates on complete test set
   - Computes accuracy by category
3. Saves results to CSV

**Outputs** (in `decision_module/{n}-digit/{study_name}/{param_type}/{model_type}_version/tests/`):
- `test_{param_type}_{epsilon}_{training_id}.csv`: Results per checkpoint

**Accuracy Metrics**:
- `overall_accuracy`: Across all test problems
- `carry_accuracy`: Problems requiring carry
- `no_carry_accuracy`: Problems without carry
- `small_accuracy`: Small operand problems
- `large_accuracy`: Large operand problems
- `carry_small_accuracy`: Intersection category
- `carry_large_accuracy`: Intersection category
- `no_carry_small_accuracy`: Intersection category
- `no_carry_large_accuracy`: Intersection category

**Example**:
```bash
# Test all training runs for WI argmax with epsilon=0.10
python test_decision_module.py 2 STUDY_1 WI argmax 0.10
```

---

### `analyze_training_decision_module.py`

**Purpose**: Analyze training dynamics and generate visualizations.

**Usage**:
```bash
python analyze_training_decision_module.py <number_size> <study_name> <param_type> <model_type>
```

**Parameters**:
- `number_size` (int): Number of digits
- `study_name` (str): Experiment identifier
- `param_type` (str): 'WI' or 'RI'
- `model_type` (str): 'argmax' or 'vector'

**Analysis Components**:
1. **Error Distance Analysis**: Distribution of prediction errors
2. **Learning Curves**: Accuracy over epochs
3. **Consolidated Data**: Aggregate results across runs

**Outputs** (in `decision_module/{n}-digit/{study_name}/{param_type}/{model_type}_version/figures/`):
- `errors_by_distance_omega_{omega}_epsilon_{epsilon}.png`: Error distance histograms
- `learning_curves_omega_{omega}.png`: Training progress plots
- `accuracy_by_category.png`: Category-wise performance
- `consolidated_training_results.csv`: Aggregated data

**Generated Figures**:
- Error distance distributions (normalized proportions)
- Accuracy evolution by omega and epsilon
- Problem-size effect visualizations
- Distance effect plots

**Example**:
```bash
# Analyze all WI argmax training runs
python analyze_training_decision_module.py 2 STUDY_1 WI argmax
```

---

### `analyze_test_decision_module.py`

**Purpose**: Analyze test results and generate comparative visualizations.

**Usage**:
```bash
python analyze_test_decision_module.py <number_size> <study_name> <param_type> <model_type>
```

**Parameters**:
- `number_size` (int): Number of digits
- `study_name` (str): Experiment identifier
- `param_type` (str): 'WI' or 'RI'
- `model_type` (str): 'argmax' or 'vector'

**Analysis Components**:
1. **Accuracy by Epsilon**: How initialization noise affects performance
2. **Category Comparisons**: Carry vs. no-carry, small vs. large
3. **Checkpoint Evolution**: Performance across training checkpoints

**Outputs** (in `decision_module/{n}-digit/{study_name}/{param_type}/{model_type}_version/figures_tests/`):
- `{category}_accuracy_by_epsilon.png`: Accuracy plots with error bars
- `tests_{param_type}_aggregated.csv`: Consolidated test results
- Comparison plots across categories

**Example**:
```bash
# Analyze test results for WI argmax
python analyze_test_decision_module.py 2 STUDY_1 WI argmax
```

---

### `analyze_training_extractor_modules.py`

**Purpose**: Analyze extractor module training dynamics.

**Usage**:
```bash
python analyze_training_extractor_modules.py <module_name> <study_name>
```

**Parameters**:
- `module_name` (str): 'unit_extractor' or 'carry_extractor'
- `study_name` (str): Experiment identifier

**Analysis Focus**:
- Convergence speed
- Final accuracy
- Error patterns
- Confusion matrices (for unit extractor)

**Example**:
```bash
python analyze_training_extractor_modules.py unit_extractor STUDY_1
```

---

### `generate_stimuli_test_pairs.py`

**Purpose**: Generate specific test stimuli based on criteria (e.g., matched for difficulty).

**Usage**: Typically called from notebooks or modified for specific experiments.

**Features**:
- Generate balanced test sets
- Control for problem-size effects
- Match human experimental stimuli

---

## Little Learner Package

### `little_learner/modules/extractor_modules/`

#### `models.py`

**Class: `ExtractorModel`**

```python
class ExtractorModel(nn.Module):
    """Neural network for extracting features from digit pairs."""
    
    def __init__(self, structure: List[int], output_dim: int):
        """
        Args:
            structure: List of hidden layer sizes (e.g., [128, 64])
            output_dim: Number of output classes (10 for unit, 2 for carry)
        """
    
    def __call__(self, x: jnp.ndarray) → jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input array of shape (batch_size, 2)
        
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
```

**Architecture**:
- Input layer: 2 neurons (two digits)
- Hidden layers: ReLU activation
- Output layer: Softmax over classes

#### `utils.py`

**Key Functions**:

```python
def load_dataset(file_path: str) → List[Tuple[int, int]]:
    """Load addition problems from file."""

def generate_test_dataset(pairs, module_name):
    """Generate test data for extractors."""

def one_hot_encode(labels, num_classes):
    """Convert labels to one-hot vectors."""

def create_and_save_initial_params(model, save_path, epsilon):
    """Initialize and save model parameters with noise."""

def load_initial_params(file_path):
    """Load saved model parameters."""

def save_results_and_module(params, results, config, save_dir):
    """Save training results and model."""
```

#### `train_utils.py`

**Key Functions**:

```python
def load_train_state(model, params, learning_rate):
    """Create optimizer state for training."""

def train_step(state, batch_x, batch_y, model):
    """Single training step with gradient update."""

def evaluate(params, x_val, y_val, model):
    """Compute validation accuracy."""

def get_predictions(params, x, model):
    """Get model predictions for inputs."""

def compute_loss(params, x, y, model):
    """Compute cross-entropy loss."""
```

### `little_learner/modules/decision_module/`

#### `model.py`

**Function: `decision_model_argmax`**

```python
def decision_model_argmax(
    extractors,
    params,
    a: int,
    b: int,
    max_digits: int,
    variability_factor: float = 0.0,
    key = None
) → jnp.ndarray:
    """
    Decision model using argmax of extractor outputs.
    
    Args:
        extractors: Tuple of (carry_model, unit_model, carry_params, unit_params)
        params: Decision module parameters (list of dicts per digit position)
        a, b: Input operands
        max_digits: Maximum number of output digits
        variability_factor: Gaussian noise std (omega)
        key: Random key for noise generation
    
    Returns:
        Array of predicted digits for each position
    """
```

**Function: `decision_model_vector`**

```python
def decision_model_vector(
    extractors,
    params,
    a: int,
    b: int,
    max_digits: int,
    variability_factor: float = 0.0,
    key = None
) → jnp.ndarray:
    """
    Decision model using full probability vectors from extractors.
    
    Similar to decision_model_argmax but preserves probability distributions.
    """
```

#### `utils.py`

**Key Functions**:

```python
def load_dataset(file_path):
    """Load arithmetic problems from file."""

def generate_test_dataset(pairs):
    """Prepare test data for decision module."""

def load_extractor_module(omega, modules_dir, model_type, study_name):
    """Load pre-trained extractor with specific omega."""

def initialize_decision_params(param_type, epsilon, structure, max_digits):
    """Initialize decision module parameters."""

def save_results_and_module(params, results, config, save_dir, epoch):
    """Save training checkpoint and results."""

def load_decision_module(module_path, structure):
    """Load trained decision module."""
```

#### `train_utils.py`

**Key Functions**:

```python
def evaluate_module(
    model_fn,
    extractors,
    params,
    test_pairs,
    max_digits,
    variability_factor,
    key
) → float:
    """Evaluate decision module accuracy."""

def update_params(
    model_fn,
    extractors,
    params,
    batch,
    learning_rate,
    max_digits,
    variability_factor,
    key
):
    """Update decision module parameters via gradient descent."""

def generate_train_dataset(
    all_pairs,
    train_pairs,
    carry_pairs,
    small_pairs,
    large_pairs,
    epoch_size,
    curriculum_type,
    alpha,
    epoch
):
    """Generate training batch according to curriculum."""

def debug_decision_example(model_fn, extractors, params, a, b, max_digits):
    """Print detailed trace of decision process for debugging."""
```

#### `test_utils.py`

**Key Functions**:

```python
def predictions(
    model_fn,
    extractors,
    params,
    test_pairs,
    max_digits,
    variability_factor,
    key
):
    """Generate predictions for test set."""

def parse_config(config_path):
    """Parse configuration file from training run."""
```

---

## Dataset Files

### `datasets/single_digit_additions.txt`

**Format**: List of tuples `(a, b, result)` for all single-digit additions.

**Example**:
```python
[(0, 0, 0), (0, 1, 1), (0, 2, 2), ..., (9, 9, 18)]
```

**Size**: 100 problems (10×10)

**Usage**: Training extractor modules

---

### `datasets/{n}-digit/all_valid_additions.txt`

**Format**: All possible n-digit addition problems.

**Example (2-digit)**:
```python
[(10, 10, 20), (10, 11, 21), ..., (99, 99, 198)]
```

**Size**: Depends on n
- 2-digit: 8,100 problems (90×90)
- 3-digit: 810,000 problems (900×900)

---

### `datasets/{n}-digit/train_pairs_not_in_stimuli.txt`

**Format**: Training set (excludes test stimuli).

**Purpose**: Ensure no overlap between train and test sets.

---

### `datasets/{n}-digit/stimuli_test_pairs.txt`

**Format**: Test set problems.

**Purpose**: Evaluation of trained models.

**Size**: Typically 200-500 problems selected for balance.

---

### Category Files

- `carry_additions.txt`: Problems where any digit pair requires carry
- `small_additions.txt`: Both operands below threshold
- `large_additions.txt`: Both operands above threshold
- `test_carry_small.txt`: Test ∩ carry ∩ small
- `test_carry_large.txt`: Test ∩ carry ∩ large
- `test_no_carry_small.txt`: Test ∩ ¬carry ∩ small
- `test_no_carry_large.txt`: Test ∩ ¬carry ∩ large

---

## Figure Generation Scripts

### `figures_generators/paper_figure_effects.py`

**Purpose**: Generate figures showing problem-size and distance effects.

**Outputs**: Publication-quality plots comparing model behavior to human data.

---

### `figures_generators/paper_anova_effects.py`

**Purpose**: Generate data and figures for ANOVA analysis.

**Outputs**: Statistical analysis tables and effect plots.

---

### `figures_generators/paper_figure_error_distance.py`

**Purpose**: Visualize error distance distributions.

**Outputs**: Histograms and density plots of prediction errors.

---

### `figures_generators/paper_figure_extractors_and_decision.py`

**Purpose**: Visualize extractor and decision module performance jointly.

**Outputs**: Combined performance plots.

---

## Jupyter Notebooks

### `CogSci_version/Easy_Multidigit_Addition_Decimal/`

Contains notebooks for CogSci 2025 experiments:

- `Create_parameters-Easy_Addition_Decimal.ipynb`: Parameter generation
- `Analysis.ipynb`: Comprehensive result analysis
- `ANOVA.ipynb`: Statistical analysis
- `Tests_results.ipynb`: Test set evaluation
- `STIMULI-Analyzed_Data.ipynb`: Stimuli-specific analysis
- `auto_RESULTS-Easy_Addition_Decimal-*.ipynb`: Automated result compilation

**Usage**: 
1. Open in Jupyter Lab/Notebook
2. Run cells sequentially
3. Modify parameters as needed
4. Generate custom analyses

---

## Output Files Reference

### Training Outputs

**`training_log.csv`**:
```csv
epoch,loss,accuracy,learning_rate
0,2.45,0.12,0.003
1,2.31,0.18,0.003
...
```

**`training_results.csv`**:
```csv
a,b,y (true),y (pred),correct,error,error_distance,omega,epsilon
23,45,68,68,True,False,0,0.05,0.10
...
```

**`config.txt`**:
```
Study Name: EXPERIMENT_1
Cluster: cuenca
Number Size: 2
Parameter Initialization Type: WI
Model Type: argmax
Epsilon: 0.10
Omega: 0.05
Epochs: 5000
Batch Size: 100
...
```

### Test Outputs

**`test_{param_type}_{epsilon}_{training_id}.csv`**:
```csv
checkpoint,training_examples,overall_accuracy,carry_accuracy,no_carry_accuracy,...
200,20000,0.85,0.78,0.92,...
400,40000,0.91,0.87,0.95,...
...
```

---

## Module Dependencies

```
train_decision_module.py
    ├── little_learner.modules.decision_module.model
    ├── little_learner.modules.decision_module.utils
    ├── little_learner.modules.decision_module.train_utils
    └── datasets/{n}-digit/*.txt

train_extractor_modules.py
    ├── little_learner.modules.extractor_modules.models
    ├── little_learner.modules.extractor_modules.utils
    ├── little_learner.modules.extractor_modules.train_utils
    └── datasets/single_digit_additions.txt

test_decision_module.py
    ├── little_learner.modules.decision_module.model
    ├── little_learner.modules.decision_module.utils
    ├── little_learner.modules.decision_module.test_utils
    └── Trained modules (*.pkl files)

analyze_*.py
    ├── pandas, matplotlib, seaborn
    ├── numpy
    └── Output CSV files from training/testing
```

---

## Best Practices

1. **Always generate datasets first** before training
2. **Train extractors before decision modules** (dependency)
3. **Use batch scripts for parameter sweeps** (efficiency)
4. **Monitor log files** during long training runs
5. **Keep study names consistent** across training/testing/analysis
6. **Backup trained modules** (especially after long training)
7. **Use descriptive study names** (include date, key parameters)
8. **Check available disk space** before large experiments
9. **Document custom modifications** in config files
10. **Version control dataset generation scripts** for reproducibility

---

For more information, see:
- [ARCHITECTURE.md](ARCHITECTURE.md): System design
- [WORKFLOWS.md](WORKFLOWS.md): Step-by-step guides
- [API.md](API.md): Function signatures and details
