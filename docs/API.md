# LearnLikeMe API Reference

Complete API documentation for all modules, functions, and classes in the LearnLikeMe framework.

## Table of Contents

1. [Extractor Modules API](#extractor-modules-api)
2. [Decision Module API](#decision-module-api)
3. [Dataset Generation API](#dataset-generation-api)
4. [Utility Functions](#utility-functions)

---

## Extractor Modules API

### `little_learner.modules.extractor_modules.models`

#### Class: `ExtractorModel`

Neural network model for feature extraction from digit pairs.

```python
class ExtractorModel(nn.Module):
    structure: List[int]
    output_dim: int
```

**Parameters:**
- `structure` (List[int]): Hidden layer sizes (e.g., `[128, 64]`)
- `output_dim` (int): Number of output classes (10 for unit, 2 for carry)

**Methods:**

##### `__call__(x)`

Forward pass through the network.

```python
def __call__(self, x: jnp.ndarray) -> jnp.ndarray
```

**Args:**
- `x` (jnp.ndarray): Input of shape `(batch_size, 2)` containing digit pairs

**Returns:**
- `jnp.ndarray`: Output logits of shape `(batch_size, output_dim)`

**Example:**
```python
from little_learner.modules.extractor_modules.models import ExtractorModel
import jax.numpy as jnp

# Create model
model = ExtractorModel(structure=[128, 64], output_dim=10)

# Initialize parameters
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))

# Forward pass
inputs = jnp.array([[7, 8], [3, 4]])
outputs = model.apply(params, inputs)
print(outputs.shape)  # (2, 10)
```

---

### `little_learner.modules.extractor_modules.utils`

#### `load_dataset`

Load arithmetic problems from file.

```python
def load_dataset(file_path: str) -> List[Tuple[int, int, int]]
```

**Args:**
- `file_path` (str): Path to dataset file

**Returns:**
- `List[Tuple[int, int, int]]`: List of (operand1, operand2, result) tuples

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `SyntaxError`: If file content is malformed

**Example:**
```python
from little_learner.modules.extractor_modules.utils import load_dataset

pairs = load_dataset('datasets/single_digit_additions.txt')
print(f"Loaded {len(pairs)} problems")
print(pairs[:5])  # [(0, 0, 0), (0, 1, 1), ...]
```

---

#### `generate_test_dataset`

Generate test inputs and labels for extractor modules.

```python
def generate_test_dataset(
    pairs: List[Tuple[int, int, int]], 
    module_name: str
) -> Tuple[jnp.ndarray, jnp.ndarray]
```

**Args:**
- `pairs` (List[Tuple]): List of (a, b, result) tuples
- `module_name` (str): 'unit_extractor' or 'carry_extractor'

**Returns:**
- `Tuple[jnp.ndarray, jnp.ndarray]`: 
  - Input array of shape `(n, 2)` 
  - Label array of shape `(n,)`

**Example:**
```python
from little_learner.modules.extractor_modules.utils import (
    load_dataset, generate_test_dataset
)

pairs = load_dataset('datasets/single_digit_additions.txt')
x_test, y_test = generate_test_dataset(pairs, 'unit_extractor')

print(x_test.shape)  # (100, 2)
print(y_test.shape)  # (100,)
print(y_test[:5])    # [0, 1, 2, 3, 4] (unit digits)
```

---

#### `one_hot_encode`

Convert label array to one-hot encoded matrix.

```python
def one_hot_encode(
    labels: jnp.ndarray, 
    num_classes: int
) -> jnp.ndarray
```

**Args:**
- `labels` (jnp.ndarray): Integer labels of shape `(n,)`
- `num_classes` (int): Number of classes

**Returns:**
- `jnp.ndarray`: One-hot matrix of shape `(n, num_classes)`

**Example:**
```python
from little_learner.modules.extractor_modules.utils import one_hot_encode
import jax.numpy as jnp

labels = jnp.array([0, 2, 5, 9])
one_hot = one_hot_encode(labels, num_classes=10)
print(one_hot.shape)  # (4, 10)
print(one_hot[1])     # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

---

#### `create_and_save_initial_params`

Initialize model parameters with optional noise and save to disk.

```python
def create_and_save_initial_params(
    model: ExtractorModel,
    save_path: str,
    epsilon: float,
    rng_key: jax.random.PRNGKey = None
) -> dict
```

**Args:**
- `model` (ExtractorModel): Model to initialize
- `save_path` (str): Path to save parameters (pickle file)
- `epsilon` (float): Noise factor (0.0 = no noise, 1.0 = fully random)
- `rng_key` (PRNGKey, optional): Random key for initialization

**Returns:**
- `dict`: Initialized parameters

**Side Effects:**
- Saves parameters to `save_path`

**Example:**
```python
from little_learner.modules.extractor_modules.models import ExtractorModel
from little_learner.modules.extractor_modules.utils import create_and_save_initial_params
import jax

model = ExtractorModel(structure=[128, 64], output_dim=10)
params = create_and_save_initial_params(
    model, 
    'initial_params.pkl', 
    epsilon=0.1,
    rng_key=jax.random.PRNGKey(42)
)
```

---

#### `load_initial_params`

Load saved model parameters from disk.

```python
def load_initial_params(file_path: str) -> dict
```

**Args:**
- `file_path` (str): Path to saved parameters (pickle file)

**Returns:**
- `dict`: Loaded parameters

**Raises:**
- `FileNotFoundError`: If file doesn't exist

**Example:**
```python
from little_learner.modules.extractor_modules.utils import load_initial_params

params = load_initial_params('path/to/module_final.pkl')
```

---

#### `save_results_and_module`

Save training results and model parameters.

```python
def save_results_and_module(
    params: dict,
    results: pd.DataFrame,
    config: dict,
    save_dir: str
) -> None
```

**Args:**
- `params` (dict): Model parameters
- `results` (pd.DataFrame): Training results with predictions
- `config` (dict): Configuration dictionary
- `save_dir` (str): Directory to save files

**Side Effects:**
- Creates `save_dir` if it doesn't exist
- Saves `module_final.pkl`
- Saves `training_results.csv`
- Saves `config.txt`

**Example:**
```python
from little_learner.modules.extractor_modules.utils import save_results_and_module
import pandas as pd

results = pd.DataFrame({
    'epoch': [1, 2, 3],
    'accuracy': [0.5, 0.7, 0.9]
})

config = {
    'learning_rate': 0.003,
    'epochs': 5000,
    'epsilon': 0.5
}

save_results_and_module(params, results, config, 'output_dir/')
```

---

### `little_learner.modules.extractor_modules.train_utils`

#### `load_train_state`

Create optimizer state for training.

```python
def load_train_state(
    model: ExtractorModel,
    params: dict,
    learning_rate: float
) -> optax.TrainState
```

**Args:**
- `model` (ExtractorModel): Model to train
- `params` (dict): Initial parameters
- `learning_rate` (float): Learning rate for Adam optimizer

**Returns:**
- `optax.TrainState`: Training state with optimizer

**Example:**
```python
from little_learner.modules.extractor_modules.train_utils import load_train_state

state = load_train_state(model, params, learning_rate=0.003)
```

---

#### `train_step`

Perform single training step with gradient update.

```python
def train_step(
    state: optax.TrainState,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    model: ExtractorModel
) -> Tuple[optax.TrainState, float]
```

**Args:**
- `state` (TrainState): Current training state
- `batch_x` (jnp.ndarray): Batch inputs of shape `(batch_size, 2)`
- `batch_y` (jnp.ndarray): Batch labels (one-hot) of shape `(batch_size, num_classes)`
- `model` (ExtractorModel): Model being trained

**Returns:**
- `Tuple[TrainState, float]`:
  - Updated training state
  - Loss value for this batch

**Example:**
```python
from little_learner.modules.extractor_modules.train_utils import train_step

new_state, loss = train_step(state, batch_x, batch_y, model)
print(f"Loss: {loss:.4f}")
```

---

#### `evaluate`

Compute validation accuracy.

```python
def evaluate(
    params: dict,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    model: ExtractorModel
) -> float
```

**Args:**
- `params` (dict): Model parameters
- `x_val` (jnp.ndarray): Validation inputs of shape `(n, 2)`
- `y_val` (jnp.ndarray): Validation labels (one-hot) of shape `(n, num_classes)`
- `model` (ExtractorModel): Model to evaluate

**Returns:**
- `float`: Accuracy (0.0 to 1.0)

**Example:**
```python
from little_learner.modules.extractor_modules.train_utils import evaluate

accuracy = evaluate(params, x_val, y_val, model)
print(f"Validation accuracy: {accuracy:.4f}")
```

---

#### `get_predictions`

Get model predictions for inputs.

```python
def get_predictions(
    params: dict,
    x: jnp.ndarray,
    model: ExtractorModel
) -> jnp.ndarray
```

**Args:**
- `params` (dict): Model parameters
- `x` (jnp.ndarray): Inputs of shape `(n, 2)`
- `model` (ExtractorModel): Model

**Returns:**
- `jnp.ndarray`: Predicted class indices of shape `(n,)`

**Example:**
```python
from little_learner.modules.extractor_modules.train_utils import get_predictions

predictions = get_predictions(params, x_test, model)
print(predictions[:10])  # [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
```

---

#### `compute_loss`

Compute cross-entropy loss.

```python
def compute_loss(
    params: dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    model: ExtractorModel
) -> float
```

**Args:**
- `params` (dict): Model parameters
- `x` (jnp.ndarray): Inputs of shape `(n, 2)`
- `y` (jnp.ndarray): One-hot labels of shape `(n, num_classes)`
- `model` (ExtractorModel): Model

**Returns:**
- `float`: Average cross-entropy loss

**Example:**
```python
from little_learner.modules.extractor_modules.train_utils import compute_loss

loss = compute_loss(params, x_batch, y_batch, model)
print(f"Batch loss: {loss:.4f}")
```

---

## Decision Module API

### `little_learner.modules.decision_module.model`

#### `decision_model_argmax`

Decision model using argmax of extractor outputs.

```python
def decision_model_argmax(
    extractors: Tuple,
    params: List[dict],
    a: int,
    b: int,
    max_digits: int,
    variability_factor: float = 0.0,
    key: jax.random.PRNGKey = None
) -> jnp.ndarray
```

**Args:**
- `extractors` (Tuple): `(carry_model, unit_model, carry_params, unit_params)`
- `params` (List[dict]): Decision parameters for each digit position
- `a` (int): First operand
- `b` (int): Second operand
- `max_digits` (int): Maximum number of output digits
- `variability_factor` (float): Gaussian noise std (omega)
- `key` (PRNGKey, optional): Random key for noise

**Returns:**
- `jnp.ndarray`: Predicted digits of shape `(max_digits,)`

**Example:**
```python
from little_learner.modules.decision_module.model import decision_model_argmax

# Load extractors
carry_model, unit_model, carry_params, unit_params = load_extractors(...)
extractors = (carry_model, unit_model, carry_params, unit_params)

# Load decision params
decision_params = load_decision_params(...)

# Make prediction
result = decision_model_argmax(
    extractors, 
    decision_params, 
    a=47, 
    b=38, 
    max_digits=3,
    variability_factor=0.05,
    key=jax.random.PRNGKey(0)
)

print(result)  # [5, 8, 0] representing 85 (or 085)
```

---

#### `decision_model_vector`

Decision model using full probability vectors from extractors.

```python
def decision_model_vector(
    extractors: Tuple,
    params: List[dict],
    a: int,
    b: int,
    max_digits: int,
    variability_factor: float = 0.0,
    key: jax.random.PRNGKey = None
) -> jnp.ndarray
```

**Args and Returns:** Same as `decision_model_argmax`

**Difference:** Uses full softmax probability distributions from extractors instead of argmax, preserving uncertainty information.

---

### `little_learner.modules.decision_module.utils`

#### `load_dataset`

Load arithmetic problems from file.

```python
def load_dataset(file_path: str) -> List[Tuple[int, int, int]]
```

Same as extractor version. See above.

---

#### `generate_test_dataset`

Generate test data for decision module.

```python
def generate_test_dataset(
    pairs: List[Tuple[int, int, int]]
) -> Tuple[List[Tuple[int, int]], List[int]]
```

**Args:**
- `pairs` (List[Tuple]): List of (a, b, result) tuples

**Returns:**
- `Tuple[List[Tuple[int, int]], List[int]]`:
  - List of (a, b) input pairs
  - List of expected results

**Example:**
```python
from little_learner.modules.decision_module.utils import (
    load_dataset, generate_test_dataset
)

pairs = load_dataset('datasets/2-digit/stimuli_test_pairs.txt')
x_test, y_test = generate_test_dataset(pairs)

print(len(x_test))  # Number of test problems
print(x_test[0])    # (47, 38)
print(y_test[0])    # 85
```

---

#### `load_extractor_module`

Load pre-trained extractor module with specific omega.

```python
def load_extractor_module(
    omega: float,
    modules_dir: str,
    model_type: str,
    study_name: str
) -> Tuple[ExtractorModel, str, List[int]]
```

**Args:**
- `omega` (float): Weber fraction used in extractor training
- `modules_dir` (str): Base directory for modules
- `model_type` (str): 'unit_extractor' or 'carry_extractor'
- `study_name` (str): Study identifier

**Returns:**
- `Tuple[ExtractorModel, str, List[int]]`:
  - Loaded model with parameters
  - Path to module directory
  - Model structure (hidden layer sizes)

**Raises:**
- `FileNotFoundError`: If module not found for given omega
- `ValueError`: If multiple modules found (ambiguous)

**Example:**
```python
from little_learner.modules.decision_module.utils import load_extractor_module

carry_model, carry_dir, carry_structure = load_extractor_module(
    omega=0.05,
    modules_dir='data/samuel_lozano/LearnLikeMe',
    model_type='carry_extractor',
    study_name='EXPERIMENT_1'
)

print(f"Loaded from: {carry_dir}")
print(f"Structure: {carry_structure}")
```

---

#### `initialize_decision_params`

Initialize decision module parameters.

```python
def initialize_decision_params(
    param_type: str,
    epsilon: float,
    structure: List[int],
    max_digits: int,
    rng_key: jax.random.PRNGKey = None
) -> List[dict]
```

**Args:**
- `param_type` (str): 'WI' (Wise Initialization) or 'RI' (Random Initialization)
- `epsilon` (float): Noise factor (0.0-1.0)
- `structure` (List[int]): Hidden layer sizes per digit position
- `max_digits` (int): Number of digit positions
- `rng_key` (PRNGKey, optional): Random key

**Returns:**
- `List[dict]`: Parameters for each digit position

**Example:**
```python
from little_learner.modules.decision_module.utils import initialize_decision_params
import jax

params = initialize_decision_params(
    param_type='WI',
    epsilon=0.10,
    structure=[32, 16],
    max_digits=3,
    rng_key=jax.random.PRNGKey(42)
)

print(f"Initialized {len(params)} digit positions")
```

---

#### `load_decision_module`

Load trained decision module from checkpoint.

```python
def load_decision_module(
    module_path: str,
    structure: List[int]
) -> Tuple[List[dict], dict]
```

**Args:**
- `module_path` (str): Path to module pickle file
- `structure` (List[int]): Model structure

**Returns:**
- `Tuple[List[dict], dict]`:
  - Loaded parameters
  - Configuration dictionary

**Example:**
```python
from little_learner.modules.decision_module.utils import load_decision_module

params, config = load_decision_module(
    'path/to/module_checkpoint_2000.pkl',
    structure=[32, 16]
)

print(f"Loaded checkpoint from epoch {config['epoch']}")
```

---

#### `_parse_structure`

Parse structure string to list of integers.

```python
def _parse_structure(structure_str: str) -> List[int]
```

**Args:**
- `structure_str` (str): Structure as string (e.g., '[128, 64]' or '128, 64')

**Returns:**
- `List[int]`: Parsed structure

**Example:**
```python
from little_learner.modules.decision_module.utils import _parse_structure

structure = _parse_structure('[128, 64]')
print(structure)  # [128, 64]

structure = _parse_structure('32, 16')
print(structure)  # [32, 16]
```

---

#### `_make_hashable`

Convert tuple to hashable format for set operations.

```python
def _make_hashable(obj: Any) -> Tuple
```

**Args:**
- `obj` (Any): Object to make hashable (usually a tuple or list)

**Returns:**
- `Tuple`: Hashable representation

**Example:**
```python
from little_learner.modules.decision_module.utils import _make_hashable

hashable = _make_hashable((23, 45, 68))
print(hashable)  # (23, 45, 68)
```

---

### `little_learner.modules.decision_module.train_utils`

#### `evaluate_module`

Evaluate decision module accuracy on test set.

```python
def evaluate_module(
    model_fn: Callable,
    extractors: Tuple,
    params: List[dict],
    test_pairs: List[Tuple[int, int, int]],
    max_digits: int,
    variability_factor: float = 0.0,
    key: jax.random.PRNGKey = None
) -> float
```

**Args:**
- `model_fn` (Callable): `decision_model_argmax` or `decision_model_vector`
- `extractors` (Tuple): Loaded extractors
- `params` (List[dict]): Decision module parameters
- `test_pairs` (List[Tuple]): Test problems
- `max_digits` (int): Maximum output digits
- `variability_factor` (float): Input noise (omega)
- `key` (PRNGKey, optional): Random key

**Returns:**
- `float`: Accuracy (0.0 to 1.0)

**Example:**
```python
from little_learner.modules.decision_module.train_utils import evaluate_module
from little_learner.modules.decision_module.model import decision_model_argmax

accuracy = evaluate_module(
    decision_model_argmax,
    extractors,
    params,
    test_pairs,
    max_digits=3,
    variability_factor=0.05,
    key=jax.random.PRNGKey(0)
)

print(f"Test accuracy: {accuracy:.4f}")
```

---

#### `update_params`

Update decision module parameters via gradient descent.

```python
def update_params(
    model_fn: Callable,
    extractors: Tuple,
    params: List[dict],
    batch: List[Tuple[int, int, int]],
    learning_rate: float,
    max_digits: int,
    variability_factor: float = 0.0,
    key: jax.random.PRNGKey = None
) -> Tuple[List[dict], float]
```

**Args:**
- `model_fn` (Callable): Decision model function
- `extractors` (Tuple): Loaded extractors (frozen)
- `params` (List[dict]): Current parameters
- `batch` (List[Tuple]): Training batch
- `learning_rate` (float): Learning rate
- `max_digits` (int): Maximum output digits
- `variability_factor` (float): Input noise
- `key` (PRNGKey, optional): Random key

**Returns:**
- `Tuple[List[dict], float]`:
  - Updated parameters
  - Average loss for batch

**Example:**
```python
from little_learner.modules.decision_module.train_utils import update_params
from little_learner.modules.decision_module.model import decision_model_argmax

new_params, loss = update_params(
    decision_model_argmax,
    extractors,
    params,
    batch=[(23, 45, 68), (12, 34, 46), ...],
    learning_rate=0.003,
    max_digits=3,
    variability_factor=0.05,
    key=jax.random.PRNGKey(epoch)
)

print(f"Batch loss: {loss:.4f}")
```

---

#### `generate_train_dataset`

Generate training batch according to curriculum.

```python
def generate_train_dataset(
    all_pairs: List[Tuple],
    train_pairs: List[Tuple],
    carry_pairs: List[Tuple],
    small_pairs: List[Tuple],
    large_pairs: List[Tuple],
    epoch_size: int,
    curriculum_type: str = "none",
    alpha: float = 0.1,
    epoch: int = 0
) -> List[Tuple]
```

**Args:**
- `all_pairs` (List[Tuple]): All possible problems
- `train_pairs` (List[Tuple]): Training set problems
- `carry_pairs` (List[Tuple]): Problems requiring carry
- `small_pairs` (List[Tuple]): Small operand problems
- `large_pairs` (List[Tuple]): Large operand problems
- `epoch_size` (int): Number of examples in epoch
- `curriculum_type` (str): 'none', 'Decreasing_exponential', or 'Balanced'
- `alpha` (float): Decay rate for exponential curriculum
- `epoch` (int): Current epoch number

**Returns:**
- `List[Tuple]`: Sampled training examples for this epoch

**Example:**
```python
from little_learner.modules.decision_module.train_utils import generate_train_dataset

epoch_data = generate_train_dataset(
    all_pairs=all_problems,
    train_pairs=training_set,
    carry_pairs=carry_problems,
    small_pairs=small_problems,
    large_pairs=large_problems,
    epoch_size=1000,
    curriculum_type='Decreasing_exponential',
    alpha=0.15,
    epoch=100
)

print(f"Generated {len(epoch_data)} training examples")
```

---

#### `debug_decision_example`

Print detailed trace of decision process for debugging.

```python
def debug_decision_example(
    model_fn: Callable,
    extractors: Tuple,
    params: List[dict],
    a: int,
    b: int,
    max_digits: int
) -> None
```

**Args:**
- `model_fn` (Callable): Decision model function
- `extractors` (Tuple): Loaded extractors
- `params` (List[dict]): Decision parameters
- `a` (int): First operand
- `b` (int): Second operand
- `max_digits` (int): Maximum output digits

**Side Effects:**
- Prints detailed information about each step of the decision process

**Example:**
```python
from little_learner.modules.decision_module.train_utils import debug_decision_example
from little_learner.modules.decision_module.model import decision_model_argmax

debug_decision_example(
    decision_model_argmax,
    extractors,
    params,
    a=47,
    b=38,
    max_digits=3
)

# Output:
# Position 0: digits=(7,8)
#   Unit extractor: [0.01, 0.02, ..., 0.85, ...] → argmax=5
#   Carry extractor: [0.1, 0.9] → argmax=1 (carry)
#   Decision output: 5
# Position 1: digits=(4,3) carry_in=1
#   ...
```

---

### `little_learner.modules.decision_module.test_utils`

#### `predictions`

Generate predictions for test set.

```python
def predictions(
    model_fn: Callable,
    extractors: Tuple,
    params: List[dict],
    test_pairs: List[Tuple[int, int]],
    max_digits: int,
    variability_factor: float = 0.0,
    key: jax.random.PRNGKey = None
) -> List[int]
```

**Args:**
- `model_fn` (Callable): Decision model function
- `extractors` (Tuple): Loaded extractors
- `params` (List[dict]): Decision parameters
- `test_pairs` (List[Tuple[int, int]]): Input pairs (without results)
- `max_digits` (int): Maximum output digits
- `variability_factor` (float): Input noise
- `key` (PRNGKey, optional): Random key

**Returns:**
- `List[int]`: Predicted results for each test pair

**Example:**
```python
from little_learner.modules.decision_module.test_utils import predictions
from little_learner.modules.decision_module.model import decision_model_argmax

test_inputs = [(23, 45), (67, 89), (12, 34)]
pred = predictions(
    decision_model_argmax,
    extractors,
    params,
    test_inputs,
    max_digits=3,
    variability_factor=0.0
)

print(pred)  # [68, 156, 46]
```

---

#### `parse_config`

Parse configuration file from training run.

```python
def parse_config(config_path: str) -> dict
```

**Args:**
- `config_path` (str): Path to `config.txt` file

**Returns:**
- `dict`: Parsed configuration with keys like 'omega', 'epsilon', 'epochs', etc.

**Example:**
```python
from little_learner.modules.decision_module.test_utils import parse_config

config = parse_config('path/to/Training_*/config.txt')
print(f"Omega: {config['omega']}")
print(f"Epsilon: {config['epsilon']}")
print(f"Epochs: {config['epochs']}")
```

---

## Dataset Generation API

### `generate_arithmetic_datasets.ArithmeticDatasetGenerator`

#### Class: `ArithmeticDatasetGenerator`

Generator for structured arithmetic datasets.

```python
class ArithmeticDatasetGenerator:
    def __init__(self, number_size: int, output_dir: str = "datasets")
```

**Parameters:**
- `number_size` (int): Number of digits for problems
- `output_dir` (str): Base output directory (default: "datasets")

**Attributes:**
- `number_size` (int): Number of digits
- `max_number` (int): Maximum operand value (10^number_size)
- `output_dir` (str): Full output path

---

#### `generate_all_valid_additions()`

Generate all possible addition problems for the digit size.

```python
def generate_all_valid_additions(self) -> List[Tuple[int, int, int]]
```

**Returns:**
- `List[Tuple]`: All valid (a, b, a+b) tuples

**Example:**
```python
from generate_arithmetic_datasets import ArithmeticDatasetGenerator

gen = ArithmeticDatasetGenerator(number_size=2)
all_problems = gen.generate_all_valid_additions()
print(f"Generated {len(all_problems)} problems")  # 8100 for 2-digit
```

---

#### `generate_train_pairs()`

Generate training set (excludes test stimuli).

```python
def generate_train_pairs(self) -> List[Tuple[int, int, int]]
```

**Returns:**
- `List[Tuple]`: Training problems

---

#### `generate_test_pairs()`

Generate test set (balanced for categories).

```python
def generate_test_pairs(self) -> List[Tuple[int, int, int]]
```

**Returns:**
- `List[Tuple]`: Test problems

---

#### `generate_carry_additions()`

Generate problems requiring carry operations.

```python
def generate_carry_additions(self) -> List[Tuple[int, int, int]]
```

**Returns:**
- `List[Tuple]`: Carry problems

---

#### `categorize_by_size()`

Categorize problems as small or large based on operand sizes.

```python
def categorize_by_size(self) -> Tuple[List[Tuple], List[Tuple]]
```

**Returns:**
- `Tuple[List[Tuple], List[Tuple]]`: (small_problems, large_problems)

---

#### `generate_test_categories()`

Generate and save four test categories.

```python
def generate_test_categories(self) -> None
```

**Side Effects:**
- Saves `test_carry_small.txt`
- Saves `test_carry_large.txt`
- Saves `test_no_carry_small.txt`
- Saves `test_no_carry_large.txt`

---

#### `_save_dataset()`

Save dataset to file.

```python
def _save_dataset(self, data: List[Tuple], filename: str) -> None
```

**Args:**
- `data` (List[Tuple]): Data to save
- `filename` (str): Output filename

**Side Effects:**
- Writes data to `self.output_dir/filename`

---

## Utility Functions

### Common JAX Operations

#### `jax.random.split`

Split random key for independent random operations.

```python
key, subkey = jax.random.split(key)
```

**Example:**
```python
import jax

key = jax.random.PRNGKey(0)
key, key1, key2 = jax.random.split(key, 3)
# Use key1 and key2 for independent random operations
```

---

#### `jax.random.normal`

Generate Gaussian random noise.

```python
noise = jax.random.normal(key, shape)
```

**Example:**
```python
import jax

key = jax.random.PRNGKey(0)
noise = jax.random.normal(key, shape=(10, 2))
print(noise.shape)  # (10, 2)
```

---

### Pandas DataFrame Operations

Most analysis scripts use pandas extensively. Key operations:

```python
import pandas as pd

# Load CSV
df = pd.read_csv('path/to/file.csv')

# Group and aggregate
grouped = df.groupby(['omega', 'epsilon']).agg({
    'accuracy': ['mean', 'std'],
    'loss': 'mean'
})

# Filter
carry_df = df[df['carry'] == True]

# Save
df.to_csv('output.csv', index=False)
```

---

### Matplotlib Plotting

Common plotting patterns:

```python
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot
plt.bar(categories, values)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bar_plot.png')
```

---

## Type Annotations

Common types used throughout the codebase:

```python
from typing import List, Tuple, Dict, Callable, Optional, Any
import jax.numpy as jnp
import jax

# Common type aliases
Problem = Tuple[int, int, int]  # (operand1, operand2, result)
InputPair = Tuple[int, int]     # (operand1, operand2)
Params = Dict[str, jnp.ndarray]  # Model parameters
PRNGKey = jax.random.PRNGKey     # Random key
```

---

## Error Handling

### Common Exceptions

- `FileNotFoundError`: Dataset or module file not found
- `ValueError`: Invalid parameter value or configuration
- `KeyError`: Missing required configuration key
- `RuntimeError`: Training failure or convergence issue

### Example Error Handling

```python
try:
    params = load_initial_params('model.pkl')
except FileNotFoundError:
    print("Model file not found, initializing new parameters")
    params = initialize_decision_params(...)
except Exception as e:
    print(f"Error loading parameters: {e}")
    raise
```

---

## Configuration File Format

### `config.txt` Format

```
Study Name: EXPERIMENT_1
Cluster: cuenca
Number Size: 2
Parameter Initialization Type (Wise initialization or Random initialization): WI
Model Type (argmax or vector version): argmax
Noise Factor for Initialization Parameters (Epsilon): 0.10
Weber fraction (Omega): 0.05
Training Epochs: 5000
Batch Size: 100
Epoch Size: 1000
Fixed Variability (Yes/No): No
Training Distribution Type: Decreasing_exponential
Alpha (Curriculum parameter): 0.15
Training ID: 2025-12-11_10-30-45
```

---

## Best Practices

### Parameter Naming

- Use descriptive names: `learning_rate` not `lr`
- Follow conventions: `epsilon`, `omega`, `alpha`
- Document units and ranges in docstrings

### Code Style

- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings for all public functions
- Include examples in docstrings

### Performance

- Use JAX JIT compilation for speed: `@jax.jit`
- Vectorize operations when possible
- Profile code to identify bottlenecks
- Clear JAX caches periodically for long training

---

For more information:
- [ARCHITECTURE.md](ARCHITECTURE.md): System design
- [MODULES.md](MODULES.md): Module specifications
- [WORKFLOWS.md](WORKFLOWS.md): Step-by-step guides
