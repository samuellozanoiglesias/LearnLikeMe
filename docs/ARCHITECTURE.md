# LearnLikeMe Architecture Documentation

## System Overview

LearnLikeMe implements a modular neural network architecture inspired by cognitive models of human arithmetic learning. The system decomposes arithmetic tasks into specialized sub-components, each responsible for a specific aspect of the computation.

## Architecture Principles

### 1. Modularity

The system follows a three-tier modular architecture:

```
┌─────────────────────────────────────────────┐
│         Decision Module (Integrator)        │
│  - Combines extractor outputs               │
│  - Produces final arithmetic result         │
│  - Learns from feedback                     │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼────────┐
│   Unit      │  │    Carry     │
│  Extractor  │  │  Extractor   │
│  (Module 1) │  │  (Module 2)  │
└─────────────┘  └──────────────┘
       │                │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  Input: (a, b) │
       └────────────────┘
```

### 2. Curriculum Learning

Training follows a structured curriculum that mimics human learning progression:

- **Phase 1**: Train individual extractors on single-digit operations
- **Phase 2**: Train decision module with pre-trained extractors
- **Phase 3**: Fine-tune with curriculum-based problem exposure

### 3. Human-Like Variability

The system incorporates variability at multiple levels:
- **Initialization**: Epsilon parameter controls weight initialization noise
- **Perception**: Omega parameter adds Gaussian noise to inputs (Weber's Law)
- **Training**: Curriculum controls problem difficulty distribution

## Core Components

### A. Extractor Modules

#### Unit Extractor
**Purpose**: Extracts individual digit sums from addition problems

**Architecture**:
```python
Input: (digit_a, digit_b)  # Range: 0-9 each
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Output: 10 classes (digits 0-9, one-hot encoded)
```

**Training Characteristics**:
- Dataset: All 100 single-digit addition pairs (0+0 through 9+9)
- Loss: Cross-entropy
- Optimizer: Adam with learning rate 0.003
- Typical convergence: 5000 epochs
- Output: Unit digit of sum (e.g., 7+8 → 5)

**Example**:
```
Input: (7, 8)
Process: 7 + 8 = 15
Output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # One-hot for digit 5
```

#### Carry Extractor
**Purpose**: Determines whether a carry operation is needed

**Architecture**:
```python
Input: (digit_a, digit_b)  # Range: 0-9 each
Hidden Layer: 16 neurons (ReLU)
Output: 2 classes (carry/no-carry, one-hot encoded)
```

**Training Characteristics**:
- Dataset: All 100 single-digit addition pairs
- Loss: Cross-entropy
- Optimizer: Adam with learning rate 0.003
- Typical convergence: 500 epochs
- Output: Binary classification (carry if sum ≥ 10)

**Example**:
```
Input: (7, 8)
Process: 7 + 8 = 15 ≥ 10
Output: [0, 1]  # One-hot for "carry"

Input: (3, 4)
Process: 3 + 4 = 7 < 10
Output: [1, 0]  # One-hot for "no carry"
```

### B. Decision Module

**Purpose**: Integrates extractor outputs to produce multi-digit arithmetic results

**Architecture**:
```python
For each digit position i (from right to left):
    Input: [unit_output_i, carry_output_i, carry_from_previous]
    # Dimensions: [10 (unit) + 2 (carry) + 2 (previous carry)] = 14
    
    Hidden layers: Configurable (typically [32, 16])
    Output: 10 classes (digits 0-9) for position i
```

**Two Variants**:

1. **Argmax Version**:
   - Takes argmax of extractor outputs before feeding to decision layer
   - Produces deterministic discrete inputs
   - More interpretable, but loses probability information

2. **Vector Version**:
   - Uses full probability distributions from extractors
   - Preserves uncertainty information
   - Better for gradient flow during training

**Training Process**:
```
For each epoch:
    1. Sample batch of problems according to curriculum
    2. Add Gaussian noise to inputs (if omega > 0)
    3. Forward pass through extractors (frozen)
    4. Forward pass through decision module
    5. Compute cross-entropy loss per digit
    6. Backpropagate and update decision module only
    7. Log accuracy metrics
```

## Data Flow

### Training Data Flow

```
1. Dataset Generation
   ├── generate_arithmetic_datasets.py
   └── Output: datasets/{n}-digit/*.txt

2. Extractor Training
   ├── train_extractor_modules.py
   │   ├── Load: single_digit_additions.txt
   │   ├── Train: unit_extractor or carry_extractor
   │   └── Save: module parameters + config
   └── Output: data/{module_name}/{STUDY_NAME}/

3. Decision Training
   ├── train_decision_module.py
   │   ├── Load: pre-trained extractors (omega parameter)
   │   ├── Load: training pairs from datasets/{n}-digit/
   │   ├── Train: decision module (extractors frozen)
   │   └── Save: checkpoints + training logs
   └── Output: decision_module/{n}-digit/{STUDY_NAME}/

4. Testing
   ├── test_decision_module.py
   │   ├── Load: trained decision module + extractors
   │   ├── Evaluate: on test sets (carry/no-carry, small/large)
   │   └── Save: test results per checkpoint
   └── Output: tests/ folder

5. Analysis
   ├── analyze_training_decision_module.py
   ├── analyze_test_decision_module.py
   └── Output: figures/ + aggregated CSVs
```

### Inference Data Flow

```
Input: Multi-digit addition problem (a, b)
Example: a=47, b=38

Step 1: Position 0 (rightmost, units place)
    ├── Extract digits: a[0]=7, b[0]=8
    ├── Unit Extractor(7,8) → probability vector for unit digit
    ├── Carry Extractor(7,8) → probability vector for carry
    └── Decision Module → outputs digit for position 0
    
Step 2: Position 1 (tens place)
    ├── Extract digits: a[1]=4, b[1]=3
    ├── Unit Extractor(4,3) → probability vector for unit digit
    ├── Carry Extractor(4,3) → probability vector for carry
    ├── Include carry from position 0
    └── Decision Module → outputs digit for position 1
    
Step 3: Position 2 (hundreds place, if carry exists)
    ├── If carry from position 1
    └── Decision Module → outputs digit for position 2
    
Final Output: Concatenate digits → 85 (or 085)
```

## File Organization

### Directory Structure

```
LearnLikeMe/
├── little_learner/                    # Core package
│   └── modules/
│       ├── extractor_modules/         # Extractor implementations
│       │   ├── models.py              # ExtractorModel class
│       │   ├── utils.py               # Data loading, preprocessing
│       │   └── train_utils.py         # Training loop, evaluation
│       └── decision_module/           # Decision module implementations
│           ├── model.py               # decision_model_argmax, decision_model_vector
│           ├── utils.py               # Parameter initialization, loading
│           ├── train_utils.py         # Training utilities
│           └── test_utils.py          # Testing utilities
│
├── datasets/                          # Generated training/test data
│   ├── single_digit_additions.txt     # Base data for extractors
│   └── {n}-digit/                     # Multi-digit problem sets
│       ├── all_valid_additions.txt
│       ├── train_pairs_not_in_stimuli.txt
│       ├── stimuli_test_pairs.txt
│       ├── carry_additions.txt
│       ├── small_additions.txt
│       ├── large_additions.txt
│       └── test_*.txt                 # Category-specific test sets
│
└── data/                              # Training outputs (not in repo)
    └── samuel_lozano/LearnLikeMe/
        ├── unit_extractor/{STUDY_NAME}/
        │   ├── Training_{timestamp}/
        │   │   ├── training_log.csv
        │   │   ├── training_results.csv
        │   │   ├── config.txt
        │   │   └── module_checkpoint_*.pkl
        │   └── initial_parameters/
        ├── carry_extractor/{STUDY_NAME}/
        │   └── (same structure as unit_extractor)
        └── decision_module/{n}-digit/{STUDY_NAME}/
            └── {PARAM_TYPE}/{MODEL_TYPE}_version/
                ├── epsilon_{value}/
                │   └── Training_{timestamp}/
                │       ├── training_log.csv
                │       ├── training_results.csv
                │       ├── config.txt
                │       └── module_checkpoint_*.pkl
                ├── tests/
                │   └── test_*.csv
                └── figures/
                    └── *.png
```

## Parameter Configuration

### Initialization Parameters

#### Epsilon (ε)
- **Range**: 0.0 - 5.0
- **Purpose**: Controls noise in parameter initialization
- **Effect**: Higher values → more random initial weights
- **Typical values**: 0.0, 0.50, 1.00, ..., 5.00
- **When to use**:
  - ε = 0.0: Perfect wise initialization
  - ε > 0.0: Add variability to study robustness

#### Omega (ω)
- **Range**: 0.0 - 1.0 (typically 0.0 - 0.20)
- **Purpose**: Weber fraction for Gaussian input noise
- **Effect**: Simulates perceptual variability
- **Formula**: `noisy_input = input + N(0, ω * |input|)`
- **Typical values**: 0.00, 0.05, 0.10, 0.15, 0.20
- **When to use**:
  - ω = 0.0: Perfect perception
  - ω > 0.0: Human-like variability

### Training Parameters

#### Initialization Type
- **WI (Wise Initialization)**: Parameters initialized close to ideal solution
- **RI (Random Initialization)**: Standard random initialization

#### Model Type
- **argmax**: Uses argmax of extractor outputs (discrete)
- **vector**: Uses full probability vectors (continuous)

#### Curriculum Type
- **None**: Uniform random sampling
- **Decreasing_exponential**: Exponentially decreasing focus on simpler problems
- **Balanced**: Balanced frequency across problem categories

#### Alpha (α)
- **Range**: 0.0 - 1.0
- **Purpose**: Controls curriculum decay rate
- **Only used with**: Decreasing_exponential curriculum
- **Effect**: Higher α → faster transition to harder problems

## Computational Considerations

### Memory Requirements

- **Extractors**: Minimal (~1-10 MB per module)
- **Decision Module**: Depends on architecture (~10-50 MB)
- **Training**: RAM usage depends on batch size and dataset size
  - Typical: 2-8 GB RAM for 2-digit problems
  - Large experiments: 16-32 GB for 3+ digit problems

### Training Time

**Extractor Modules** (CPU):
- Unit extractor: ~30-60 minutes (5000 epochs)
- Carry extractor: ~5-10 minutes (500 epochs)

**Decision Module** (CPU):
- 2-digit: ~2-4 hours (5000 epochs, 1000 examples/epoch)
- 3-digit: ~4-8 hours (10000 epochs)

**GPU Acceleration**:
- 5-10x speedup possible with JAX GPU support
- Requires `jax[cuda]` installation

### Parallelization

The system supports parallel training through bash scripts:
- Multiple epsilon values in parallel
- Multiple omega values in parallel
- Configurable `MAX_PARALLEL` parameter

## Design Patterns

### 1. Frozen Extractor Pattern

Extractors are trained once and frozen during decision module training:

```python
# Extractors are loaded but not trained
carry_params = load_extractor_module(omega, "carry_extractor")
unit_params = load_extractor_module(omega, "unit_extractor")

# Only decision params are updated
decision_params = initialize_decision_params()

for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass uses frozen extractors
        carry_out = frozen_carry_extractor(batch, carry_params)
        unit_out = frozen_unit_extractor(batch, unit_params)
        
        # Only decision module gradients computed
        decision_out = decision_model(carry_out, unit_out, decision_params)
        loss = compute_loss(decision_out, labels)
        decision_params = update(decision_params, gradients(loss))
```

### 2. Curriculum Sampling Pattern

Training uses curriculum-aware sampling:

```python
def generate_train_dataset(curriculum_type, alpha, epoch, epoch_size):
    if curriculum_type == "decreasing_exponential":
        # Probability decays exponentially with problem difficulty
        weights = compute_exponential_weights(problems, alpha, epoch)
        sample = weighted_sample(problems, weights, epoch_size)
    elif curriculum_type == "balanced":
        # Equal representation across categories
        sample = balanced_sample(problems, epoch_size)
    else:
        # Uniform random
        sample = random_sample(problems, epoch_size)
    return sample
```

### 3. Checkpoint Pattern

Regular checkpoints enable analysis of learning dynamics:

```python
if epoch % CHECKPOINT_EVERY == 0:
    checkpoint_path = f"module_checkpoint_{epoch}.pkl"
    save_checkpoint(decision_params, checkpoint_path)
    
    # Evaluate on validation set
    accuracy = evaluate_module(decision_params, val_data)
    log_metrics(epoch, accuracy)
```

## Extension Points

### Adding New Arithmetic Operations

1. **Create new extractor modules** for operation-specific features
2. **Modify decision module input** to accommodate new extractors
3. **Update dataset generation** for new operation type
4. **Adjust training/test scripts** for new data format

### Custom Neural Architectures

Modify architecture in:
- `little_learner/modules/extractor_modules/models.py`: Change `ExtractorModel`
- `little_learner/modules/decision_module/model.py`: Change decision layers

### New Curriculum Strategies

Add curriculum logic in:
- `little_learner/modules/decision_module/train_utils.py`: `generate_train_dataset()`

## References

For implementation details, see:
- [MODULES.md](MODULES.md): Detailed module specifications
- [API.md](API.md): Function and class references
- [WORKFLOWS.md](WORKFLOWS.md): Step-by-step usage guides
