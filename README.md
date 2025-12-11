# LearnLikeMe: Modular Neural Networks for Human-Like Arithmetic Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-See%20LICENSE-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/CogSci-2025-orange.svg)](https://escholarship.org/uc/item/5dt3d93g)

**LearnLikeMe** is a research framework for training modular neural networks that learn multi-digit arithmetic through curriculum learning, mimicking human cognitive development. This repository contains the complete implementation for our CogSci 2025 paper.

## 📄 Publication

**"Towards a curriculum for neural networks to simulate symbolic arithmetic"**  
*Samuel Lozano, Markus Spitzer, Younes Strittmatter, Korbinian Moeller and Miguel Ruiz-Garcia*  
Proceedings of the 47th Annual Meeting of the Cognitive Science Society (2025)  
[📖 Read the paper](https://escholarship.org/uc/item/5dt3d93g)

If you use this code in your research, please cite:

```bibtex
@inproceedings{Lozano2025CogSci,
  author = {Lozano, Samuel and Spitzer, Markus and Strittmatter, Younes and Moeller, Korbinian and Ruiz-Garcia, Miguel},
  title = {Towards a curriculum for neural networks to simulate symbolic arithmetic},
  booktitle = {Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume = {47},
  year = {2025},
  url = {https://escholarship.org/uc/item/5dt3d93g}
}
```

## 🎯 Overview

LearnLikeMe demonstrates how neural networks can learn arithmetic through a modular, curriculum-based approach that mirrors human learning:

- **Modular Architecture**: Separate neural modules for digit extraction (`unit_extractor`), carry detection (`carry_extractor`), and decision-making (`decision_module`)
- **Curriculum Learning**: Multiple training strategies including exponential decay and balanced frequency distributions
- **Human-Like Learning**: Models exhibit problem-size effects, distance effects, and other phenomena observed in human arithmetic cognition
- **Extensive Analysis Tools**: Built-in scripts for training, testing, and analyzing model behavior with statistical rigor

### Key Features

✅ Multi-digit addition and multiplication tasks (2-digit, 3-digit, n-digit)  
✅ Wise and random parameter initialization strategies  
✅ Curriculum learning with multiple distribution types  
✅ Comprehensive analysis pipelines for training and test performance  
✅ ANOVA-ready output for statistical analysis  
✅ Visualization tools for publication-quality figures  
✅ Batch processing scripts for high-throughput experiments  

## 🚀 Quick Start

### Installation

**Prerequisites**: Python 3.10+ (recommended: 3.10.18)

#### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/samuellozanoiglesias/LearnLikeMe.git
cd LearnLikeMe

# Create and activate conda environment
conda create -n learnlikeme python=3.10.18
conda activate learnlikeme

# Install dependencies
pip install -r requirements.txt

# Install the little_learner package in development mode
cd little_learner
pip install -e .
cd ..
```

#### Option 2: Using pip/venv

```bash
# Clone the repository
git clone https://github.com/samuellozanoiglesias/LearnLikeMe.git
cd LearnLikeMe

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the little_learner package
cd little_learner
pip install -e .
cd ..
```

### Verify Installation

```bash
python -c "import jax; import little_learner; print('Installation successful!')"
```

## 📊 Usage

### 1. Generate Datasets

Before training, generate arithmetic datasets for your desired number size:

```bash
# Generate 2-digit addition datasets
python generate_arithmetic_datasets.py 2

# Generate 3-digit addition datasets
python generate_arithmetic_datasets.py 3
```

This creates structured datasets in `datasets/{n}-digit/` including:
- `all_valid_additions.txt`: All possible addition problems
- `train_pairs_not_in_stimuli.txt`: Training set
- `stimuli_test_pairs.txt`: Test set
- Category-specific sets (carry, small, large, etc.)

### 2. Train Extractor Modules

Train the foundational modules (unit extraction and carry detection):

```bash
# Train unit extractor (learns to extract individual digits)
python train_extractor_modules.py cuenca unit_extractor STUDY_NAME 0.50 0.05 No Yes Decreasing_exponential 0.1

# Train carry extractor (learns to detect carry operations)
python train_extractor_modules.py cuenca carry_extractor STUDY_NAME 0.50 0.05 No Yes Balanced 0.1
```

**Parameters:**
- `cuenca/brigit/local`: Cluster/system configuration
- `unit_extractor/carry_extractor`: Module type
- `STUDY_NAME`: Identifier for this experiment series
- `0.50`: Epsilon (noise factor for initialization)
- `0.05`: Omega (Weber fraction for Gaussian noise)
- `No`: Fixed variability (Yes/No)
- `Yes`: Early stopping (Yes/No)
- `Decreasing_exponential/Balanced`: Training distribution type
- `0.1`: Alpha (curriculum parameter for exponential decay)

### 3. Train Decision Module

Train the decision module that integrates extractor outputs:

```bash
python train_decision_module.py cuenca 2 STUDY_NAME WI argmax 0.10 0.05 5000 100 1000 No Decreasing_exponential 0.15
```

**Parameters:**
- `cuenca/brigit/local`: Cluster configuration
- `2`: Number of digits (2 for two-digit addition)
- `STUDY_NAME`: Study identifier
- `WI/RI`: Wise Initialization or Random Initialization
- `argmax/vector`: Model output type
- `0.10`: Epsilon (initialization noise)
- `0.05`: Omega (extractor module version)
- `5000`: Training epochs
- `100`: Batch size
- `1000`: Examples per epoch
- `No`: Fixed variability (Yes/No)
- `Decreasing_exponential/Balanced`: Training curriculum
- `0.15`: Alpha (curriculum decay rate)

### 4. Test Decision Module

Evaluate trained models on test sets:

```bash
python test_decision_module.py 2 STUDY_NAME WI argmax 0.10
```

### 5. Analyze Results

#### Training Analysis

```bash
python analyze_training_decision_module.py 2 STUDY_NAME WI argmax
```

Generates:
- Learning curves by omega and epsilon
- Error distance distributions
- Accuracy evolution plots
- Consolidated CSV files for statistical analysis

#### Test Analysis

```bash
python analyze_test_decision_module.py 2 STUDY_NAME WI argmax
```

Generates:
- Test accuracy by category (carry/no-carry, small/large)
- Performance across different checkpoints
- Comparative analyses across parameter settings

## 📁 Project Structure

```
LearnLikeMe/
├── README.md                                    # This file
├── LICENSE                                      # License information
├── requirements.txt                             # Python dependencies
├── docs/                                        # 📚 Detailed documentation
│   ├── ARCHITECTURE.md                         # System architecture
│   ├── MODULES.md                              # Module descriptions
│   ├── WORKFLOWS.md                            # Common workflows
│   └── API.md                                  # API reference
│
├── train_decision_module.py                    # Train decision module
├── train_extractor_modules.py                  # Train extractor modules
├── test_decision_module.py                     # Test decision module
├── analyze_training_decision_module.py         # Analyze training results
├── analyze_test_decision_module.py             # Analyze test results
├── analyze_training_extractor_modules.py       # Analyze extractor training
├── generate_arithmetic_datasets.py             # Generate datasets
├── generate_stimuli_test_pairs.py              # Generate test stimuli
│
├── little_learner/                             # Core package
│   ├── setup.py                                # Package setup
│   ├── environment.yml                         # Conda environment
│   └── modules/
│       ├── decision_module/                    # Decision module implementation
│       │   ├── model.py                        # Model architecture
│       │   ├── utils.py                        # Utility functions
│       │   ├── train_utils.py                  # Training utilities
│       │   └── test_utils.py                   # Testing utilities
│       └── extractor_modules/                  # Extractor implementations
│           ├── models.py                       # Extractor architectures
│           ├── utils.py                        # Utility functions
│           └── train_utils.py                  # Training utilities
│
├── datasets/                                   # Generated datasets
│   ├── single_digit_additions.txt             # Base single-digit additions
│   ├── 2-digit/                               # Two-digit problem sets
│   ├── 3-digit/                               # Three-digit problem sets
│   └── ...
│
├── figures_generators/                         # Publication figure scripts
│   ├── paper_figure_effects.py                # Generate effect plots
│   ├── paper_anova_effects.py                 # ANOVA analysis
│   ├── paper_figure_error_distance.py         # Error analysis
│   └── ...
│
└── CogSci_version/                            # CogSci 2025 experiments
    ├── Easy_Multidigit_Addition_Decimal/      # Experiment notebooks
    ├── JAX_MODULES-Easy_Multidigit_Addition_Decimal/
    ├── Multidigit_Addition/
    └── ...
```

## 🔬 Research Applications

### Replicating Paper Results

To replicate the results from our CogSci 2025 paper:

1. **Generate datasets**: `python generate_arithmetic_datasets.py 2`
2. **Train extractors**: Use configurations from `CogSci_version/Easy_Multidigit_Addition_Decimal/`
3. **Train decision modules**: Follow the parameter settings in the paper (Section 3)
4. **Run analyses**: Use the analysis scripts to generate figures
5. **Statistical tests**: Import CSV outputs into R or Python for ANOVA

### Running New Experiments

1. **Design experiment**: Define your parameter space (epsilon, omega, curriculum type)
2. **Edit batch scripts**: Modify `cuenca/automatic-*.sh` with your parameter ranges
3. **Launch experiments**: Run batch scripts with appropriate resource allocation
4. **Monitor progress**: Check log files in `cuenca/logs/`
5. **Analyze results**: Use analysis scripts to process outputs

### Custom Datasets

To work with custom arithmetic problems:

1. Format problems as tuples: `(operand1, operand2, result)`
2. Save to `datasets/{n}-digit/custom_problems.txt`
3. Modify data loading in training scripts to use your file
4. Adjust test sets accordingly

## 🧪 Example: Complete Training Pipeline

```bash
# Step 1: Generate datasets
python generate_arithmetic_datasets.py 2

# Step 2: Train unit extractor with wise initialization
python train_extractor_modules.py local unit_extractor EXPERIMENT_1 0.50 0.05 No Yes Balanced 0.1

# Step 3: Train carry extractor
python train_extractor_modules.py local carry_extractor EXPERIMENT_1 0.50 0.05 No Yes Balanced 0.1

# Step 4: Train decision module (after extractors are trained)
python train_decision_module.py local 2 EXPERIMENT_1 WI argmax 0.10 0.05 5000 100 1000 No Balanced 0.1

# Step 5: Test the trained model
python test_decision_module.py 2 EXPERIMENT_1 WI argmax 0.10

# Step 6: Analyze training dynamics
python analyze_training_decision_module.py 2 EXPERIMENT_1 WI argmax

# Step 7: Analyze test performance
python analyze_test_decision_module.py 2 EXPERIMENT_1 WI argmax
```

## 📈 Output and Results

### Training Outputs

Each training run generates:
- `training_log.csv`: Epoch-by-epoch metrics (loss, accuracy)
- `training_results.csv`: Individual predictions and errors
- `config.txt`: Complete configuration for reproducibility
- `module_checkpoint_*.pkl`: Saved model parameters
- `figures/`: Visualization of training dynamics

### Test Outputs

Test runs produce:
- `test_*.csv`: Detailed test results with accuracy by category
- Performance metrics for carry/no-carry and small/large problems
- Checkpoint-wise accuracy tracking

### Analysis Outputs

Analysis scripts generate:
- Publication-ready figures (PNG, 300 DPI)
- Consolidated CSV files for statistical analysis
- Summary statistics tables
- ANOVA-compatible data formats

## 🔧 Advanced Configuration

### Cluster/HPC Usage

The codebase is designed for HPC environments (Cuenca, Brigit clusters). To adapt for your system:

1. **Modify cluster paths** in training scripts:
   ```python
   if CLUSTER == "your_cluster":
       CLUSTER_DIR = "/your/path"
       CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
   ```

2. **Adjust parallel execution** in batch scripts:
   ```bash
   MAX_PARALLEL=10  # Set based on available resources
   ```

3. **Configure job scheduling** (if using SLURM, PBS, etc.)

### JAX Configuration

By default, the code runs on CPU. To use GPU:

```python
# Comment out or remove this line in training scripts:
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
```

Ensure you have `jax[cuda]` installed for GPU support.

### Custom Architectures

To modify neural architectures, edit:
- **Extractors**: `little_learner/modules/extractor_modules/models.py`
- **Decision module**: `little_learner/modules/decision_module/model.py`

## 🐛 Troubleshooting

### Common Issues

**Import errors for `little_learner`**:
```bash
cd little_learner
pip install -e .
```

**JAX installation issues**:
```bash
# For CPU-only
pip install jax[cpu]

# For GPU
pip install jax[cuda12]  # Adjust CUDA version as needed
```

**Memory errors during training**:
- Reduce `BATCH_SIZE` in training scripts
- Reduce `EPOCH_SIZE` for decision module training

**Missing datasets**:
- Run `generate_arithmetic_datasets.py` first
- Check that dataset files exist in `datasets/{n}-digit/`

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design and component interactions
- **[MODULES.md](docs/MODULES.md)**: Detailed module specifications
- **[WORKFLOWS.md](docs/WORKFLOWS.md)**: Step-by-step guides for common tasks
- **[API.md](docs/API.md)**: Function and class reference

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

**Samuel Lozano Iglesias**  
Universidad Complutense de Madrid  
📧 samuel.lozano@ucm.es

For questions about:
- **Research/theory**: Contact Samuel Lozano Iglesias
- **Technical issues**: Open a GitHub issue
- **Collaboration**: Email Samuel with your proposal

## 📜 License

See the [LICENSE](LICENSE) file for rights and limitations.