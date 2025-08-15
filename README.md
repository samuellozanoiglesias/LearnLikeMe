# LearnLikeMe

A Python package implementing a modular neural network approach for learning arithmetic operations through curriculum learning. The project focuses on breaking down complex arithmetic tasks into simpler neural modules that can be trained independently and then combined.

Peer-reviewed paper, titled "Towards a curriculum for neural networks to simulate symbolic arithmetic", is now available in the official conference proceedings: https://escholarship.org/uc/item/5dt3d93g (Publication in the Proceedings of the Annual Meeting of the Cognitive Science Society, vol. 47).

## Features

- **Modular Neural Architecture**: Implements specialized LSTM-based modules for different arithmetic operations:
  - Unit Addition Module: Handles basic digit-by-digit addition
  - Carry Module: Manages carry-over operations in multi-digit arithmetic
  - Decimal Module: Processes decimal arithmetic operations

- **Curriculum Learning**: Progressive training approach from simple to complex arithmetic tasks
- **JAX Implementation**: Efficient neural network training using JAX and Flax
- **Extensive Analysis Tools**: Jupyter notebooks for analyzing model performance and behavior

## Installation

### Using Conda (Recommended)

```bash
conda create -n learnlikeme python=3.8
conda activate learnlikeme
pip install -e .
```

### Using Pip

```bash
pip install .
```

## Project Structure

- `little_learner/`: Core package containing the neural network modules
  - `modules/extractor_modules/`: Neural network model implementations
- `datasets/`: Training and testing datasets for arithmetic operations
- `CogSci_version/`: Implementation and analysis notebooks for different arithmetic tasks presented in the Proceedings of the Annual Meeting of the Cognitive Science Society, vol. 47.
  - Various arithmetic operation directories with analysis notebooks
- `JAX_MODULES-*/`: JAX-based implementations of the neural modules

## Usage

Basic usage example:

```python
from little_learner.modules.extractor_modules.models import CarryLSTMModel, UnitLSTMModel
from little_learner.modules.extractor_modules.utils import generate_carry_data, generate_unit_data

# Create and train individual modules
unit_model = UnitLSTMModel()
carry_model = CarryLSTMModel()

# Generate training data
unit_data = generate_unit_data()
carry_data = generate_carry_data()
```

## Research Context

This project implements a cognitive science approach to understanding how neural networks can learn arithmetic operations in a way that mimics human learning. The modular architecture allows for studying how different components of arithmetic processing work together.

## License

See the [LICENSE](LICENSE) file for rights and limitations.