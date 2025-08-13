# LearnLikeMe

A package for curriculum learning and neural module training (Carry Over, Unit Extractor, etc.).

## Installation

To install with conda (from local folder):

```bash
conda create -n learnlikeme python=3.8
conda activate learnlikeme
pip install -e .
```

Or, to install as a pip package:

```bash
pip install .
```

## Usage

You can now import modules and functions in your Python code:

```python
from little_learner.modules.extractor_modules.models import CarryLSTMModel, UnitLSTMModel
from little_learner.modules.extractor_modules.utils import generate_carry_data, generate_unit_data
```
# LearnLikeMe