
# LearnLikeMe

Welcome! This project is designed for psychologists, cognitive scientists, and anyone interested in how neural networks can learn arithmetic like humans do. You **do not need to be a programmer** to use the main features—just follow the simple steps below.

## What is this project about?

LearnLikeMe is a set of tools and models that simulate how people learn to do arithmetic (like addition and multiplication) step by step. Instead of training one big neural network, we break the problem into smaller pieces (modules), each learning a simple part (like adding digits or handling carry-overs). This makes it easier to understand and analyze how learning happens.

## Our Paper

This project is part of our research published at CogSci 2025:

**"Towards a curriculum for neural networks to simulate symbolic arithmetic"**  
Proceedings of the Annual Meeting of the Cognitive Science Society, vol. 47  
[Read the paper here](https://escholarship.org/uc/item/5dt3d93g)


## What can you do with LearnLikeMe?

- **Train neural networks** to solve arithmetic problems in a way that mimics human learning.
- **Analyze how learning happens** step by step, using easy-to-read Jupyter notebooks and scripts.
- **Experiment with different training strategies** (curriculum learning).
- **Visualize results** and compare them to human data.

## Requirements

All required Python packages are listed in `requirements.txt`.
To install everything, run:

```bash
pip install -r requirements.txt
```

Or, using Conda (recommended):

```bash
conda create -n learnlikeme python=3.10.18
conda activate learnlikeme
pip install -r requirements.txt
```

## How do I run experiments? (Step-by-step for non-coders)

### 1. Training a Model

To train a decision module (the neural network that learns arithmetic), use:

```bash
nohup python train_decision_module.py 0.01 WI 0.05 argmax > logs_train_decision.out 2>&1 &
```

- `0.01` = noise factor for parameter initialization (epsilon)
- `WI` = Wise Initialization (or use `RI` for Random Initialization)
- `0.05` = Omega value (controls pre-trained modules)
- `argmax` = model type (can also use `vector`)
- Output is saved to `logs_train_decision.out`

### 2. Testing a Model

To test a trained module:

```bash
nohup python test_decision_module.py 0.01 WI argmax > logs_test_decision.out 2>&1 &
```

- Same arguments as above (epsilon, initialization type, model type)
- Output is saved to `logs_test_decision.out`

### 3. Analyzing Training Results

To analyze the results of your training runs:

```bash
nohup python analyze_training_decision_module.py WI argmax > logs_analysis_training_decision.out 2>&1 &
```

- `WI` = initialization type
- `argmax` = model type
- Output is saved to `logs_analysis_training_decision.out`
- This script will generate figures and summary tables to help you interpret the learning process.

### 4. Analyzing Test Results

To analyze the results of your test runs:

```bash
nohup python analyze_test_decision_module.py WI argmax > logs_analysis_test_decision.out 2>&1 &
```

- Same arguments as above
- Output is saved to `logs_analysis_test_decision.out`
- This script will generate figures and summary tables for test performance.

## Project Structure (What’s inside?)

- `requirements.txt`: List of all required Python packages
- `train_decision_module.py`: Script to train the neural network
- `test_decision_module.py`: Script to test the trained network
- `analyze_training_decision_module.py`: Script to analyze training results
- `analyze_test_decision_module.py`: Script to analyze test results
- `little_learner/`: Core code for the neural modules (no need to edit for basic use)
- `datasets/`: All the data for training/testing
- `CogSci_version/`: Notebooks for different arithmetic tasks and analyses
- `JAX_MODULES-*`: Advanced implementations (for programmers)

## Frequently Asked Questions

**Q: Do I need to know Python?**  
A: No! Just follow the steps above and use the notebooks.

**Q: Can I use my own data?**  
A: Yes! Put your data in the `datasets/` folder and update the notebook to use your file.

**Q: Where do I find more help?**  
A: Each notebook and script includes step-by-step comments to guide you. For more details, please read our [CogSci 2025 paper](https://escholarship.org/uc/item/5dt3d93g). If you have questions or need further assistance, feel free to contact Samuel Lozano at samuel.lozano@ucm.es.

## Citation

If you use this repository in your work or project, please cite it as:

Lozano, S. (2025). *LearnLikeMe*. GitHub. https://github.com/samuellozanoiglesias/LearnLikeMe

```bibtex
@misc{LozanoIglesias2025LearnLikeMe,
  author       = {Samuel Lozano},
  title        = {LearnLikeMe},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/samuellozanoiglesias/LearnLikeMe},
  note         = {Accessed: YYYY-MM-DD}
}

## License

See the [LICENSE](LICENSE) file for rights and limitations.
