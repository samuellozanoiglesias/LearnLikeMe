# LearnLikeMe Documentation Index

Welcome to the comprehensive documentation for the LearnLikeMe framework. This index will help you navigate the documentation based on your needs.

## 📚 Documentation Structure

### [README.md](../README.md)
**Start here!** The main project README with quick start guide, installation instructions, and basic usage examples.

**Who should read this:**
- First-time users
- Researchers wanting a project overview
- Anyone looking for installation instructions
- Users needing citation information

**Key sections:**
- Publication information and citation
- Quick start guide
- Installation instructions (Conda and pip)
- Basic usage examples
- Project structure overview
- Troubleshooting tips

---

### [ARCHITECTURE.md](ARCHITECTURE.md)
Comprehensive system design and architectural principles.

**Who should read this:**
- Researchers wanting to understand the modular design
- Developers planning to extend the framework
- Students learning about cognitive modeling
- Anyone interested in the implementation details

**Key sections:**
- Modular architecture overview
- Curriculum learning principles
- Core component descriptions (Extractors, Decision Module)
- Data flow diagrams
- Parameter configuration explanations
- Design patterns and extension points

---

### [MODULES.md](MODULES.md)
Detailed specifications for all scripts, modules, and files.

**Who should read this:**
- Users running experiments
- Researchers analyzing results
- Developers modifying code
- Anyone needing to understand what each file does

**Key sections:**
- Core Python scripts (train, test, analyze)
- Little learner package structure
- Batch processing scripts
- Dataset file formats
- Figure generation scripts
- Output file specifications

---

### [WORKFLOWS.md](WORKFLOWS.md)
Step-by-step guides for common tasks and workflows.

**Who should read this:**
- New users learning the system
- Researchers running experiments
- Anyone troubleshooting issues
- Users wanting to replicate paper results

**Key sections:**
- Getting started workflows
- Training complete models
- Running parameter sweeps
- Analyzing results
- Reproducing paper results
- Custom experiments
- Troubleshooting guides

---

### [API.md](API.md)
Complete API reference for all functions and classes.

**Who should read this:**
- Developers writing code
- Users needing function signatures
- Researchers implementing custom modules
- Anyone debugging code issues

**Key sections:**
- Extractor modules API
- Decision module API
- Dataset generation API
- Utility functions
- Type annotations
- Error handling
- Best practices

---

## 🎯 Quick Navigation by Task

### I want to...

#### **Install and set up the framework**
→ Start with [README.md](../README.md) → "Quick Start" section

#### **Understand how the system works**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) → "System Overview"

#### **Run my first experiment**
→ Follow [WORKFLOWS.md](WORKFLOWS.md) → "Workflow 1-5"

#### **Train models with different parameters**
→ See [WORKFLOWS.md](WORKFLOWS.md) → "Running Parameter Sweeps"

#### **Understand what each script does**
→ Check [MODULES.md](MODULES.md) → "Core Python Scripts"

#### **Modify or extend the code**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) → "Extension Points"  
→ Then [API.md](API.md) for function details

#### **Analyze my results**
→ Follow [WORKFLOWS.md](WORKFLOWS.md) → "Analyzing Results"  
→ Check [MODULES.md](MODULES.md) → "Output Files Reference"

#### **Reproduce the paper results**
→ Follow [WORKFLOWS.md](WORKFLOWS.md) → "Reproducing Paper Results"

#### **Troubleshoot an error**
→ See [WORKFLOWS.md](WORKFLOWS.md) → "Troubleshooting Workflows"  
→ Check [README.md](../README.md) → "Troubleshooting" section

#### **Find a specific function**
→ Use [API.md](API.md) and search (Ctrl+F)

#### **Understand the data format**
→ Check [MODULES.md](MODULES.md) → "Dataset Files"

#### **Write custom curriculum**
→ Read [WORKFLOWS.md](WORKFLOWS.md) → "Workflow 14"  
→ Then [API.md](API.md) → `generate_train_dataset`

---

## 📖 Reading Order by Experience Level

### Beginners (No prior experience with the framework)

1. **[README.md](../README.md)** - Read the entire file (15 min)
2. **[WORKFLOWS.md](WORKFLOWS.md)** - Focus on "Getting Started" section (20 min)
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Skim "System Overview" (10 min)
4. **Practice** - Follow Workflow 1-6 to train your first model
5. **Return to docs** - As needed when you have questions

**Total time:** ~1 hour reading + practice time

---

### Intermediate Users (Some experience with neural networks)

1. **[README.md](../README.md)** - Quick review (5 min)
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Read completely (30 min)
3. **[MODULES.md](MODULES.md)** - Skim to understand components (20 min)
4. **[WORKFLOWS.md](WORKFLOWS.md)** - Read relevant workflows (20 min)
5. **[API.md](API.md)** - Use as reference when needed

**Total time:** ~1-2 hours

---

### Advanced Users (Researchers/Developers)

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep dive into design (45 min)
2. **[API.md](API.md)** - Study function signatures (30 min)
3. **[MODULES.md](MODULES.md)** - Understand all components (30 min)
4. **Code** - Read `little_learner/` source code
5. **[WORKFLOWS.md](WORKFLOWS.md)** - Reference for specific tasks

**Total time:** ~2-3 hours + code reading

---

## 🔍 Documentation Search Tips

### Finding Information Quickly

**Use your text editor/browser search:**
- **Ctrl+F** (or **Cmd+F** on Mac) to search within a file
- Search for keywords like: "train", "test", "accuracy", "epsilon", "omega"

**Common search terms:**

| What you want | Search for |
|---------------|------------|
| How to train | "train" in WORKFLOWS.md |
| Parameter meanings | "epsilon" or "omega" in ARCHITECTURE.md |
| Function details | Function name in API.md |
| File formats | "format" or "csv" in MODULES.md |
| Error solutions | "Error" or "Troubleshooting" in WORKFLOWS.md |
| Configuration | "config" or "parameters" |

---

## 📊 Documentation Statistics

- **Total documentation pages:** 5
- **Total words:** ~35,000
- **Code examples:** 100+
- **Workflows covered:** 18
- **Functions documented:** 50+
- **Figure types explained:** 10+

---

## 🆘 Getting Additional Help

### If you can't find what you need:

1. **Check the documentation index above** - Are you looking in the right file?
2. **Use search (Ctrl+F)** - Try different keywords
3. **Look at code examples** - Sometimes examples clarify better than text
4. **Check the paper** - The CogSci 2025 paper has theoretical background
5. **Open a GitHub issue** - For bugs or unclear documentation
6. **Contact the authors** - Email samuel.lozano@ucm.es for research questions

---

## 📝 Documentation Conventions

### Formatting Used in Docs

- **Bold text**: Important terms, parameters, filenames
- `code font`: Code, commands, function names, file paths
- *Italic text*: Emphasis, paper titles
- > Blockquotes: Important notes or warnings
- ✅ Checkmarks: Features or completed tasks
- 📁 📊 🔧 Icons: Visual organization

### Code Blocks

```bash
# Bash commands look like this
python script.py argument
```

```python
# Python code looks like this
def function():
    return value
```

### Parameter Format

**`parameter_name`** (type): Description
- Range or possible values
- Default value if applicable
- Example usage

---

## 🔄 Documentation Updates

This documentation was created for version 1.0 of LearnLikeMe (December 2025).

**Documentation version:** 1.0  
**Last updated:** December 11, 2025  
**Compatible with:** LearnLikeMe v1.0+

For updates and new versions, check the GitHub repository.

---

## 📚 Related Resources

### External Resources

- **CogSci 2025 Paper**: [Read online](https://escholarship.org/uc/item/5dt3d93g)
- **JAX Documentation**: [jax.readthedocs.io](https://jax.readthedocs.io/)
- **Flax Documentation**: [flax.readthedocs.io](https://flax.readthedocs.io/)
- **GitHub Repository**: [github.com/samuellozanoiglesias/LearnLikeMe](https://github.com/samuellozanoiglesias/LearnLikeMe)

### Jupyter Notebooks

Interactive examples are available in:
- `CogSci_version/Easy_Multidigit_Addition_Decimal/`
- `CogSci_version/JAX_MODULES-Easy_Multidigit_Addition_Decimal/`

---

## ✨ Documentation Quality

We strive for:
- **Completeness**: All features documented
- **Accuracy**: Up-to-date with code
- **Clarity**: Accessible to diverse audiences
- **Examples**: Practical code samples
- **Organization**: Easy navigation

**Found an error or unclear section?**  
Please open a GitHub issue or contact samuel.lozano@ucm.es

---

## 📋 Quick Reference Card

### Essential Commands

```bash
# Setup
conda create -n learnlikeme python=3.10.18
conda activate learnlikeme
pip install -r requirements.txt
cd little_learner && pip install -e . && cd ..

# Generate datasets
python generate_arithmetic_datasets.py 2

# Train extractors
python train_extractor_modules.py local unit_extractor STUDY 0.50 0.05 No Yes Balanced 0.1
python train_extractor_modules.py local carry_extractor STUDY 0.50 0.05 No Yes Balanced 0.1

# Train decision module
python train_decision_module.py local 2 STUDY WI argmax 0.10 0.05 5000 100 1000 No Balanced 0.1

# Test
python test_decision_module.py 2 STUDY WI argmax 0.10

# Analyze
python analyze_training_decision_module.py 2 STUDY WI argmax
python analyze_test_decision_module.py 2 STUDY WI argmax
```

### Key Parameters

- **epsilon**: Initialization noise (0.0-1.0)
- **omega**: Input noise/Weber fraction (0.0-0.2)
- **WI/RI**: Wise/Random initialization
- **argmax/vector**: Model output type
- **Balanced/Decreasing_exponential**: Curriculum type

---

**Happy researching! 🚀**

For questions: samuel.lozano@ucm.es
