# Exercise: The Hyperparameter Optimizer 🎯

## Overview
In this exercise, you will implement a **mini random search hyperparameter optimizer** for a Fashion-MNIST image classifier. Instead of manually tuning hyperparameters, you'll build an automated system that explores different configurations and finds the best one.

---

## Learning Objectives
By completing this exercise, you will:
1. Understand how hyperparameter search spaces are defined and sampled
2. Implement random search optimization from scratch
3. Build structured experiment logging systems
4. Apply early stopping criteria to save computation time
5. Analyze and compare experimental results programmatically

---

## Background (From Class)
As covered in the presentation:
- **Hyperparameters** determine network structure (layers, units, dropout) and training behavior (learning rate, epochs, batch size)
- Finding good hyperparameters is crucial for model performance
- The Fashion-MNIST dataset has 10 clothing categories (T-shirt, Trouser, Pullover, etc.)

**What's NEW in this exercise:**
- You'll implement **Random Search** - an automated method to sample and evaluate hyperparameter configurations
- You'll implement **Early Stopping** - stopping training when validation loss stops improving
- You'll build a **structured logging system** for tracking experiments

---

## Exercise Structure

### Part 1: Search Space Definition (TODO 1-2)
Define a hyperparameter search space as a dictionary and implement random sampling from it.

```python
# Example search space format:
search_space = {
    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
    "batch_size": {"type": "choice", "options": [32, 64, 128]},
    "num_filters": {"type": "int", "low": 16, "high": 64}
}
```

### Part 2: Training Wrapper with Early Stopping (TODO 3-4)
Implement a function that trains a model with given hyperparameters and returns metrics. Include early stopping when validation loss doesn't improve for `patience` epochs.

### Part 3: Experiment Logger (TODO 5-6)
Build a logger that records each experiment in a specific format:
```
[EXP-001] lr=0.0012 | batch=64 | filters=32 | val_acc=0.8934 | status=COMPLETED
```

### Part 4: Search Loop & Best Config Selector (TODO 7-8)
Implement the main random search loop and a function to find the best configuration based on validation accuracy.

---

## Required Output Format ⚠️

Your code **MUST** produce output in these exact formats:

### Experiment Log Format:
```
[EXP-XXX] lr=Y.YYYY | batch=ZZ | filters=FF | val_acc=A.AAAA | status=STATUS
```
Where:
- `XXX` = 3-digit experiment number (001, 002, ...)
- `Y.YYYY` = learning rate with 4 decimal places
- `ZZ` = batch size (integer)
- `FF` = number of filters (integer)
- `A.AAAA` = validation accuracy with 4 decimal places
- `STATUS` = one of: `COMPLETED`, `EARLY_STOPPED`, `FAILED`

### Final Summary Format:
```
=== SEARCH COMPLETE ===
Total experiments: N
Best config: lr=Y.YYYY, batch=ZZ, filters=FF
Best validation accuracy: A.AAAA
```

---

---

## Submission Requirements

1. Complete all **8 TODOs** in `code_starter_for_stud.py`
2. Both files must pass all automated tests
3. Output must match the required format exactly

---

