"""
Hyperparameter Optimizer - Student Starter Code
================================================
Complete all TODOs to implement a random search hyperparameter optimizer.

Your task: Implement a system that automatically searches for the best
hyperparameters for a Fashion-MNIST classifier.

IMPORTANT: Follow the exact output format specified in each function's docstring.
"""

import random
from typing import Dict, List, Optional, Tuple
from mock_trainer import MockModelTrainer


# =============================================================================
# PART 1: Search Space Definition and Sampling (TODOs 1-2)
# =============================================================================

def create_search_space() -> Dict:
    """
    Create and return the hyperparameter search space.
    
    The search space should include:
    - learning_rate: float between 0.0001 and 0.01
    - batch_size: choice from [32, 64, 128]
    - num_filters: integer between 16 and 64
    
    Returns:
        Dictionary defining the search space with this exact structure:
        {
            "param_name": {
                "type": "float" | "int" | "choice",
                "low": <number>,      # for float/int types
                "high": <number>,     # for float/int types
                "options": [<list>]   # for choice type
            }
        }
    """
    # Define the search space dictionary
    # Your code should define search_space with three parameters:
    # - learning_rate (float, 0.0001 to 0.01)
    # - batch_size (choice, options: 32, 64, 128)
    # - num_filters (int, 16 to 64)
    
    search_space = {
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
        "batch_size": {"type": "choice", "options": [32, 64, 128]},
        "num_filters": {"type": "int", "low": 16, "high": 64}
    }
    
    return search_space


def sample_configuration(search_space: Dict, seed: Optional[int] = None) -> Dict:
    """
    Sample a random configuration from the search space.
    
    Args:
        search_space: Dictionary defining parameter ranges and types
        seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary mapping parameter names to sampled values
        
    Example:
        >>> space = create_search_space()
        >>> config = sample_configuration(space, seed=42)
        >>> print(config)
        {'learning_rate': 0.0034, 'batch_size': 64, 'num_filters': 45}
    """
    if seed is not None:
        random.seed(seed)
    
    config = {}
    
    # Implement sampling logic for each parameter type
    # For each parameter in search_space:
    #   - If type is "float": use random.uniform(low, high)
    #   - If type is "int": use random.randint(low, high)
    #   - If type is "choice": use random.choice(options)
    
    for param_name, param_spec in search_space.items():
        param_type = param_spec["type"]
        
        if param_type == "float":
            config[param_name] = random.uniform(param_spec["low"], param_spec["high"])
        elif param_type == "int":
            config[param_name] = random.randint(param_spec["low"], param_spec["high"])
        elif param_type == "choice":
            config[param_name] = random.choice(param_spec["options"])
    
    return config


# =============================================================================
# PART 2: Training Wrapper with Early Stopping (TODOs 3-4)
# =============================================================================

def train_with_early_stopping(
    config: Dict,
    max_epochs: int = 20,
    patience: int = 3
) -> Tuple[Dict, str]:
    """
    Train a model with the given configuration and early stopping.
    
    Early stopping logic:
    - Track the best validation loss seen so far
    - Count epochs without improvement
    - Stop if no improvement for 'patience' consecutive epochs
    
    Args:
        config: Hyperparameter configuration dictionary
        max_epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Tuple of (metrics_dict, status_string):
        - metrics_dict: {'val_acc': float, 'val_loss': float, 'epochs_trained': int}
        - status_string: "COMPLETED" if trained all epochs, "EARLY_STOPPED" if stopped early
        
    Example:
        >>> config = {'learning_rate': 0.001, 'batch_size': 64, 'num_filters': 32}
        >>> metrics, status = train_with_early_stopping(config, max_epochs=10, patience=2)
        >>> print(status)
        'COMPLETED'
    """
    trainer = MockModelTrainer(config)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Initialize tracking variables
    # You need to track:
    # - current_epoch (start at 0)
    # - final_val_acc (will store the last validation accuracy)
    # - final_val_loss (will store the last validation loss)
    
    current_epoch = 0
    final_val_acc = 0.0
    final_val_loss = float('inf')
    
    # Implement the training loop with early stopping
    # For each epoch from 0 to max_epochs-1:
    #   1. Train for one epoch (use trainer.train(epochs=current_epoch+1) and get history)
    #   2. Get current validation loss from history['val_loss'][current_epoch]
    #   3. Get current validation accuracy from history['val_acc'][current_epoch]
    #   4. Update final_val_acc and final_val_loss
    #   5. Check if current val_loss < best_val_loss:
    #      - If yes: update best_val_loss and reset epochs_without_improvement to 0
    #      - If no: increment epochs_without_improvement
    #   6. If epochs_without_improvement >= patience: break the loop
    #
    # Hint: You need to call trainer.train() with increasing epochs each iteration
    #       OR train once with all epochs and check metrics epoch by epoch
    
    status = "COMPLETED"
    for current_epoch in range(max_epochs):
        history = trainer.train(epochs=current_epoch + 1)
        
        val_loss = history['val_loss'][current_epoch]
        val_acc = history['val_acc'][current_epoch]
        
        final_val_loss = val_loss
        final_val_acc = val_acc
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            status = "EARLY_STOPPED"
            break
    
    metrics = {
        'val_acc': final_val_acc,
        'val_loss': final_val_loss,
        'epochs_trained': current_epoch + 1
    }
    
    return metrics, status


# =============================================================================
# PART 3: Experiment Logger (TODOs 5-6)
# =============================================================================

class ExperimentLogger:
    """
    Logger for tracking hyperparameter search experiments.
    
    Maintains a list of experiment records and provides formatted output.
    """
    
    def __init__(self):
        """Initialize an empty experiment log."""
        self.experiments = []
        self.experiment_count = 0
    
    # Implement the log_experiment method
    def log_experiment(
        self,
        config: Dict,
        metrics: Dict,
        status: str
    ) -> str:
        """
        Log a single experiment and return the formatted log entry.
        
        Args:
            config: Hyperparameter configuration used
            metrics: Results dictionary with 'val_acc' key
            status: Experiment status ("COMPLETED", "EARLY_STOPPED", or "FAILED")
            
        Returns:
            Formatted log string in EXACT format:
            "[EXP-XXX] lr=Y.YYYY | batch=ZZ | filters=FF | val_acc=A.AAAA | status=STATUS"
            
            Where:
            - XXX is 3-digit experiment number (001, 002, ...)
            - Y.YYYY is learning_rate with 4 decimal places
            - ZZ is batch_size (integer)
            - FF is num_filters (integer)
            - A.AAAA is val_acc with 4 decimal places
            - STATUS is the status string
            
        Example:
            >>> logger = ExperimentLogger()
            >>> config = {'learning_rate': 0.0012, 'batch_size': 64, 'num_filters': 32}
            >>> metrics = {'val_acc': 0.8934}
            >>> entry = logger.log_experiment(config, metrics, "COMPLETED")
            >>> print(entry)
            [EXP-001] lr=0.0012 | batch=64 | filters=32 | val_acc=0.8934 | status=COMPLETED
        """
        # Increment experiment count
        self.experiment_count += 1
        
        # YOUR CODE HERE
        # 1. Create the formatted log entry string
        # 2. Store the experiment record (config, metrics, status) in self.experiments
        # 3. Return the formatted string
        
        exp_id = f"{self.experiment_count:03d}"
        lr = config['learning_rate']
        batch = config['batch_size']
        filters = config['num_filters']
        acc = metrics['val_acc']
        
        log_entry = f"[EXP-{exp_id}] lr={lr:.4f} | batch={batch} | filters={filters} | val_acc={acc:.4f} | status={status}"
        
        self.experiments.append({
            'config': config,
            'metrics': metrics,
            'status': status
        })
        
        return log_entry
    
    # Implement the get_summary method
    def get_summary(self) -> str:
        """
        Generate a summary of all experiments.
        
        Returns:
            Formatted summary string in EXACT format:
            === SEARCH COMPLETE ===
            Total experiments: N
            Best config: lr=Y.YYYY, batch=ZZ, filters=FF
            Best validation accuracy: A.AAAA
            
        Example:
            >>> logger = ExperimentLogger()
            >>> # ... log some experiments ...
            >>> print(logger.get_summary())
            === SEARCH COMPLETE ===
            Total experiments: 5
            Best config: lr=0.0031, batch=64, filters=48
            Best validation accuracy: 0.9123
        """
        if not self.experiments:
            return "=== SEARCH COMPLETE ===\nTotal experiments: 0\nNo experiments logged."
        
        # YOUR CODE HERE
        # 1. Find the experiment with the highest val_acc
        # 2. Format and return the summary string
        
        best_exp = max(self.experiments, key=lambda x: x['metrics']['val_acc'])
        best_config = best_exp['config']
        
        summary = (
            f"=== SEARCH COMPLETE ===\n"
            f"Total experiments: {len(self.experiments)}\n"
            f"Best config: lr={best_config['learning_rate']:.4f}, batch={best_config['batch_size']}, filters={best_config['num_filters']}\n"
            f"Best validation accuracy: {best_exp['metrics']['val_acc']:.4f}"
        )
        
        return summary


# =============================================================================
# PART 4: Main Search Loop (TODOs 7-8)
# =============================================================================

# TODO 7: Implement the random_search function
def random_search(
    search_space: Dict,
    n_trials: int = 10,
    max_epochs: int = 15,
    patience: int = 3,
    seed: int = 42
) -> Tuple[Dict, ExperimentLogger]:
    """
    Perform random search over the hyperparameter space.
    
    Args:
        search_space: Dictionary defining the search space
        n_trials: Number of configurations to try
        max_epochs: Maximum epochs per trial
        patience: Early stopping patience
        seed: Base random seed (each trial uses seed + trial_number)
        
    Returns:
        Tuple of (best_config, logger):
        - best_config: Configuration with highest validation accuracy
        - logger: ExperimentLogger with all experiment records
        
    Process:
        1. Create an ExperimentLogger
        2. For each trial (0 to n_trials-1):
           a. Sample a configuration (use seed=seed+trial_number for reproducibility)
           b. Train with early stopping
           c. Log the experiment
           d. Print the log entry
        3. Print the summary
        4. Return the best configuration and the logger
    """
    # YOUR CODE HERE
    logger = ExperimentLogger()
    
    for i in range(n_trials):
        config = sample_configuration(search_space, seed=seed + i)
        metrics, status = train_with_early_stopping(config, max_epochs=max_epochs, patience=patience)
        log_entry = logger.log_experiment(config, metrics, status)
        print(log_entry)
        
    best_config = find_best_config(logger)
    return best_config, logger


# Implement the find_best_config function
def find_best_config(logger: ExperimentLogger) -> Dict:
    """ 
    Find the configuration with the best validation accuracy.
    
    Args:
        logger: ExperimentLogger containing experiment records
        
    Returns:
        Dictionary with the best configuration's hyperparameters
        
    Raises:
        ValueError: If no experiments have been logged
    """
    # Find the configuration with the best validation accuracy
    if not logger.experiments:
        raise ValueError("No experiments have been logged")
        
    best_exp = max(logger.experiments, key=lambda x: x['metrics']['val_acc'])
    return best_exp['config']


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main function to run the hyperparameter search.
    """
    print("=" * 60)
    print("FASHION-MNIST HYPERPARAMETER OPTIMIZER")
    print("=" * 60)
    print()
    
    # Create search space
    search_space = create_search_space()
    print("Search Space Defined:")
    for param, spec in search_space.items():
        print(f"  - {param}: {spec}")
    print()
    
    # Run random search
    print("Starting Random Search...")
    print("-" * 60)
    
    best_config, logger = random_search(
        search_space,
        n_trials=10,
        max_epochs=15,
        patience=3,
        seed=42
    )
    
    print("-" * 60)
    print()
    print(logger.get_summary())
    print()
    print("Optimization complete!")


if __name__ == "__main__":
    main()
