import random
import math

class MockModelTrainer:
    """
    Mock trainer for Fashion-MNIST hyperparameter optimization.
    Simulates training metrics based on provided hyperparameters.
    """
    
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.num_filters = config.get('num_filters', 32)
        
        # Determine "quality" of hyperparameters to simulate different results
        # Best LR is around 0.001, Best Batch is 64, Best filters is 48
        lr_factor = max(0, 1 - abs(math.log10(self.learning_rate) - math.log10(0.001)))
        batch_factor = 1.0 if self.batch_size == 64 else 0.8
        filter_factor = max(0, 1 - abs(self.num_filters - 48) / 48)
        
        self.max_potential_acc = 0.85 + (0.12 * (lr_factor * 0.5 + batch_factor * 0.2 + filter_factor * 0.3))
        self.min_loss = 0.2 + (0.3 * (1 - lr_factor))
        
        self.history = {
            'val_loss': [],
            'val_acc': []
        }

    def train(self, epochs):
        """
        Simulate training metrics for the given number of epochs.
        """
        # We simulate epoch by epoch to maintain consistency if called multiple times
        start_epoch = len(self.history['val_loss'])
        
        for i in range(start_epoch, epochs):
            # Use a seed for consistent results per trial based on config hash
            # but slightly randomized per epoch
            random.seed(hash(str(self.config)) + i)
            
            # Simulated learning curve: starts low, increases/decreases asymptotically
            progress = 1 - math.exp(-i / 4)
            
            acc = 0.4 + (self.max_potential_acc - 0.4) * progress
            acc += random.uniform(-0.01, 0.01) # Add slight noise
            
            loss = 1.5 * math.exp(-i / 3) + self.min_loss
            loss += random.uniform(-0.02, 0.02) # Add slight noise
            
            self.history['val_acc'].append(round(min(acc, 0.99), 4))
            self.history['val_loss'].append(round(max(loss, 0.05), 4))
            
        return self.history
