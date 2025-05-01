import sys
import os
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and trainer
from LSTM.model.lstm import LSTMModel
from LSTM.utils.train import LSTMTrainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        """Set up model, optimizer, and data for testing."""
        # Model parameters
        self.input_size = 5
        self.hidden_size = 32  # Using a smaller hidden size for faster testing
        self.num_layers = 1  # Using a single layer for faster testing
        self.output_size = 1
        self.device = torch.device("cpu")  # Use CPU for testing
        
        # Create model
        self.model = LSTMModel(
            self.input_size, 
            self.hidden_size, 
            self.num_layers, 
            self.output_size
        )
        
        # Create optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        
        # Create trainer
        self.trainer = LSTMTrainer(self.model, self.optimizer, self.criterion, self.device)
        
        # Create dummy dataset
        np.random.seed(42)
        n_samples = 50
        seq_length = 10
        
        # X: (n_samples, seq_length, input_size)
        X = np.random.randn(n_samples, seq_length, self.input_size)
        # y: (n_samples, output_size)
        y = np.random.randn(n_samples, self.output_size)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
    def test_train_epoch(self):
        """Test that a single training epoch runs without errors."""
        try:
            loss = self.trainer._train_epoch(self.dataloader)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Training epoch failed with error: {e}")
        
        self.assertTrue(test_passed)
        self.assertIsInstance(loss, float)
        
    def test_validate_epoch(self):
        """Test that a single validation epoch runs without errors."""
        try:
            loss = self.trainer._validate(self.dataloader)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Validation epoch failed with error: {e}")
        
        self.assertTrue(test_passed)
        self.assertIsInstance(loss, float)
    
    def test_train_early_stopping(self):
        """Test that training with early stopping works."""
        try:
            # Training with minimal epochs and patience for testing
            self.trainer.train(self.dataloader, self.dataloader, epochs=2, patience=1)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Training with early stopping failed with error: {e}")
        
        self.assertTrue(test_passed)

if __name__ == "__main__":
    unittest.main() 