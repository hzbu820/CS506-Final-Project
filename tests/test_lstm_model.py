import sys
import os
import unittest
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model
from LSTM.model.lstm import LSTMModel

class TestLSTMModel(unittest.TestCase):
    def test_model_output_shape(self):
        """Test that the model produces the expected output shape."""
        batch_size = 32
        seq_length = 24
        input_size = 5
        hidden_size = 64
        num_layers = 2
        output_size = 1
        
        # Create model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Get output
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, output_size))
    
    def test_model_forward_pass(self):
        """Test that the model forward pass runs without errors."""
        batch_size = 16
        seq_length = 24
        input_size = 5
        hidden_size = 64
        num_layers = 2
        output_size = 1
        
        # Create model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Run forward pass
        try:
            model(x)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Forward pass failed with error: {e}")
        
        self.assertTrue(test_passed)

if __name__ == "__main__":
    unittest.main() 