import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data utils (adjust the path if necessary)
from LSTM.utils.data_utils import prepare_sequences

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing."""
        # Create a simple dataframe with stock data
        dates = pd.date_range(start='2022-01-01', periods=100)
        self.df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(110, 210, 100),
            'low': np.random.uniform(90, 190, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
    def test_prepare_sequences(self):
        """Test that prepare_sequences creates the expected shapes."""
        features = ['open', 'high', 'low', 'close', 'volume']
        target_col = 'close'
        seq_len = 10
        forecast_horizon = 1
        
        # Generate sequences
        X, y, scaler = prepare_sequences(self.df, features, target_col, seq_len, forecast_horizon)
        
        # Expected shapes
        expected_X_shape = (100 - seq_len - forecast_horizon + 1, seq_len, len(features))
        expected_y_shape = (100 - seq_len - forecast_horizon + 1, forecast_horizon)
        
        # Check shapes
        self.assertEqual(X.shape[0], expected_X_shape[0])
        self.assertEqual(X.shape[1], expected_X_shape[1])
        self.assertEqual(X.shape[2], expected_X_shape[2])
        
        self.assertEqual(y.shape[0], expected_y_shape[0])
        self.assertEqual(y.shape[1], expected_y_shape[1])
        
    def test_scaling(self):
        """Test that values are properly scaled."""
        features = ['open', 'high', 'low', 'close', 'volume']
        target_col = 'close'
        seq_len = 10
        forecast_horizon = 1
        
        # Generate sequences
        X, y, scaler = prepare_sequences(self.df, features, target_col, seq_len, forecast_horizon)
        
        # All values should be between 0 and 1 after scaling
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(X <= 1))
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 1))

if __name__ == "__main__":
    unittest.main() 