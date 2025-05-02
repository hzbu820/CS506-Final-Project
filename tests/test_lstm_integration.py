import sys
import os
import unittest
import tempfile
import numpy as np
import pandas as pd
import torch
import yaml
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from LSTM.model.lstm import LSTMModel
from LSTM.utils.data_utils import prepare_sequences
from LSTM.utils.train import LSTMTrainer
from LSTM.utils.evaluate import compute_metrics, plot_predictions


class TestLSTMIntegration(unittest.TestCase):
    def setUp(self):
        """Create sample data and configuration for testing."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create sample stock data
        np.random.seed(42)
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(120)]

        # Generate price data with a trend
        close_price = np.linspace(100, 150, 120) + np.random.normal(0, 5, 120)
        open_price = close_price - np.random.uniform(0, 2, 120)
        high_price = close_price + np.random.uniform(0, 2, 120)
        low_price = open_price - np.random.uniform(0, 2, 120)
        volume = np.random.uniform(1000, 5000, 120)

        # Create DataFrame
        self.df = pd.DataFrame({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }, index=dates)

        # Save to CSV for testing
        self.csv_path = os.path.join(self.test_dir, "test_stock_data.csv")
        self.df.to_csv(self.csv_path)

        # Create a test config
        self.config = {
            "data_path": self.csv_path,
            "target_column": "close",
            "features": ["open", "high", "low", "close", "volume"],
            "model": {
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "seq_length": 10,
                "forecast_horizon": 1
            },
            "training": {
                "batch_size": 8,
                "epochs": 2,
                "patience": 1,
                "learning_rate": 0.001
            }
        }

        # Save config to YAML
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        """Clean up temporary files after test."""
        # Remove temporary files
        os.remove(self.csv_path)
        os.remove(self.config_path)
        os.rmdir(self.test_dir)


if __name__ == "__main__":
    unittest.main() 