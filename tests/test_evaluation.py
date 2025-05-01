import sys
import os
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation utilities
from LSTM.utils.evaluate import compute_metrics, plot_predictions

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        self.true_values = np.random.normal(0, 1, 100)
        self.pred_values = self.true_values + np.random.normal(0, 0.2, 100)  # Add some noise
        
    def test_compute_metrics(self):
        """Test that metrics computation works correctly."""
        metrics = compute_metrics(self.true_values, self.pred_values)
        
        # Check that all expected metrics are returned
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('R2', metrics)
        
        # Check that metric values are reasonable
        self.assertGreater(metrics['MSE'], 0)
        self.assertGreater(metrics['RMSE'], 0)
        self.assertGreater(metrics['MAE'], 0)
        self.assertLess(metrics['MSE'], 1)  # Should be small since pred is close to true
        
        # Check that RMSE is sqrt of MSE
        self.assertAlmostEqual(metrics['RMSE'], np.sqrt(metrics['MSE']), places=6)
        
    def test_plot_predictions(self):
        """Test that the plotting function works without errors."""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create some dates
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(len(self.true_values))]
        
        # Test plotting function
        try:
            plot_predictions(dates, self.true_values, self.pred_values, "test_plot.png")
            os.remove("test_plot.png")  # Clean up
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Plot prediction failed with error: {e}")
        
        self.assertTrue(test_passed)

if __name__ == "__main__":
    unittest.main() 