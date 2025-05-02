import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We'll use statsmodels directly for this test
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

class TestARIMA(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing."""
        # Generate sample time series data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=100)
        
        # Create a time series with trend and noise
        trend = np.linspace(100, 150, 100)
        noise = np.random.normal(0, 5, 100)
        prices = trend + noise
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'close': prices
        }, index=dates)
        
    def test_arima_fit(self):
        """Test that ARIMA model fitting works correctly."""
        try:
            # Prepare the data
            train_size = int(0.8 * len(self.df))
            train_data = self.df['close'][:train_size]
            test_data = self.df['close'][train_size:]
            
            # Fit ARIMA model with simple parameters
            model = ARIMA(train_data, order=(1, 0, 0))
            model_fit = model.fit()
            
            # Make forecasts
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Check that forecast is the right shape
            self.assertEqual(len(forecast), len(test_data))
            
            # Check that forecast values are reasonable
            self.assertTrue(all(not np.isnan(x) for x in forecast))
            self.assertTrue(all(abs(x - self.df['close'].mean()) < 3 * self.df['close'].std() for x in forecast))
            
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"ARIMA fit test failed with error: {e}")
            
        self.assertTrue(test_passed)
    
    def test_arima_order_selection(self):
        """Test selection of ARIMA order parameters."""
        try:
            # Try different orders
            best_aic = float('inf')
            best_order = None
            
            # Only test a few combinations for speed
            for p in range(2):
                for d in range(2):
                    for q in range(2):
                        try:
                            model = ARIMA(self.df['close'], order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Check that we found a valid order
            self.assertIsNotNone(best_order)
            
            # Fit the best model
            best_model = ARIMA(self.df['close'], order=best_order)
            best_model_fit = best_model.fit()
            
            # Make a one-step forecast
            forecast = best_model_fit.forecast(steps=1)[0]
            
            # Check forecast value
            self.assertFalse(np.isnan(forecast))
            self.assertTrue(abs(forecast - self.df['close'].iloc[-1]) < 20)  # Should be reasonably close
            
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"ARIMA order selection test failed with error: {e}")
            
        self.assertTrue(test_passed)
        
    def test_arima_binary_classification(self):
        """Test ARIMA for binary classification (up/down trend)."""
        try:
            # Prepare the data
            train_size = int(0.8 * len(self.df))
            train_data = self.df['close'][:train_size]
            test_data = self.df['close'][train_size:]
            
            # Fit ARIMA model
            model = ARIMA(train_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Make forecasts
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Convert to binary predictions (up=1, down=0)
            # For forecast: predict up if forecast value > previous value
            actual_diff = np.diff(np.hstack([train_data.iloc[-1], test_data]))
            forecast_diff = np.diff(np.hstack([train_data.iloc[-1], forecast]))
            
            actual_trend = (actual_diff > 0).astype(int)
            predicted_trend = (forecast_diff > 0).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(actual_trend == predicted_trend)
            
            # Calculate confusion matrix elements
            tp = np.sum((actual_trend == 1) & (predicted_trend == 1))
            fp = np.sum((actual_trend == 0) & (predicted_trend == 1))
            fn = np.sum((actual_trend == 1) & (predicted_trend == 0))
            
            # Calculate F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # F1 score should be a valid value
            self.assertFalse(np.isnan(f1))
            self.assertTrue(0 <= f1 <= 1)
            
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"ARIMA binary classification test failed with error: {e}")
            
        self.assertTrue(test_passed)

if __name__ == "__main__":
    unittest.main() 