#!/usr/bin/env python
"""
StockPredictor main class for LSTM-based stock price prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path so Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now try to import the module with the corrected path
try:
    from models.lstm_model import LSTMModel
except ModuleNotFoundError:
    # If that still doesn't work, try relative import
    try:
        from .lstm_model import LSTMModel
    except ImportError:
        # If all else fails, look for the file in the same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        try:
            from lstm_model import LSTMModel
        except ImportError:
            print("ERROR: Could not import LSTMModel. Make sure lstm_model.py is in the same directory or in a 'models' subdirectory.")
            sys.exit(1)

# Import baseline models
try:
    from models.baseline_models import MovingAverageBaseline, DirectionBaseline
except ModuleNotFoundError:
    try:
        from .baseline_models import MovingAverageBaseline, DirectionBaseline
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        try:
            from baseline_models import MovingAverageBaseline, DirectionBaseline
        except ImportError:
            print("WARNING: Could not import baseline models. Baseline comparison will not be available.")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import glob
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

# Import from the src package
from utils.trainer import LSTMTrainer
from data.stock_data_loader import StockDataLoader

class StockPredictor:
    """
    Class for training and using LSTM models for stock price prediction
    """
    def __init__(self, ticker, start_date=None, end_date=None, sequence_length=20):
        """
        Initialize the stock predictor
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (default: 5 years ago)
            end_date: End date for data (default: today)
            sequence_length: Length of sequence for prediction
        """
        self.ticker = ticker
        self.data_loader = StockDataLoader(ticker, start_date, end_date)
        self.model = None
        self.scaler = None
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_model_path = None
        print(f"Using device: {self.device}")
        
    def prepare_data(self):
        """
        Prepare data for training
        
        Returns:
            train_loader, test_loader: DataLoader objects for training and testing
        """
        # Download and prepare data
        self.data_loader.fetch_data()
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        # Preprocess data with the specified sequence length
        processed_data = self.data_loader.preprocess_data(sequence_length=self.sequence_length)
        
        if processed_data is None:
            raise ValueError("Failed to preprocess data. Check if data is available.")
        
        # Extract components from processed data
        train_loader = processed_data['train_loader']
        test_loader = processed_data['val_loader']
        test_X = processed_data['X_val']
        test_y = processed_data['y_val']
        self.scaler = processed_data['scaler']
        
        # Store feature columns for later use
        self.data_loader.feature_columns = processed_data['feature_columns']
        
        # Convert test data to tensors if not already
        if not isinstance(test_X, torch.Tensor):
            test_X = torch.FloatTensor(test_X).to(self.device)
            test_y = torch.FloatTensor(test_y).to(self.device)
        
        return train_loader, test_loader, test_X, test_y
        
    def train_model(self, hidden_size=128, num_layers=2, learning_rate=0.001, epochs=100):
        """
        Train the LSTM model
        
        Args:
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            
        Returns:
            history: Dictionary containing training history
        """
        # Prepare data
        train_loader, test_loader, test_X, test_y = self.prepare_data()
        
        # Initialize model
        input_size = len(self.data_loader.feature_columns)
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3  # Higher dropout for financial data to prevent overfitting
        ).to(self.device)
        
        # Initialize trainer
        trainer = LSTMTrainer(self.model, learning_rate=learning_rate)
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/models', exist_ok=True)
        
        # Generate model save path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = f'outputs/models/{self.ticker}_model_{timestamp}.pth'
        self.latest_model_path = model_save_path
        
        # Train model
        history = trainer.train(
            train_loader, 
            test_loader, 
            epochs=epochs, 
            save_path=model_save_path
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        # Make predictions on test data
        self.evaluate_model(test_X, test_y)
        
        return history
    
    def plot_training_history(self, history):
        """
        Plot training and validation loss
        """
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_losses'], label='Training Loss')
        plt.plot(history['valid_losses'], label='Validation Loss')
        plt.title(f'{self.ticker} LSTM Model Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"outputs/figures/{self.ticker}_training_history.png")
        plt.close()
        
    def evaluate_model(self, test_X, test_y, model_path=None):
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
            
        # Load best model if path provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.latest_model_path = model_path
        elif self.latest_model_path:
            self.model.load_state_dict(torch.load(self.latest_model_path))
        
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(test_X).cpu().numpy()
            
        # Convert predictions and actual values back to original scale
        actual_prices = self.data_loader.inverse_transform_predictions(test_y.cpu().numpy())
        predicted_prices = self.data_loader.inverse_transform_predictions(predictions)
        
        # Calculate metrics
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        r2 = r2_score(actual_prices, predicted_prices)
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        # Plot predictions vs actual
        self.plot_predictions(actual_prices, predicted_prices)
        
        # Compare with baselines if available
        try:
            self.compare_with_baselines(actual_prices, predicted_prices)
        except (NameError, ImportError):
            print("Baseline comparison not available. Skipping baseline comparison.")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'actual': actual_prices,
            'predicted': predicted_prices
        }
    
    def plot_predictions(self, actual_prices, predicted_prices):
        """
        Plot actual vs predicted prices
        """
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual Price', alpha=0.8)
        plt.plot(predicted_prices, label='Predicted Price', alpha=0.8)
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"outputs/figures/{self.ticker}_predictions.png")
        plt.close()
        
    def find_latest_model(self):
        """
        Find the latest model file for this ticker
        
        Returns:
            Path to the latest model file
        """
        # Get all model files for this ticker
        model_files = glob.glob(f'outputs/models/{self.ticker}_model_*.pth')
        
        if not model_files:
            # Fallback to best_model.pth if no ticker-specific models found
            if os.path.exists('outputs/models/best_model.pth'):
                return 'outputs/models/best_model.pth'
            return None
            
        # Sort by modification time, newest first
        latest_model = max(model_files, key=os.path.getmtime)
        return latest_model
        
    def predict_future(self, days=30):
        """
        Predict future stock prices
        
        Args:
            days: Number of days to predict into the future
            
        Returns:
            future_predictions: Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
            
        # Find and load the latest model
        model_path = self.latest_model_path or self.find_latest_model()
        
        if not model_path:
            raise ValueError("No model found. Please train a model first.")
            
        print(f"Using model: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Get the most recent data
        recent_data = self.data_loader.get_recent_data(self.sequence_length)
        
        # Scale the data
        scaled_data = self.scaler.transform(recent_data[self.data_loader.feature_columns].values)
        
        # Reshape for LSTM input [batch_size, sequence_length, input_size]
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        last_sequence = torch.FloatTensor(last_sequence).to(self.device)
        
        # Make predictions for future days
        future_predictions = []
        
        for _ in range(days):
            with torch.no_grad():
                # Predict next day
                next_day_scaled = self.model(last_sequence).cpu().numpy()[0, 0]
                future_predictions.append(next_day_scaled)
                
                # Create a new sequence by removing the first day and adding the prediction
                new_sequence = last_sequence.cpu().numpy()[0, 1:, :]
                
                # Find the index of Close price
                close_idx = self.data_loader.feature_columns.index('Close')
                
                # Update the Close price with our prediction
                new_row = new_sequence[-1, :].copy()
                new_row[close_idx] = next_day_scaled
                
                # Append the new row to create the next sequence
                next_sequence = np.vstack([new_sequence, new_row.reshape(1, -1)])
                last_sequence = torch.FloatTensor(next_sequence.reshape(1, self.sequence_length, -1)).to(self.device)
        
        # Convert predictions back to original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        original_scale_predictions = self.data_loader.inverse_transform_predictions(future_predictions)
        
        # Create a date range for future predictions
        last_date = recent_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        # Fetch current real market price for validation
        try:
            current_data = yf.download(self.ticker, period="5d")
            real_close = float(current_data['Close'].iloc[-1])
            model_price = float(recent_data['Close'].iloc[-1])
            
            # If there's a large discrepancy between model price and real price, apply scaling correction
            if abs((real_close - model_price) / real_close) > 0.2:  # More than 20% different
                print(f"WARNING: Price discrepancy detected. Model: ${model_price:.2f}, Real: ${real_close:.2f}")
                
                # Calculate scaling factor based on real market price
                scaling_factor = real_close / original_scale_predictions[0]
                original_scale_predictions = original_scale_predictions * scaling_factor
                
                print(f"Applied scaling correction. New first prediction: ${original_scale_predictions[0]:.2f}")
        except Exception as e:
            print(f"Could not fetch real-time price data for validation: {e}")
        
        # Plot predictions
        self.plot_future_predictions(original_scale_predictions, future_dates)
        
        # Create a DataFrame with predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': original_scale_predictions
        })
        
        return future_df
    
    def plot_future_predictions(self, predictions, dates):
        """
        Plot future predictions
        """
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        # Get historical data for context
        historical_data = self.data_loader.processed_data.tail(60)
        
        plt.figure(figsize=(14, 7))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data['Close'], label='Historical Price', color='blue')
        
        # Plot predictions
        plt.plot(dates, predictions, label='Predicted Price', color='red', linestyle='--')
        
        # Add shading to indicate prediction region
        plt.axvspan(dates[0], dates[-1], alpha=0.2, color='gray')
        
        plt.title(f'{self.ticker} Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"outputs/figures/{self.ticker}_future_predictions.png")
        plt.close()
        
        return 

    def load_model(self, model_path):
        """
        Load a model from a file
        
        Args:
            model_path: Path to the model file
            
        Returns:
            The loaded model
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} not found")
            
        # Get input size from feature columns
        input_size = len(self.data_loader.feature_columns)
        
        # Create model with appropriate architecture
        # We'll need to determine the architecture from the model file
        # For now, let's try to load the state dict and see if it works
        
        # First, create a model with default architecture
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        try:
            # Try to load the state dict
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded successfully from {model_path}")
            self.model = model
            self.latest_model_path = model_path
            return model
        except Exception as e:
            print(f"Error loading model with default architecture: {e}")
            
            # Try with a larger architecture (for enhanced models)
            model = LSTMModel(
                input_size=input_size,
                hidden_size=256,
                num_layers=3,
                dropout=0.4
            ).to(self.device)
            
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded successfully with enhanced architecture from {model_path}")
                self.model = model
                self.latest_model_path = model_path
                return model
            except Exception as e2:
                raise ValueError(f"Failed to load model: {e2}") 

    def inverse_transform_predictions(self, scaled_predictions):
        """
        Convert scaled predictions back to original scale
        
        Args:
            scaled_predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        # Create a dummy array with zeros
        dummy_array = np.zeros((len(scaled_predictions), len(self.data_loader.feature_columns)))
        
        # Find the index of Close price
        close_idx = self.data_loader.feature_columns.index('Close')
        
        # Put the predictions in the right column
        dummy_array[:, close_idx] = scaled_predictions.flatten()
        
        # Inverse transform
        original_scale_array = self.scaler.inverse_transform(dummy_array)
        
        # Extract the Close price
        original_scale_predictions = original_scale_array[:, close_idx]
        
        # Add a double-check to ensure values are reasonable (in case of extreme scaling issues)
        # Get the last known price from raw data
        last_known_price = self.raw_data['Close'].iloc[-1]
        
        # If predictions are more than 50% different from last known price, adjust scaling
        first_pred = original_scale_predictions[0]
        if abs((first_pred - last_known_price) / last_known_price) > 0.5:
            print(f"WARNING: Scaling issue detected. First prediction: ${first_pred:.2f}, Last known price: ${last_known_price:.2f}")
            print("Applying scaling correction...")
            
            # Simple scaling correction - adjust to be around the last known price
            scaling_factor = last_known_price / first_pred
            original_scale_predictions = original_scale_predictions * scaling_factor
            
            print(f"Adjusted first prediction: ${original_scale_predictions[0]:.2f}")
        
        return original_scale_predictions 

    def compare_with_baselines(self, actual_prices, predicted_prices):
        """
        Compare model performance with baseline models
        
        Args:
            actual_prices: Actual price values
            predicted_prices: Predicted price values from LSTM model
        """
        print("\nComparing LSTM model with baselines...")
        
        # Calculate LSTM model metrics
        lstm_metrics = {
            'mse': mean_squared_error(actual_prices, predicted_prices),
            'rmse': np.sqrt(mean_squared_error(actual_prices, predicted_prices)),
            'mae': mean_absolute_error(actual_prices, predicted_prices),
            'r2': r2_score(actual_prices, predicted_prices)
        }
        
        # Get the raw data for comparison
        raw_data = self.data_loader.data.copy()
        raw_prices = raw_data['Close'].values
        
        # Calculate direction for F1 score (1=up, 0=down/flat)
        actual_direction = (np.diff(actual_prices) > 0).astype(int)
        lstm_direction = (np.diff(predicted_prices) > 0).astype(int)
        lstm_metrics['f1_score'] = f1_score(actual_direction, lstm_direction)
        
        # Create Moving Average baseline
        ma_window = min(20, len(raw_prices) // 10)  # Use 20 or 10% of data points, whichever is smaller
        ma_baseline = MovingAverageBaseline(window_size=ma_window)
        ma_predictions = ma_baseline.predict(raw_prices)
        
        # Calculate Moving Average metrics
        ma_metrics = {
            'mse': mean_squared_error(raw_prices[ma_window:], ma_predictions[ma_window:]),
            'rmse': np.sqrt(mean_squared_error(raw_prices[ma_window:], ma_predictions[ma_window:])),
            'mae': mean_absolute_error(raw_prices[ma_window:], ma_predictions[ma_window:]),
            'r2': r2_score(raw_prices[ma_window:], ma_predictions[ma_window:])
        }
        
        # Calculate direction for Moving Average
        ma_direction = (np.diff(ma_predictions) > 0).astype(int)
        raw_direction = (np.diff(raw_prices) > 0).astype(int)
        ma_metrics['f1_score'] = f1_score(raw_direction[ma_window-1:], ma_direction[ma_window-1:])
        
        # Create Direction baseline
        direction_baseline = DirectionBaseline(strategy='momentum', lookback=5)
        baseline_directions, _ = direction_baseline.predict(raw_prices)
        direction_metrics = {
            'f1_score': f1_score(raw_direction[5:], baseline_directions[5:])
        }
        
        # Print comparison
        print("\nMetrics Comparison:")
        print(f"{'Metric':<10} {'LSTM':<10} {'MA Baseline':<15} {'Improvement':<15}")
        print("-" * 50)
        
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            improvement = ((ma_metrics[metric] - lstm_metrics[metric]) / ma_metrics[metric]) * 100
            # Handle R² differently (higher is better)
            if metric == 'r2':
                improvement = ((lstm_metrics[metric] - ma_metrics[metric]) / (1 - ma_metrics[metric])) * 100
            
            # Format improvement as percentage with sign
            if improvement >= 0:
                improvement_str = f"+{improvement:.2f}%"
            else:
                improvement_str = f"{improvement:.2f}%"
                
            print(f"{metric.upper():<10} {lstm_metrics[metric]:<10.4f} {ma_metrics[metric]:<15.4f} {improvement_str:<15}")
        
        # Handle F1 score separately (direction prediction)
        if 'f1_score' in lstm_metrics and 'f1_score' in direction_metrics:
            improvement = ((lstm_metrics['f1_score'] - direction_metrics['f1_score']) / direction_metrics['f1_score']) * 100
            improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
            print(f"{'F1 SCORE':<10} {lstm_metrics['f1_score']:<10.4f} {direction_metrics['f1_score']:<15.4f} {improvement_str:<15}")
        
        # Calculate improvement over baseline F1 = 0.6937
        target_f1 = 0.6937
        if 'f1_score' in lstm_metrics:
            f1_improvement = ((lstm_metrics['f1_score'] - target_f1) / target_f1) * 100
            f1_improvement_str = f"+{f1_improvement:.2f}%" if f1_improvement >= 0 else f"{f1_improvement:.2f}%"
            print(f"{'F1 TARGET':<10} {lstm_metrics['f1_score']:<10.4f} {target_f1:<15.4f} {f1_improvement_str:<15}")
        
        # Plot comparison
        self.plot_baseline_comparison(raw_prices, ma_predictions, ma_window)
            
        return {
            'lstm': lstm_metrics,
            'ma_baseline': ma_metrics,
            'direction_baseline': direction_metrics
        }
    
    def plot_baseline_comparison(self, actual_prices, baseline_predictions, window_size):
        """
        Plot actual prices vs baseline predictions
        
        Args:
            actual_prices: Actual price values
            baseline_predictions: Baseline model predictions
            window_size: Window size used for moving average
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual prices
        plt.plot(actual_prices, label='Actual Prices', color='blue')
        
        # Plot baseline predictions (skip the first window_size values)
        valid_predictions = baseline_predictions[window_size:]
        valid_indices = range(window_size, len(actual_prices))
        plt.plot(valid_indices, valid_predictions, label=f'MA({window_size}) Baseline', color='red', linestyle='--')
        
        plt.title(f'{self.ticker} Stock Price - Model vs Baseline', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save figure
        plt.savefig(f"outputs/figures/{self.ticker}_model_vs_baseline.png")
        plt.close()
        
        print(f"Baseline comparison plot saved to outputs/figures/{self.ticker}_model_vs_baseline.png") 