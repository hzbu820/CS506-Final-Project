import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
import glob
import yfinance as yf

# Import from the src package
from models.lstm_model import LSTMModel
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
        self.data_loader = StockDataLoader(ticker, start_date, end_date, sequence_length)
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
        self.data_loader.prepare_features()
        self.data_loader.scale_data()
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        # Visualize the data
        self.data_loader.plot_stock_data(f"outputs/figures/{self.ticker}_stock_history.png")
        
        # Get training data
        train_X, train_y, test_X, test_y, self.scaler = self.data_loader.prepare_data_for_training()
        
        # Convert to PyTorch tensors
        train_X = torch.FloatTensor(train_X).to(self.device)
        train_y = torch.FloatTensor(train_y).to(self.device)
        test_X = torch.FloatTensor(test_X).to(self.device)
        test_y = torch.FloatTensor(test_y).to(self.device)
        
        # Create DataLoader objects
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
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
        output_size = 1  # We're predicting only the closing price
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.3  # Higher dropout for financial data to prevent overfitting
        ).to(self.device)
        
        # Initialize trainer
        trainer = LSTMTrainer(self.model, learning_rate=learning_rate)
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/models', exist_ok=True)
        
        # Generate model save path with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = f'outputs/models/{self.ticker}_model_{timestamp}.pth'
        self.latest_model_path = model_save_path
        
        # Train model
        history = trainer.train(
            train_loader, 
            test_loader, 
            epochs=epochs, 
            model_save_path=model_save_path
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
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
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
        
        # Calculate direction accuracy
        actual_direction = np.diff(actual_prices) > 0
        predicted_direction = np.diff(predicted_prices) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        print(f"\nEvaluation Metrics for {self.ticker}:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.4f}")
        
        # Plot predictions vs actual
        self.plot_predictions(actual_prices, predicted_prices)
        
        return mse, rmse, mae, direction_accuracy
    
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
            output_size=1,
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
                output_size=1,
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