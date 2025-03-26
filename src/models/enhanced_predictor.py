import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.stock_predictor import StockPredictor
from models.lstm_model import LSTMModel
from utils.trainer import LSTMTrainer
from data.stock_data_loader import StockDataLoader

class EnhancedPredictor(StockPredictor):
    """
    Enhanced predictor with continued training capabilities
    """
    def __init__(self, ticker, start_date=None, end_date=None, sequence_length=30, 
                 model_path=None, scaler_path=None):
        """
        Initialize the enhanced predictor
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (default: 5 years ago)
            end_date: End date for data (default: today)
            sequence_length: Length of sequence for prediction
            model_path: Path to a previously trained model (for continued training)
            scaler_path: Path to a previously fitted scaler (for continued training)
        """
        # Initialize with standard StockPredictor
        super().__init__(ticker, start_date, end_date, sequence_length)
        
        # Additional attributes for enhanced predictor
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        # Load previous model if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading previous model from {model_path}")
            self.load_model(model_path)
            
        # Load previous scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading previous scaler from {scaler_path}")
            self.load_scaler(scaler_path)
    
    def train_model(self, hidden_size=256, num_layers=3, learning_rate=0.001, epochs=150):
        """
        Train the LSTM model with enhanced parameters
        
        Args:
            hidden_size: Number of hidden units in LSTM (default: 256, more capacity)
            num_layers: Number of LSTM layers (default: 3, deeper network)
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            
        Returns:
            history: Dictionary containing training history
        """
        # Prepare data using standard method
        train_loader, test_loader, test_X, test_y = self.prepare_data()
        
        # Initialize model with enhanced architecture
        input_size = len(self.data_loader.feature_columns)
        output_size = 1
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.4  # Higher dropout for better regularization
        ).to(self.device)
        
        # Initialize trainer with enhanced learning rate schedule
        trainer = LSTMTrainer(self.model, learning_rate=learning_rate)
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/models', exist_ok=True)
        
        # Train model and save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = f'outputs/models/{self.ticker}_model_{timestamp}.pth'
        
        history = trainer.train(
            train_loader, 
            test_loader, 
            epochs=epochs, 
            model_save_path=model_save_path
        )
        
        # Save the training history
        self.training_history = history
        
        # Save scaler for future use
        scaler_save_path = f'outputs/models/{self.ticker}_scaler_{timestamp}.pkl'
        self.save_scaler(scaler_save_path)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Make predictions on test data
        self.evaluate_model(test_X, test_y)
        
        return history
    
    def continue_training(self, new_start_date=None, new_end_date=None, learning_rate=0.0005, epochs=50):
        """
        Continue training an existing model with new data
        
        Args:
            new_start_date: Start date for new data
            new_end_date: End date for new data
            learning_rate: Learning rate for optimization (lower for fine-tuning)
            epochs: Number of training epochs
            
        Returns:
            history: Dictionary containing training history
        """
        if self.model is None:
            raise ValueError("No model to continue training. Call load_model() or train_model() first.")
            
        if self.scaler is None:
            raise ValueError("No scaler available. Cannot continue training without a scaler.")
        
        # Update the data loader with new dates if provided
        if new_start_date or new_end_date:
            old_loader = self.data_loader
            self.data_loader = StockDataLoader(
                self.ticker, 
                start_date=new_start_date or self.data_loader.start_date, 
                end_date=new_end_date or self.data_loader.end_date, 
                sequence_length=self.sequence_length
            )
            self.data_loader.scaler = old_loader.scaler
        
        # Prepare data for continued training
        train_loader, test_loader, test_X, test_y = self.prepare_data()
        
        # Store the length of current training history
        self.continued_epoch = len(self.training_history['train_loss'])
        
        # Initialize trainer with lower learning rate for fine-tuning
        trainer = LSTMTrainer(self.model, learning_rate=learning_rate)
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/models', exist_ok=True)
        
        # Continue training
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = f'outputs/models/{self.ticker}_model_continued_{timestamp}.pth'
        
        history = trainer.train(
            train_loader, 
            test_loader, 
            epochs=epochs, 
            model_save_path=model_save_path
        )
        
        # Update the training history
        for key in history:
            self.training_history[key].extend(history[key])
        
        # Plot complete training history
        self.plot_continued_training_history()
        
        # Make predictions on test data
        self.evaluate_model(test_X, test_y)
        
        return history
    
    def load_model(self, model_path):
        """
        Load a previously trained model
        
        Args:
            model_path: Path to the model file
        """
        # Determine input size from the data loader
        input_size = len(self.data_loader.feature_columns)
        
        # Create a model with the correct input size
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=256,
            num_layers=3,
            output_size=1,
            dropout=0.4
        ).to(self.device)
        
        # Load the state dictionary
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        
    def save_scaler(self, scaler_path):
        """
        Save the scaler for future use
        
        Args:
            scaler_path: Path to save the scaler
        """
        if self.scaler is None:
            raise ValueError("No scaler to save. Prepare data first.")
            
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"Scaler saved to {scaler_path}")
        
    def load_scaler(self, scaler_path):
        """
        Load a previously saved scaler
        
        Args:
            scaler_path: Path to the scaler file
        """
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        print(f"Scaler loaded from {scaler_path}")
        
    def plot_continued_training_history(self):
        """
        Plot the complete training history including continued training
        """
        if not self.training_history['train_loss']:
            raise ValueError("No training history available.")
            
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        
        # Add a vertical line to separate initial and continued training
        if hasattr(self, 'continued_epoch') and self.continued_epoch < len(self.training_history['train_loss']):
            plt.axvline(x=self.continued_epoch, color='r', linestyle='--', 
                        label='Continued Training Started')
            
        plt.title(f'{self.ticker} LSTM Model Complete Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"outputs/figures/{self.ticker}_complete_training_history.png")
        plt.close()
        
    def evaluate_model(self, test_X, test_y, model_path=None):
        """
        Evaluate the model on test data
        
        Args:
            test_X: Test features
            test_y: Test targets
            model_path: Path to the model to evaluate (optional)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
            
        # Load the model if path is provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            
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