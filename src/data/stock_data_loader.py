import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

class StockDataLoader:
    """
    Class for downloading and preprocessing stock data
    """
    def __init__(self, ticker, start_date=None, end_date=None, sequence_length=20):
        """
        Initialize the stock data loader
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (default: 5 years ago)
            end_date: End date for data (default: today)
            sequence_length: Length of sequence for prediction
        """
        self.ticker = ticker
        
        # Set default dates if not provided
        if start_date is None:
            self.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        self.sequence_length = sequence_length
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
        self.scaler = None
        
    def download_data(self):
        """
        Download stock data from Yahoo Finance
        
        Returns:
            DataFrame with stock data
        """
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        self.raw_data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        print(f"Downloaded {len(self.raw_data)} days of data.")
        
        # Ensure data is sorted by date
        self.raw_data = self.raw_data.sort_index()
        
        return self.raw_data
    
    def prepare_features(self):
        """
        Prepare features for stock prediction
        
        Returns:
            DataFrame with features
        """
        if self.raw_data is None:
            self.download_data()
            
        data = self.raw_data.copy()
        
        # Technical indicators
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_1d'] = data['Close'].pct_change(periods=1)
        data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
        data['Price_Change_20d'] = data['Close'].pct_change(periods=20)
        
        # Volatility
        data['Volatility_5d'] = data['Close'].pct_change().rolling(window=5).std()
        data['Volatility_20d'] = data['Close'].pct_change().rolling(window=20).std()
        
        # Volume indicators
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().abs()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Drop NaN values
        data = data.dropna()
        
        # Define feature columns
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                               'MA5', 'MA20', 'MA50', 
                               'Price_Change', 'Price_Change_1d', 'Price_Change_5d', 'Price_Change_20d',
                               'Volatility_5d', 'Volatility_20d',
                               'Volume_Change', 'Volume_MA5',
                               'RSI', 'MACD', 'MACD_Signal']
        
        self.processed_data = data
        
        return self.processed_data
    
    def scale_data(self):
        """
        Scale the data using Min-Max scaling
        
        Returns:
            Scaled DataFrame
        """
        if self.processed_data is None:
            self.prepare_features()
            
        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.processed_data[self.feature_columns].values)
        
        # Create a new DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=self.feature_columns, index=self.processed_data.index)
        
        self.processed_data[self.feature_columns] = scaled_df[self.feature_columns]
        
        return self.processed_data
    
    def prepare_data_for_training(self, test_size=0.2):
        """
        Prepare data for LSTM training
        
        Args:
            test_size: Proportion of data to use for testing
            
        Returns:
            train_X, train_y, test_X, test_y: Training and testing data
        """
        if self.processed_data is None:
            self.scale_data()
            
        # Create sequences
        X, y = [], []
        data = self.processed_data[self.feature_columns].values
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            # We'll predict the closing price for the next day
            close_idx = self.feature_columns.index('Close')
            y.append(data[i + self.sequence_length, close_idx])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape y to match the expected output dimensions
        y = y.reshape(-1, 1)
        
        # Split data into training and testing sets
        split_idx = int(len(X) * (1 - test_size))
        train_X, test_X = X[:split_idx], X[split_idx:]
        train_y, test_y = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {train_X.shape}")
        print(f"Testing data shape: {test_X.shape}")
        
        return train_X, train_y, test_X, test_y, self.scaler
    
    def inverse_transform_predictions(self, scaled_predictions):
        """
        Convert scaled predictions back to original scale
        
        Args:
            scaled_predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        # Create a dummy array with zeros
        dummy_array = np.zeros((len(scaled_predictions), len(self.feature_columns)))
        
        # Find the index of Close price
        close_idx = self.feature_columns.index('Close')
        
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
        if np.mean(original_scale_predictions) < last_known_price * 0.5 or np.mean(original_scale_predictions) > last_known_price * 2:
            print(f"WARNING: Scaling issue detected. Mean prediction: ${np.mean(original_scale_predictions):.2f}, Last known price: ${last_known_price:.2f}")
            print("Applying scaling correction...")
            
            # Simple scaling correction - adjust to be around the last known price
            scaling_factor = last_known_price / original_scale_predictions[0]
            original_scale_predictions = original_scale_predictions * scaling_factor
            
            print(f"Adjusted mean prediction: ${np.mean(original_scale_predictions):.2f}")
        
        return original_scale_predictions
    
    def plot_stock_data(self, save_path=None):
        """
        Plot stock data with moving averages
        
        Args:
            save_path: Path to save the figure
        """
        if self.processed_data is None:
            self.prepare_features()
            
        plt.figure(figsize=(14, 7))
        
        # Plot stock price and moving averages
        plt.plot(self.processed_data.index, self.processed_data['Close'], label='Close Price')
        plt.plot(self.processed_data.index, self.processed_data['MA5'], label='MA5')
        plt.plot(self.processed_data.index, self.processed_data['MA20'], label='MA20')
        plt.plot(self.processed_data.index, self.processed_data['MA50'], label='MA50')
        
        plt.title(f'{self.ticker} Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ensure the directory exists
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.close()
        
    def get_recent_data(self, n_days):
        """
        Get the most recent n_days of data
        
        Args:
            n_days: Number of days of recent data to return
            
        Returns:
            DataFrame with recent data
        """
        if self.processed_data is None:
            self.prepare_features()
            
        return self.processed_data.tail(n_days) 