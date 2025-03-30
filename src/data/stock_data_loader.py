"""
Stock data loader class for fetching and preprocessing stock data
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

class StockDataLoader:
    """Class for loading and preprocessing stock data"""
    
    def __init__(self, ticker, start_date=None, end_date=None):
        """
        Initialize the data loader
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data (format: 'YYYY-MM-DD')
            end_date (str): End date for data (format: 'YYYY-MM-DD')
        """
        self.ticker = ticker
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            # Default to 5 years of data
            start_datetime = datetime.now() - timedelta(days=5*365)
            self.start_date = start_datetime.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self):
        """
        Fetch stock data from Yahoo Finance
        
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            
            if data.empty or len(data) < 30:  # Require at least 30 days of data
                print(f"Error: Not enough data available for {self.ticker} in the specified date range.")
                return None
                
            # Keep only OHLCV columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Handle missing values
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            
            self.data = data
            return data
            
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {str(e)}")
            return None
    
    def preprocess_data(self, sequence_length=60, target_column='Close', feature_columns=None):
        """
        Preprocess data for LSTM model
        
        Args:
            sequence_length (int): Length of input sequences
            target_column (str): Column to predict
            feature_columns (list): List of columns to use as features
            
        Returns:
            dict: Dictionary containing processed data and metadata
        """
        if self.data is None:
            print("No data available. Call fetch_data() first.")
            return None
            
        data = self.data.copy()
        
        # Default to all columns if feature_columns not specified
        if feature_columns is None:
            feature_columns = data.columns.tolist()
        
        # Create additional features
        self._create_features(data)
        
        # Scale target column (typically 'Close')
        target_values = data[target_column].values.reshape(-1, 1)
        scaled_target = self.scaler.fit_transform(target_values)
        
        # Scale features
        feature_values = data[feature_columns].values
        scaled_features = self.feature_scaler.fit_transform(feature_values)
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target, sequence_length)
        
        # Split into train and validation sets (80% train, 20% validation)
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'sequence_length': sequence_length,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'train_size': train_size
        }
    
    def _create_features(self, data):
        """Create additional features for the model"""
        # Add moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # Add price momentum
        data['Price_Momentum'] = data['Close'].pct_change(periods=5)
        
        # Add volatility
        data['Volatility'] = data['Close'].rolling(window=10).std()
        
        # Normalize volume
        data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        # Fill NaN values created by rolling windows
        data.fillna(method='bfill', inplace=True)
        
        return data
    
    def _create_sequences(self, features, target, sequence_length):
        """
        Create input sequences and target values
        
        Args:
            features: Scaled feature values
            target: Scaled target values
            sequence_length: Length of input sequences
            
        Returns:
            tuple: (X, y) where X is input sequences and y is target values
        """
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(target[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, scaled_values):
        """
        Convert scaled values back to original scale
        
        Args:
            scaled_values: Scaled values to convert
            
        Returns:
            numpy.array: Original scale values
        """
        # Reshape to column vector if needed
        if len(scaled_values.shape) == 1:
            scaled_values = scaled_values.reshape(-1, 1)
            
        return self.scaler.inverse_transform(scaled_values)
    
    def get_current_price(self):
        """
        Get the most recent closing price
        
        Returns:
            float: Current price
        """
        if self.data is None:
            print("No data available. Call fetch_data() first.")
            return None
            
        return self.data['Close'].iloc[-1]
    
    def generate_future_dates(self, days):
        """
        Generate future dates for prediction
        
        Args:
            days (int): Number of days to predict
            
        Returns:
            list: List of future dates
        """
        last_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        future_dates = []
        
        for i in range(1, days + 1):
            future_date = last_date + timedelta(days=i)
            # Skip weekends
            while future_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                future_date += timedelta(days=1)
            future_dates.append(future_date.strftime('%Y-%m-%d'))
            
        return future_dates 