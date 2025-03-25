import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class StockDataLoader:
    """
    Class to download, process and prepare stock data for LSTM prediction
    """
    def __init__(self, ticker_symbol, start_date=None, end_date=None, sequence_length=20):
        """
        Initialize the stock data loader
        
        Args:
            ticker_symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for historical data (default: 5 years ago)
            end_date: End date for historical data (default: today)
            sequence_length: Number of days to use for sequence prediction
        """
        self.ticker = ticker_symbol
        self.sequence_length = sequence_length
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            # Default to 5 years of data
            start = datetime.now() - timedelta(days=5*365)
            self.start_date = start.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.data = None
        self.scaled_data = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
    
    def download_data(self):
        """
        Download stock data from Yahoo Finance
        """
        print(f"Downloading stock data for {self.ticker} from {self.start_date} to {self.end_date}")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Check if we have enough data
        if data.empty or len(data) < 100:
            raise ValueError(f"Not enough data available for {self.ticker}. Please check the ticker symbol and date range.")
        
        # Reset index if it's a multi-index DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            # Convert multi-level columns to single level
            data.columns = [col[0] for col in data.columns]
        
        self.data = data
        return data
    
    def prepare_features(self):
        """
        Calculate technical indicators and prepare features for the model
        """
        if self.data is None:
            self.download_data()
            
        df = self.data.copy()
        
        try:
            # Technical indicators
            # 1. Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # 2. Price momentum
            df['Returns'] = df['Close'].pct_change()
            df['Returns_5d'] = df['Close'].pct_change(periods=5)
            
            # 3. Volatility
            df['Volatility_10d'] = df['Returns'].rolling(window=10).std()
            
            # 4. Trading range
            df['Range'] = df['High'] - df['Low']
            # Calculate Range_Pct directly
            df['Range_Pct'] = df['Range'] / df['Open']
            
            # 5. Volume indicators
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            
            # 6. Price relative to moving averages
            df['Price_MA5_Ratio'] = df['Close'] / df['MA5']
            df['Price_MA20_Ratio'] = df['Close'] / df['MA20']
            
            # 7. Moving average convergence/divergence (simple implementation)
            df['MACD'] = df['MA10'] - df['MA20']
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Select features for the model
            self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                                  'MA5', 'MA20', 'Returns', 'Volatility_10d', 
                                  'Range', 'Volume_Change', 'MACD', 
                                  'Price_MA5_Ratio', 'Price_MA20_Ratio']
            
            self.processed_data = df
            self.feature_data = df[self.feature_columns].values
            
            return df
            
        except Exception as e:
            print(f"Error in prepare_features: {e}")
            # Print data types for debugging
            print(f"Data types: {df.dtypes}")
            print(f"DataFrame columns: {df.columns}")
            print(f"Is multi-index: {isinstance(df.columns, pd.MultiIndex)}")
            if isinstance(df.columns, pd.MultiIndex):
                print(f"Multi-index levels: {df.columns.levels}")
            raise
    
    def scale_data(self):
        """
        Scale the data using MinMaxScaler
        """
        if self.feature_data is None:
            self.prepare_features()
            
        self.scaled_data = self.scaler.fit_transform(self.feature_data)
        return self.scaled_data
    
    def create_sequences(self):
        """
        Create sequences for LSTM model
        
        Returns:
            X: sequences of shape (n_sequences, sequence_length, n_features)
            y: target values (next day closing prices) of shape (n_sequences, 1)
        """
        if self.scaled_data is None:
            self.scale_data()
            
        # Find the index of the Close price in our feature set
        close_idx = self.feature_columns.index('Close')
        
        X, y = [], []
        for i in range(len(self.scaled_data) - self.sequence_length):
            # Sequence of historical data
            X.append(self.scaled_data[i:(i + self.sequence_length)])
            # Target is the next day's closing price
            y.append(self.scaled_data[i + self.sequence_length, close_idx])
            
        return np.array(X), np.array(y).reshape(-1, 1)
    
    def prepare_data_for_training(self, train_split=0.8):
        """
        Prepare data for training and testing
        
        Args:
            train_split: ratio of training data
        
        Returns:
            train_X, train_y, test_X, test_y, scaler
        """
        X, y = self.create_sequences()
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        train_X, train_y = X[:train_size], y[:train_size]
        test_X, test_y = X[train_size:], y[train_size:]
        
        print(f"Training data shape: {train_X.shape}")
        print(f"Test data shape: {test_X.shape}")
        
        return train_X, train_y, test_X, test_y, self.scaler
    
    def inverse_transform_predictions(self, predictions):
        """
        Convert scaled predictions back to original scale
        
        Args:
            predictions: scaled predictions from the model
            
        Returns:
            original_predictions: predictions in original scale
        """
        # Find the index of the Close price
        close_idx = self.feature_columns.index('Close')
        
        # Create a dummy array with zeros
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        # Put the predictions in the close price column
        dummy[:, close_idx] = predictions.flatten()
        
        # Inverse transform the dummy array
        original_scale_dummy = self.scaler.inverse_transform(dummy)
        # Extract the close price predictions
        original_predictions = original_scale_dummy[:, close_idx]
        
        return original_predictions
    
    def plot_stock_data(self, save_path=None):
        """
        Plot the stock price history with volume
        """
        if self.data is None:
            self.download_data()
            
        # Need to calculate MA20 and MA50 if not already present in self.data
        if 'MA20' not in self.data.columns:
            self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        if 'MA50' not in self.data.columns:
            self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot stock price
        ax1.plot(self.data.index, self.data['Close'], label='Close Price')
        ax1.plot(self.data.index, self.data['MA20'], label='20-day MA', alpha=0.7)
        ax1.plot(self.data.index, self.data['MA50'], label='50-day MA', alpha=0.7)
        ax1.set_title(f'{self.ticker} Stock Price History')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot volume
        ax2.bar(self.data.index, self.data['Volume'], color='blue', alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Stock price chart saved to {save_path}")
        
        plt.close()
        
    def get_recent_data(self, days=60):
        """
        Get the most recent data for display
        """
        if self.processed_data is None:
            self.prepare_features()
            
        return self.processed_data.tail(days) 