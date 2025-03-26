import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from .stock_data_loader import StockDataLoader

class EnhancedDataLoader(StockDataLoader):
    """
    Enhanced data loader that combines multiple data sources and adds more sophisticated features
    """
    def __init__(self, ticker, start_date=None, end_date=None, sequence_length=20):
        """
        Initialize the enhanced data loader
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (default: 5 years ago)
            end_date: End date for data (default: today)
            sequence_length: Length of sequence for prediction
        """
        super().__init__(ticker, start_date, end_date, sequence_length)
        self.additional_data = {}
        
    def fetch_additional_market_data(self):
        """
        Fetch additional market data such as market indices
        
        Returns:
            DataFrames with additional market data
        """
        # Get market index data (S&P 500 for US stocks)
        spy_data = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
        vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        
        # Add to additional data dictionary
        self.additional_data['market_index'] = spy_data
        self.additional_data['volatility_index'] = vix_data
        
        print(f"Downloaded additional market data: S&P 500 and VIX")
        return self.additional_data
    
    def fetch_sector_data(self):
        """
        Fetch sector ETF data relevant to the stock
        
        Returns:
            DataFrame with sector data
        """
        # Common sector ETFs
        sectors = {
            'XLF': 'Financial',
            'XLK': 'Technology',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        # Fetch data for all sectors
        sector_data = {}
        for symbol, name in sectors.items():
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if not data.empty:
                    sector_data[symbol] = data
                    print(f"Downloaded {name} sector data: {len(data)} days")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        self.additional_data['sectors'] = sector_data
        return sector_data
    
    def prepare_enhanced_features(self):
        """
        Prepare enhanced features including market data and advanced technical indicators
        
        Returns:
            DataFrame with enhanced features
        """
        # First get the base features
        if self.processed_data is None:
            self.prepare_features()
            
        # Fetch additional data
        self.fetch_additional_market_data()
        
        data = self.processed_data.copy()
        
        # Add market relative features if available
        if 'market_index' in self.additional_data:
            market_data = self.additional_data['market_index']
            
            # Ensure we're working with single-index DataFrames
            if isinstance(market_data.columns, pd.MultiIndex):
                market_data.columns = [col[0] if isinstance(col, tuple) else col for col in market_data.columns]
                
            # Align dates (make sure index is DatetimeIndex)
            common_dates = data.index.intersection(market_data.index)
            if len(common_dates) == 0:
                print("Warning: No overlapping dates between stock data and market data")
                return data
                
            data = data.loc[common_dates]
            market_data = market_data.loc[common_dates]
            
            # Calculate relative strength (stock performance vs market)
            data['Market_Close'] = market_data['Close'].values
            data['Relative_Strength'] = (data['Close'] / data['Market_Close']).values
            data['Relative_Strength_Change'] = data['Relative_Strength'].pct_change().values
            
            # Market momentum
            data['Market_Returns'] = market_data['Close'].pct_change().values
            data['Market_MA20'] = market_data['Close'].rolling(window=20).mean().values
            data['Market_Trend'] = (market_data['Close'] > market_data['Close'].rolling(window=20).mean()).astype(int).values
        
        # Add volatility index features if available
        if 'volatility_index' in self.additional_data:
            vix_data = self.additional_data['volatility_index']
            
            # Ensure we're working with single-index DataFrames
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = [col[0] if isinstance(col, tuple) else col for col in vix_data.columns]
            
            # Align dates
            common_dates = data.index.intersection(vix_data.index)
            if len(common_dates) > 0:
                data = data.loc[common_dates]
                vix_data = vix_data.loc[common_dates]
                
                # Add VIX features
                data['VIX'] = vix_data['Close'].values
                data['VIX_MA10'] = vix_data['Close'].rolling(window=10).mean().values
                data['VIX_Change'] = vix_data['Close'].pct_change().values
            else:
                print("Warning: No overlapping dates between stock data and VIX data")
            
        # Enhanced technical indicators
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Average True Range (ATR) - Volatility indicator
        data['High_Low'] = data['High'] - data['Low']
        data['High_Close'] = abs(data['High'] - data['Close'].shift())
        data['Low_Close'] = abs(data['Low'] - data['Close'].shift())
        data['True_Range'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        data['ATR'] = data['True_Range'].rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        obv = 0
        obv_values = []
        for i, row in data.iterrows():
            if len(obv_values) == 0:
                obv_values.append(0)
                continue
            
            prev_idx = data.index[data.index.get_loc(i) - 1]
            if row['Close'] > data.loc[prev_idx, 'Close']:
                obv += row['Volume']
            elif row['Close'] < data.loc[prev_idx, 'Close']:
                obv -= row['Volume']
            
            obv_values.append(obv)
            
        data['OBV'] = obv_values
        data['OBV_MA20'] = data['OBV'].rolling(window=20).mean()
        
        # Awesome Oscillator
        data['AO'] = (data['High'] + data['Low']) / 2
        data['AO_5'] = data['AO'].rolling(window=5).mean()
        data['AO_34'] = data['AO'].rolling(window=34).mean()
        data['Awesome_Oscillator'] = data['AO_5'] - data['AO_34']
        
        # Elder's Force Index
        data['Force_Index'] = data['Close'].diff() * data['Volume']
        data['Force_Index_13'] = data['Force_Index'].ewm(span=13).mean()
        
        # Drop NaN values
        data = data.dropna()
        
        # Update feature columns
        # Keep the original columns and add the new ones
        additional_columns = [
            'Relative_Strength', 'Relative_Strength_Change', 'Market_Returns', 
            'Market_Trend', 'VIX', 'VIX_Change', 'BB_Width', 'BB_Position',
            'ATR', 'OBV', 'Awesome_Oscillator', 'Force_Index_13'
        ]
        
        # Add only columns that exist in the dataframe
        self.feature_columns = list(self.feature_columns)
        for col in additional_columns:
            if col in data.columns:
                self.feature_columns.append(col)
        
        self.processed_data = data
        
        return self.processed_data
    
    def scale_enhanced_data(self):
        """
        Scale the enhanced data using Min-Max scaling
        
        Returns:
            Scaled DataFrame
        """
        if self.processed_data is None or 'Relative_Strength' not in self.processed_data.columns:
            self.prepare_enhanced_features()
            
        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.processed_data[self.feature_columns].values)
        
        # Create a new DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=self.feature_columns, index=self.processed_data.index)
        
        self.processed_data[self.feature_columns] = scaled_df[self.feature_columns]
        
        return self.processed_data
    
    def prepare_data_for_continued_training(self, existing_scaler=None, test_size=0.2):
        """
        Prepare data for continued training, optionally using an existing scaler
        
        Args:
            existing_scaler: Optional scaler from previous training
            test_size: Proportion of data to use for testing
            
        Returns:
            train_X, train_y, test_X, test_y: Training and testing data
        """
        if self.processed_data is None or 'Relative_Strength' not in self.processed_data.columns:
            self.scale_enhanced_data()
            
        # If there's an existing scaler, use it instead
        if existing_scaler is not None:
            print("Using existing scaler for data transformation...")
            self.scaler = existing_scaler
            
            # Re-scale the data using the existing scaler
            scaled_data = self.scaler.transform(self.processed_data[self.feature_columns].values)
            scaled_df = pd.DataFrame(scaled_data, columns=self.feature_columns, index=self.processed_data.index)
            self.processed_data[self.feature_columns] = scaled_df[self.feature_columns]
        
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