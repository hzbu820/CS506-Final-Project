"""
Stock Market Data Preprocessing Pipeline

This module processes Alpha Vantage JSON data files into formats suitable for
time series forecasting models like ARIMA and LSTM.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union
import glob
from datetime import datetime


class StockDataProcessor:
    """
    A class to process stock market data from Alpha Vantage JSON files
    into formats suitable for time series forecasting models.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the processor with the data directory.
        
        Args:
            data_dir: Directory containing Alpha Vantage JSON files
        """
        self.data_dir = data_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def get_available_symbols(self) -> List[str]:
        """Get a list of all available stock symbols from the data directory."""
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        symbols = set()
        for file in files:
            filename = os.path.basename(file)
            symbol = filename.split('_')[0]
            symbols.add(symbol)
        return sorted(list(symbols))
    
    def get_available_timeframes(self) -> List[str]:
        """Get a list of all available timeframes from the data directory."""
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        timeframes = set()
        for file in files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) >= 2:
                timeframe = parts[1]
                timeframes.add(timeframe)
        return sorted(list(timeframes))
    
    def load_alpha_vantage_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load data from Alpha Vantage JSON file.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Time interval (e.g., '1min', '5min', 'daily')
            
        Returns:
            DataFrame with processed time series data
        """
        # Find the matching file
        pattern = f"{symbol}_{timeframe}_alphavantage_*.json"
        matching_files = glob.glob(os.path.join(self.data_dir, pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No data file found for {symbol} with timeframe {timeframe}")
        
        # Use the most recent file if there are multiple matches
        file_path = sorted(matching_files)[-1]
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Determine the data structure based on timeframe
        if timeframe == 'daily':
            time_series_key = 'Time Series (Daily)'
        else:
            time_series_key = f'Time Series ({timeframe})'
        
        if time_series_key not in data:
            raise KeyError(f"Expected key '{time_series_key}' not found in JSON data")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        
        # Rename columns to standard names
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Set index to datetime and sort by date
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Add symbol and timeframe columns
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators similar to those in the original code.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_with_indicators = df.copy()
        
        # EMA 9
        df_with_indicators['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        
        # SMA 14
        df_with_indicators['sma_14'] = df['close'].rolling(window=14).mean()
        
        # RSI 14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_with_indicators['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        short_ema = df['close'].ewm(span=12, adjust=False).mean()
        long_ema = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_with_indicators['macd_line'] = macd_line
        df_with_indicators['macd_signal'] = signal_line
        df_with_indicators['macd_hist'] = macd_line - signal_line
        
        # Bollinger Bands
        middle = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df_with_indicators['bollinger_upper'] = middle + 2 * std
        df_with_indicators['bollinger_middle'] = middle
        df_with_indicators['bollinger_lower'] = middle - 2 * std
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_with_indicators['atr_14'] = tr.rolling(window=14).mean()
        
        # Add percentage change
        df_with_indicators['pct_change'] = df['close'].pct_change() * 100
        
        # Add log returns
        df_with_indicators['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Drop NaN values created by calculations
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators
    
    def prepare_arima_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for ARIMA modeling.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with closing prices suitable for ARIMA
        """
        # For ARIMA, we primarily need the closing price time series
        arima_df = pd.DataFrame({
            'date': df.index,
            'close': df['close'],
            'symbol': df['symbol'].iloc[0],
            'timeframe': df['timeframe'].iloc[0]
        })
        
        # Set date as index
        arima_df = arima_df.set_index('date')
        
        return arima_df
    
    def prepare_lstm_data(
        self, 
        df: pd.DataFrame, 
        window_size: int = 60,
        features: List[str] = None,
        target_col: str = 'close',
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for LSTM modeling with a sliding window approach.
        
        Args:
            df: DataFrame with stock data and technical indicators
            window_size: Number of past time steps to use as input features
            features: List of column names to use as features (if None, uses all numeric columns)
            target_col: Column to predict
            forecast_horizon: Number of steps to forecast into the future
            
        Returns:
            Tuple of (X_scaled, y_scaled, scaler)
            - X_scaled: 3D numpy array with shape (n_samples, window_size, n_features)
            - y_scaled: 2D numpy array with shape (n_samples, forecast_horizon)
            - scaler: Fitted scaler for inverse transformation
        """
        # Select features if not provided
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove the target from features to avoid data leakage
            if target_col in features:
                features.remove(target_col)
        
        # Create DataFrame with selected features and target
        selected_data = df[features + [target_col]].copy()
        
        # Scale the data
        data_values = selected_data.values
        scaled_data = self.scaler.fit_transform(data_values)
        
        # Create sliding windows
        X = []
        y = []
        
        for i in range(window_size, len(scaled_data) - forecast_horizon + 1):
            # Input window
            X.append(scaled_data[i - window_size:i, :])
            
            # Target (future values of target column)
            target_index = selected_data.columns.get_loc(target_col)
            y.append(scaled_data[i:i + forecast_horizon, target_index])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, self.scaler
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_dir: str, 
        symbol: str, 
        timeframe: str,
        data_type: str = 'full'
    ) -> str:
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            output_dir: Directory to save to
            symbol: Stock symbol
            timeframe: Time interval
            data_type: Type of processed data (e.g., 'full', 'arima', 'lstm')
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{symbol}_{timeframe}_{data_type}_{timestamp}.csv"
        file_path = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(file_path)
        
        return file_path
    
    def save_lstm_arrays(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        output_dir: str, 
        symbol: str, 
        timeframe: str,
        window_size: int
    ) -> Tuple[str, str]:
        """
        Save LSTM input and target arrays to NPZ files.
        
        Args:
            X: Input features array
            y: Target values array
            output_dir: Directory to save to
            symbol: Stock symbol
            timeframe: Time interval
            window_size: Window size used for LSTM data
            
        Returns:
            Tuple of paths to saved X and y files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filenames
        timestamp = datetime.now().strftime('%Y%m%d')
        x_filename = f"{symbol}_{timeframe}_lstm_X_w{window_size}_{timestamp}.npz"
        y_filename = f"{symbol}_{timeframe}_lstm_y_w{window_size}_{timestamp}.npz"
        
        x_path = os.path.join(output_dir, x_filename)
        y_path = os.path.join(output_dir, y_filename)
        
        # Save arrays
        np.savez_compressed(x_path, X=X)
        np.savez_compressed(y_path, y=y)
        
        return x_path, y_path
    
    def process_symbol(
        self, 
        symbol: str, 
        timeframe: str, 
        output_dir: str,
        save_full: bool = True,
        save_arima: bool = True,
        save_lstm: bool = True,
        lstm_window_sizes: List[int] = None
    ) -> Dict:
        """
        Process data for a specific symbol and timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval
            output_dir: Directory to save processed data
            save_full: Whether to save the full processed DataFrame
            save_arima: Whether to save ARIMA-ready data
            save_lstm: Whether to save LSTM-ready data
            lstm_window_sizes: List of window sizes for LSTM data preparation
            
        Returns:
            Dictionary with paths to saved files
        """
        if lstm_window_sizes is None:
            lstm_window_sizes = [60]  # Default window size
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'files': {}
        }
        
        try:
            # Load data
            print(f"Loading data for {symbol} ({timeframe})...")
            df = self.load_alpha_vantage_data(symbol, timeframe)
            
            # Calculate indicators
            print(f"Calculating technical indicators for {symbol} ({timeframe})...")
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Save full processed data
            if save_full:
                print(f"Saving full processed data for {symbol} ({timeframe})...")
                full_path = self.save_processed_data(
                    df_with_indicators, 
                    output_dir, 
                    symbol, 
                    timeframe, 
                    'full'
                )
                result['files']['full'] = full_path
            
            # Prepare and save ARIMA data
            if save_arima:
                print(f"Preparing ARIMA data for {symbol} ({timeframe})...")
                arima_df = self.prepare_arima_data(df)
                arima_path = self.save_processed_data(
                    arima_df, 
                    output_dir, 
                    symbol, 
                    timeframe, 
                    'arima'
                )
                result['files']['arima'] = arima_path
            
            # Prepare and save LSTM data
            if save_lstm:
                result['files']['lstm'] = {}
                
                for window_size in lstm_window_sizes:
                    print(f"Preparing LSTM data with window size {window_size} for {symbol} ({timeframe})...")
                    X, y, _ = self.prepare_lstm_data(
                        df_with_indicators, 
                        window_size=window_size
                    )
                    
                    x_path, y_path = self.save_lstm_arrays(
                        X, 
                        y, 
                        output_dir, 
                        symbol, 
                        timeframe,
                        window_size
                    )
                    
                    result['files']['lstm'][f'window_{window_size}'] = {
                        'X': x_path,
                        'y': y_path,
                        'shape_X': X.shape,
                        'shape_y': y.shape
                    }
            
            print(f"Processing complete for {symbol} ({timeframe}).")
            return result
            
        except Exception as e:
            print(f"Error processing {symbol} ({timeframe}): {str(e)}")
            result['error'] = str(e)
            return result
    
    def process_all_data(
        self, 
        output_dir: str,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        save_full: bool = True,
        save_arima: bool = True,
        save_lstm: bool = True,
        lstm_window_sizes: List[int] = None
    ) -> List[Dict]:
        """
        Process data for all specified symbols and timeframes.
        
        Args:
            output_dir: Directory to save processed data
            symbols: List of symbols to process (if None, processes all available)
            timeframes: List of timeframes to process (if None, processes all available)
            save_full: Whether to save the full processed DataFrame
            save_arima: Whether to save ARIMA-ready data
            save_lstm: Whether to save LSTM-ready data
            lstm_window_sizes: List of window sizes for LSTM data preparation
            
        Returns:
            List of dictionaries with results for each symbol-timeframe combination
        """
        # Use all available if not specified
        if symbols is None:
            symbols = self.get_available_symbols()
        
        if timeframes is None:
            timeframes = self.get_available_timeframes()
        
        results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                result = self.process_symbol(
                    symbol,
                    timeframe,
                    output_dir,
                    save_full,
                    save_arima,
                    save_lstm,
                    lstm_window_sizes
                )
                results.append(result)
        
        return results


# Usage example
if __name__ == "__main__":
    # Directory with Alpha Vantage JSON files
    data_dir = "data_alpha_vantage"
    
    # Directory to save processed data
    output_dir = "processed_data"
    
    # Create processor
    processor = StockDataProcessor(data_dir)
    
    # Process all data
    results = processor.process_all_data(
        output_dir=output_dir,
        symbols=None,  # Process all available symbols
        timeframes=None,  # Process all available timeframes
        save_full=True,
        save_arima=True,
        save_lstm=True,
        lstm_window_sizes=[30, 60, 90]  # Different window sizes
    )
    
    # Print summary
    print("\nProcessing Summary:")
    
    for result in results:
        symbol = result['symbol']
        timeframe = result['timeframe']
        
        if 'error' in result:
            print(f"❌ {symbol} ({timeframe}): Error - {result['error']}")
        else:
            print(f"✅ {symbol} ({timeframe}): Success")
            for file_type, file_path in result['files'].items():
                if isinstance(file_path, dict):
                    for window_size, paths in file_path.items():
                        print(f"   - LSTM {window_size}: X shape {paths['shape_X']}, y shape {paths['shape_y']}")
                else:
                    print(f"   - {file_type}: {file_path}")

# Example of how to load and use this data for model training:
"""
def load_lstm_training_data(x_path, y_path):
    X_data = np.load(x_path)['X']
    y_data = np.load(y_path)['y']
    return X_data, y_data

def load_arima_training_data(csv_path):
    df = pd.read_csv(csv_path, index_col='date', parse_dates=['date'])
    return df

# For LSTM training:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, output_units=1):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_units))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Example usage:
X, y = load_lstm_training_data('processed_data/AAPL_daily_lstm_X_w60_20250304.npz', 
                              'processed_data/AAPL_daily_lstm_y_w60_20250304.npz')

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create and train model
model = create_lstm_model((X.shape[1], X.shape[2]))
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
"""
