#!/usr/bin/env python3
"""
This version stores:
 1) ARIMA data as CSV in a dedicated 'arima' subfolder
 2) LSTM data as NPZ in a dedicated 'lstm' subfolder
 3) Uses a dictionary of LSTM window sizes per timeframe, e.g.:
    {
      "1m": [30, 60, 120],
      "5m": [36, 72],
      "15m": [32, 64],
      "30m": [24, 48],
      "60m": [24, 72],
      "1d": [20, 60]
    }

Folder structure example:

 yfinance
   └─ processed_data
        ├─ arima
        │    AAPL_1m_arima.csv
        │    ...
        └─ lstm
             AAPL_1m_ws30.npz
             AAPL_1m_ws60.npz
             AAPL_1m_ws120.npz
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YFinanceDataProcessor:
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to your raw yfinance data folder, e.g. 'D:/yfinance/output'
        """
        self.data_dir = data_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Dictionary of LSTM window sizes for each timeframe
        # You can edit these values to suit your preference.
        self.lstm_windows_map = {
            "1m": [30, 60, 120],
            "5m": [36, 72],
            "15m": [32, 64],
            "30m": [24, 48],
            "60m": [24, 72],
            "1d": [20, 60]
        }

    def get_available_symbols(self) -> List[str]:
        entries = os.listdir(self.data_dir)
        symbols = []
        for entry in entries:
            full_path = os.path.join(self.data_dir, entry)
            if os.path.isdir(full_path):
                symbols.append(entry)
        symbols.sort()
        return symbols

    def get_available_timeframes(self, symbol: str) -> List[str]:
        folder_path = os.path.join(self.data_dir, symbol)
        files = glob.glob(os.path.join(folder_path, f"{symbol}_*.json"))
        timeframes = set()
        for file_path in files:
            filename = os.path.basename(file_path)
            parts = filename.split("_")
            if len(parts) >= 3:
                tf = parts[1]
                timeframes.add(tf)
        return sorted(list(timeframes))

    def load_yfinance_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        folder_path = os.path.join(self.data_dir, symbol)
        pattern = f"{symbol}_{timeframe}_*.json"
        matching_files = glob.glob(os.path.join(folder_path, pattern))
        if not matching_files:
            raise FileNotFoundError(f"No JSON file found for {symbol} at timeframe {timeframe}.")

        file_path = sorted(matching_files)[-1]
        logger.info(f"Loading file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        price_data = data.get("price_data", {})
        timestamps = data.get("timestamps", [])

        df = pd.DataFrame({
            "datetime": timestamps,
            "open": price_data.get("open", []),
            "high": price_data.get("high", []),
            "low": price_data.get("low", []),
            "close": price_data.get("close", []),
            "volume": price_data.get("volume", [])
        })

        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        df['symbol'] = symbol
        df['timeframe'] = timeframe
        logger.info(f"Loaded {len(df)} records for {symbol} ({timeframe})")
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ind = df.copy()
        df_ind['ema_9'] = df_ind['close'].ewm(span=9, adjust=False).mean()
        df_ind['sma_14'] = df_ind['close'].rolling(window=14).mean()

        delta = df_ind['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_ind['rsi_14'] = 100 - (100 / (1 + rs))

        short_ema = df_ind['close'].ewm(span=12, adjust=False).mean()
        long_ema = df_ind['close'].ewm(span=26, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_ind['macd_line'] = macd_line
        df_ind['macd_signal'] = signal_line
        df_ind['macd_hist'] = macd_line - signal_line

        middle = df_ind['close'].rolling(window=20).mean()
        std = df_ind['close'].rolling(window=20).std()
        df_ind['bollinger_upper'] = middle + 2 * std
        df_ind['bollinger_middle'] = middle
        df_ind['bollinger_lower'] = middle - 2 * std

        high_low = df_ind['high'] - df_ind['low']
        high_close = (df_ind['high'] - df_ind['close'].shift()).abs()
        low_close = (df_ind['low'] - df_ind['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_ind['atr_14'] = tr.rolling(window=14).mean()

        df_ind['pct_change'] = df_ind['close'].pct_change() * 100
        df_ind['log_return'] = np.log(df_ind['close'] / df_ind['close'].shift(1))

        df_ind.dropna(inplace=True)
        return df_ind

    def prepare_arima_data(self, df: pd.DataFrame) -> pd.DataFrame:
        arima_df = pd.DataFrame({
            'date': df.index,
            'close': df['close'],
            'symbol': df['symbol'].iloc[0] if len(df) else None,
            'timeframe': df['timeframe'].iloc[0] if len(df) else None
        })
        arima_df.set_index('date', inplace=True)
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
        Creates X, y arrays for LSTM. By default, window_size=60 if none is provided,
        but we will override it for each timeframe from the dictionary.
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in features:
                features.remove(target_col)

        selected_cols = features + [target_col]
        sub_df = df[selected_cols].copy()

        data_values = sub_df.values
        scaled_data = self.scaler.fit_transform(data_values)

        X, y = [], []
        target_idx = sub_df.columns.get_loc(target_col)

        for i in range(window_size, len(scaled_data) - forecast_horizon + 1):
            X.append(scaled_data[i - window_size:i, :])
            y.append(scaled_data[i:i + forecast_horizon, target_idx])

        X = np.array(X)
        y = np.array(y)

        return X, y, self.scaler

    def save_arima_data(
        self, df: pd.DataFrame, output_dir: str, symbol: str, timeframe: str
    ) -> str:
        """
        Save ARIMA data (CSV) in an 'arima' subfolder. Contains date, close, etc.
        """
        arima_dir = os.path.join(output_dir, "arima")
        os.makedirs(arima_dir, exist_ok=True)
        filename = f"{symbol}_{timeframe}_arima.csv"
        filepath = os.path.join(arima_dir, filename)
        df.to_csv(filepath)
        return filepath

    def save_lstm_data(
        self, X: np.ndarray, y: np.ndarray, output_dir: str, symbol: str, timeframe: str, window_size: int
    ) -> str:
        """
        Save LSTM arrays as NPZ in a 'lstm' subfolder. Filenames contain window_size.
        """
        lstm_dir = os.path.join(output_dir, "lstm")
        os.makedirs(lstm_dir, exist_ok=True)
        filename = f"{symbol}_{timeframe}_ws{window_size}.npz"
        filepath = os.path.join(lstm_dir, filename)
        # Save using compressed format
        np.savez_compressed(filepath, X=X, y=y)
        return filepath

    def save_full_csv(
        self, df: pd.DataFrame,
        output_dir: str,
        symbol: str,
        timeframe: str
    ) -> str:
        """
        If you want to store the full indicator CSV, we can keep it in e.g. "full" subfolder.
        """
        full_dir = os.path.join(output_dir, "full")
        os.makedirs(full_dir, exist_ok=True)
        filename = f"{symbol}_{timeframe}_full.csv"
        filepath = os.path.join(full_dir, filename)
        df.to_csv(filepath)
        return filepath

    def process_all_data(
        self,
        output_dir: str,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        save_full: bool = True,
        save_arima: bool = True,
        save_lstm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Processes all data. For each symbol/timeframe:
         - calculates indicators
         - saves optional full CSV
         - prepares ARIMA CSV
         - prepares LSTM NPZ files using window sizes from self.lstm_windows_map

        Args:
            output_dir: Directory to store output subfolders (arima, lstm, full)
            symbols: Which symbols to process (None => all discovered)
            timeframes: Which timeframes to process (None => all discovered for that symbol)
            save_full: Whether to save a CSV with all indicators in "full" subfolder
            save_arima: Whether to save ARIMA data in "arima" subfolder
            save_lstm: Whether to save LSTM .npz data in "lstm" subfolder

        Returns:
            A list of dictionaries summarizing the processed files.
        """
        results = []

        if not symbols:
            symbols = self.get_available_symbols()

        for symbol in symbols:
            if not timeframes:
                symbol_timeframes = self.get_available_timeframes(symbol)
            else:
                symbol_timeframes = timeframes

            for tf in symbol_timeframes:
                result_entry = {
                    'symbol': symbol,
                    'timeframe': tf,
                    'files': {}
                }
                try:
                    df = self.load_yfinance_data(symbol, tf)
                    df_ind = self.calculate_technical_indicators(df)

                    # Optional: save CSV with all indicators
                    if save_full:
                        path_full = self.save_full_csv(df_ind, output_dir, symbol, tf)
                        result_entry['files']['full'] = path_full

                    # Optional: save ARIMA CSV
                    if save_arima:
                        arima_df = self.prepare_arima_data(df_ind)
                        path_arima = self.save_arima_data(arima_df, output_dir, symbol, tf)
                        result_entry['files']['arima'] = path_arima

                    # Optional: create NPZ for LSTM with one or more window sizes
                    if save_lstm:
                        # look up the window sizes for this timeframe
                        # default to [60] if not found
                        ws_list = self.lstm_windows_map.get(tf, [60])
                        lstm_dict = {}
                        for ws in ws_list:
                            X, y, _ = self.prepare_lstm_data(df_ind, window_size=ws)
                            npz_path = self.save_lstm_data(X, y, output_dir, symbol, tf, ws)
                            lstm_dict[ws] = {
                                'X_shape': X.shape,
                                'y_shape': y.shape,
                                'path': npz_path
                            }
                        result_entry['files']['lstm'] = lstm_dict

                except Exception as e:
                    result_entry['error'] = str(e)
                results.append(result_entry)

        return results

if __name__ == "__main__":
    data_dir = "data_raw/yfinance/output"
    output_dir = "data_processed/yfinance"

    processor = YFinanceDataProcessor(data_dir)

    processing_results = processor.process_all_data(
        output_dir=output_dir,
        symbols=None,
        timeframes=None,
        save_full=True,     # saves CSV with all indicators to processed_data/full
        save_arima=True,    # saves ARIMA CSV to processed_data/arima
        save_lstm=True      # saves NPZ to processed_data/lstm
    )

    logger.info("\nProcessing Summary:")
    for item in processing_results:
        symbol = item['symbol']
        timeframe = item['timeframe']
        if 'error' in item:
            logger.error(f"❌ {symbol} ({timeframe}): Error - {item['error']}")
        else:
            logger.info(f"✅ {symbol} ({timeframe}): Success")
            for ftype, fval in item['files'].items():
                if isinstance(fval, dict):
                    for ws, subf in fval.items():
                        logger.info(f"   LSTM ws={ws}, X={subf['X_shape']}, y={subf['y_shape']}, path={subf['path']}")
                else:
                    logger.info(f"   {ftype}: {fval}")
