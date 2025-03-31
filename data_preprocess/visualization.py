import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
    
    now = datetime.now()
    five_years_ago = now - timedelta(days=5*365)
    
    df = df[df.index >= five_years_ago]
    
    return df

def compute_indicators(df):
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
    
    df['UpperBB'], df['MiddleBB'], df['LowerBB'] = talib.BBANDS(df['close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return df

def plot_chart(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    mpf.plot(df, type='candle', ax=axes[0], volume=axes[1],
             mav=(50, 200),
             addplot=[
                 mpf.make_addplot(df['UpperBB'], color='purple', linestyle='dotted', ax=axes[0]),
                 mpf.make_addplot(df['MiddleBB'], color='gray', linestyle='dotted', ax=axes[0]),
                 mpf.make_addplot(df['LowerBB'], color='purple', linestyle='dotted', ax=axes[0])
             ])
    
    axes[2].plot(df.index, df['MACD'], label='MACD', color='blue')
    axes[2].plot(df.index, df['MACD_Signal'], label='Signal', color='red')
    axes[2].bar(df.index, df['MACD_Hist'], label='Hist', color='gray')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'data_processed/yfinance/full/AAPL_1d_full.csv'
    df = load_data(file_path)
    df = compute_indicators(df)
    plot_chart(df)

if __name__ == '__main__':
    main()
