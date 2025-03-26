import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize intraday stock data and patterns')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--interval', type=str, default='5m',
                        help='Data interval (1m, 5m, 15m, 30m, 1h) (default: 5m)')
    parser.add_argument('--period', type=str, default='5d',
                        help='Period to download (1d, 5d, 1mo) (default: 5d)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Custom name for the output files (default: auto-generated)')
    
    return parser.parse_args()

def download_intraday_data(ticker, period='5d', interval='5m'):
    """Download intraday data from Yahoo Finance"""
    print(f"Downloading {interval} interval data for {ticker} over {period}...")
    
    try:
        # Download data with retries
        max_retries = 3
        data = None
        
        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, period=period, interval=interval)
                if not data.empty:
                    break
                print(f"Attempt {attempt+1}/{max_retries}: No data returned, retrying...")
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries}: Error downloading data: {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
        
        if data is None or data.empty:
            raise ValueError(f"Failed to download data for {ticker} after {max_retries} attempts")
    
        print(f"Downloaded {len(data)} data points.")
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            data.columns = [col[0] for col in data.columns]
            
        # Remove rows with NaN values
        data = data.dropna()
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        return data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def calculate_technical_indicators(data):
    """Calculate technical indicators for visualization"""
    df = data.copy()
    
    # Simple moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # Bollinger Bands (20-period, 2 standard deviations)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # Price change
    df['Price_Change'] = df['Close'].diff()
    df['Percent_Change'] = df['Close'].pct_change() * 100
    
    # Drop NaN values after calculations
    df = df.dropna()
    
    return df

def visualize_intraday_data(data, ticker, interval, output_name=None):
    """Visualize intraday data with technical indicators"""
    # Create output directory
    os.makedirs('outputs/figures', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_name:
        base_filename = f"{ticker}_intraday_{interval}_{output_name}"
    else:
        base_filename = f"{ticker}_intraday_{interval}_{timestamp}"
    
    # 1. Price chart with moving averages and Bollinger Bands
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(data['Datetime'], data['Close'], color='blue', linewidth=1.5, label='Price')
    
    # Plot moving averages
    ax1.plot(data['Datetime'], data['MA5'], color='red', linewidth=1, label='5-period MA')
    ax1.plot(data['Datetime'], data['MA20'], color='green', linewidth=1, label='20-period MA')
    ax1.plot(data['Datetime'], data['MA60'], color='purple', linewidth=1, label='60-period MA')
    
    # Plot Bollinger Bands
    ax1.plot(data['Datetime'], data['BB_upper'], color='gray', linestyle='--', linewidth=0.8, label='Upper BB')
    ax1.plot(data['Datetime'], data['BB_lower'], color='gray', linestyle='--', linewidth=0.8, label='Lower BB')
    ax1.fill_between(data['Datetime'], data['BB_upper'], data['BB_lower'], color='gray', alpha=0.1)
    
    # Format chart
    ax1.set_title(f'{ticker} Intraday Price ({interval} intervals)', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Volume subplot
    ax2.bar(data['Datetime'], data['Volume'], color='blue', alpha=0.5, label='Volume')
    ax2.plot(data['Datetime'], data['Volume_MA5'], color='red', linewidth=1, label='5-period Vol MA')
    ax2.plot(data['Datetime'], data['Volume_MA20'], color='green', linewidth=1, label='20-period Vol MA')
    
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis with dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{base_filename}_price.png")
    plt.close()
    
    # 2. MACD and RSI chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # MACD
    ax1.plot(data['Datetime'], data['MACD'], color='blue', linewidth=1.5, label='MACD')
    ax1.plot(data['Datetime'], data['Signal'], color='red', linewidth=1, label='Signal')
    ax1.bar(data['Datetime'], data['MACD_Histogram'], color='green', alpha=0.5, label='Histogram')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax1.set_title(f'{ticker} MACD ({interval} intervals)', fontsize=16)
    ax1.set_ylabel('MACD', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # RSI
    ax2.plot(data['Datetime'], data['RSI'], color='purple', linewidth=1.5)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=0.8)
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.8)
    ax2.fill_between(data['Datetime'], 70, 30, color='gray', alpha=0.1)
    
    ax2.set_title(f'{ticker} RSI (14-period)', fontsize=16)
    ax2.set_ylabel('RSI', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Format x-axis with dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{base_filename}_indicators.png")
    plt.close()
    
    # 3. Intraday patterns: Display hourly patterns
    # Group by hour and calculate average metrics
    data['Hour'] = data['Datetime'].dt.hour
    hourly_patterns = data.groupby('Hour').agg({
        'Close': 'mean',
        'Volume': 'mean',
        'Percent_Change': 'mean',
        'MACD': 'mean',
        'RSI': 'mean'
    }).reset_index()
    
    # Plot hourly patterns
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Average price by hour
    axes[0].bar(hourly_patterns['Hour'], hourly_patterns['Close'], color='blue', alpha=0.7)
    axes[0].set_title(f'{ticker} Average Price by Hour', fontsize=16)
    axes[0].set_ylabel('Avg Price ($)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Average volume by hour
    axes[1].bar(hourly_patterns['Hour'], hourly_patterns['Volume'], color='green', alpha=0.7)
    axes[1].set_title(f'{ticker} Average Volume by Hour', fontsize=16)
    axes[1].set_ylabel('Avg Volume', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Average percent change by hour
    axes[2].bar(hourly_patterns['Hour'], hourly_patterns['Percent_Change'], color='red' if hourly_patterns['Percent_Change'].mean() < 0 else 'green', alpha=0.7)
    axes[2].set_title(f'{ticker} Average Percent Change by Hour', fontsize=16)
    axes[2].set_xlabel('Hour of Day', fontsize=14)
    axes[2].set_ylabel('Avg % Change', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{base_filename}_patterns.png")
    plt.close()
    
    # Save processed data to CSV
    os.makedirs('outputs/data', exist_ok=True)
    data.to_csv(f"outputs/data/{base_filename}.csv", index=False)
    
    print(f"Visualizations saved to outputs/figures/{base_filename}_*.png")
    print(f"Processed data saved to outputs/data/{base_filename}.csv")
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print(f"Total Data Points: {len(data)}")
    print(f"Date Range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
    print(f"Current RSI: {data['RSI'].iloc[-1]:.2f}")
    
    if data['RSI'].iloc[-1] > 70:
        rsi_status = "Overbought (>70)"
    elif data['RSI'].iloc[-1] < 30:
        rsi_status = "Oversold (<30)"
    else:
        rsi_status = "Neutral"
    
    print(f"RSI Status: {rsi_status}")
    
    if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
        macd_status = "Bullish (MACD > Signal)"
    else:
        macd_status = "Bearish (MACD < Signal)"
    
    print(f"MACD Status: {macd_status}")
    
    # Price change over the last day
    last_day = data.iloc[-int(len(data)/5):] if len(data) > 5 else data
    price_change = (last_day['Close'].iloc[-1] - last_day['Close'].iloc[0]) / last_day['Close'].iloc[0] * 100
    print(f"Price Change (Last Day): {price_change:.2f}%")
    
    return data

def main():
    """Main function to download and visualize intraday data"""
    args = parse_arguments()
    
    try:
        # Download data
        data = download_intraday_data(args.ticker, args.period, args.interval)
        
        # Calculate technical indicators
        print("Calculating technical indicators...")
        data_with_indicators = calculate_technical_indicators(data)
        
        # Visualize data
        print("Generating visualizations...")
        visualize_intraday_data(data_with_indicators, args.ticker, args.interval, args.output_name)
        
        print("\nIntraday analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in intraday analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 