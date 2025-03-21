import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from load_custom_data import train_with_custom_data, make_predictions

"""
This script demonstrates a real-world example of using the LSTM model
with a stock price dataset. It shows the complete workflow from data loading
to model training and prediction.
"""

def load_stock_data():
    """
    This function simulates loading a stock price dataset.
    In a real-world scenario, you would load your own data file.
    
    For demonstration, we're creating synthetic stock price data
    that resembles real stock behavior with trend, seasonality, and noise.
    """
    # Create synthetic stock price data
    days = 1000
    t = np.arange(days)
    
    # Base price with upward trend
    base_price = 100 + 0.05 * t
    
    # Add seasonality (weekly pattern)
    weekly_pattern = 5 * np.sin(2 * np.pi * t / 7)
    
    # Add longer-term cycles (quarterly)
    quarterly_pattern = 15 * np.sin(2 * np.pi * t / 90)
    
    # Add realistic noise (with some volatility clusters)
    np.random.seed(42)
    noise = np.random.normal(0, 1, days)
    # Create volatility clusters
    volatility = 1 + 0.5 * np.sin(2 * np.pi * t / 180)
    noise = noise * volatility
    
    # Combine components to create price
    price = base_price + weekly_pattern + quarterly_pattern + noise.cumsum()
    
    # Ensure positive prices
    price = 100 + (price - min(price))
    
    # Create volume (correlated with price volatility)
    volume = 10000 + 5000 * volatility + 1000 * np.random.normal(0, 1, days)
    volume = np.abs(volume).astype(int)
    
    # Create a dataframe
    dates = pd.date_range(start='2020-01-01', periods=days)
    df = pd.DataFrame({
        'Date': dates,
        'Open': price * 0.99,
        'High': price * 1.02,
        'Low': price * 0.98,
        'Close': price,
        'Volume': volume
    })
    
    # Calculate additional features often used in stock price prediction
    # 1. Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 2. Price momentum (percent change)
    df['Returns'] = df['Close'].pct_change()
    df['Returns_5d'] = df['Close'].pct_change(periods=5)
    
    # 3. Volatility (standard deviation of returns)
    df['Volatility_5d'] = df['Returns'].rolling(window=5).std()
    
    # 4. Trading range
    df['Range'] = df['High'] - df['Low']
    
    # Drop rows with NaN values (due to rolling calculations)
    df = df.dropna()
    
    # For this example, let's use these features
    selected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                         'MA5', 'MA20', 'Returns', 'Volatility_5d', 'Range']
    
    # The target will be next day's closing price
    data = df[selected_features].values
    
    # Print dataset info
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {selected_features}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return data, df, selected_features

def main():
    """Main function demonstrating the complete workflow"""
    
    print("=== Loading Stock Price Data ===")
    data, df, features = load_stock_data()
    
    # Show some statistics about the data
    print("\n=== Data Statistics ===")
    for i, feature in enumerate(features):
        print(f"{feature}: mean={data[:, i].mean():.2f}, std={data[:, i].std():.2f}, min={data[:, i].min():.2f}, max={data[:, i].max():.2f}")
    
    # Visualize the stock price data
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.plot(df['Date'], df['MA20'], label='20-day MA', alpha=0.7)
    plt.title('Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('stock_price_history.png')
    plt.close()
    
    print("\n=== Training LSTM Model ===")
    # We'll predict the 'Close' price (index 3 in our feature set)
    target_idx = 3
    
    # Train with custom parameters suitable for stock price prediction
    model, history, data_processor, test_X, test_y = train_with_custom_data(
        data,
        sequence_length=20,  # Use 20 days of history to predict next day
        train_split=0.8,     # Use 80% for training, 20% for testing
        batch_size=32,
        hidden_size=128,     # Larger network for complex price patterns
        num_layers=2,        # Two LSTM layers
        learning_rate=0.001,
        epochs=100           # Train for 100 epochs with early stopping
    )
    
    print("\n=== Making Predictions ===")
    # Make predictions for close price
    predictions = make_predictions(model, test_X, test_y, data_processor, target_idx)
    
    # Calculate additional performance metrics specific to financial data
    test_y_numpy = data_processor.inverse_transform(test_y.numpy())[:, target_idx]
    predictions_flat = predictions.flatten()
    
    # Direction accuracy
    direction_actual = np.diff(test_y_numpy) > 0
    direction_pred = np.diff(predictions_flat) > 0
    direction_accuracy = np.mean(direction_actual == direction_pred)
    
    print(f"\nDirection Prediction Accuracy: {direction_accuracy:.4f}")
    
    # Calculate profit/loss of a simple trading strategy
    # Buy when predicted to go up, sell when predicted to go down
    pnl = []
    position = 0
    
    for i in range(1, len(predictions_flat)):
        if direction_pred[i-1]:  # Predicted up
            if position == 0:
                # Buy
                entry_price = predictions_flat[i-1]
                position = 1
        else:  # Predicted down
            if position == 1:
                # Sell
                exit_price = predictions_flat[i-1]
                pnl.append((exit_price - entry_price) / entry_price)
                position = 0
    
    if pnl:
        avg_return_per_trade = np.mean(pnl)
        total_return = np.prod([1 + r for r in pnl]) - 1
        
        print(f"Average Return per Trade: {avg_return_per_trade:.4f}")
        print(f"Total Strategy Return: {total_return:.4f}")
    
    print("\nLSTM model training and prediction complete. Results and plots saved.")

if __name__ == "__main__":
    main() 