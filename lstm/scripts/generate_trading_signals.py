import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import traceback
import sys

# Add src directory to the Python path
sys.path.append("src")

# Import intraday data visualizer functions
from scripts.visualize_intraday_data import download_intraday_data, calculate_technical_indicators

class IntradayLSTM(nn.Module):
    """LSTM model for intraday predictions"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(IntradayLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Intraday Trading Advisor using LSTM')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--interval', type=str, default='5m',
                        help='Data interval (1m, 5m, 15m, 30m, 1h) (default: 5m)')
    parser.add_argument('--period', type=str, default='5d',
                        help='Period to download (1d, 5d, 1mo) (default: 5d)')
    parser.add_argument('--sequence_length', type=int, default=12,
                        help='Sequence length for LSTM (default: 12 - 1 hour for 5m data)')
    parser.add_argument('--future_predictions', type=int, default=12,
                        help='Number of future intervals to predict (default: 12)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (if None, will train a new model)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    return parser.parse_args()

def preprocess_data_for_lstm(data, sequence_length):
    """Prepare data for LSTM model"""
    # Features to use for LSTM
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 
                   'MA5', 'MA20', 'MACD', 'RSI', 'Volume_MA5']
    
    # Check if all features exist
    for col in feature_cols:
        if col not in data.columns:
            raise ValueError(f"Required feature {col} not found in data")
    
    # Extract features
    features = data[feature_cols].values
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X = []
    y = []
    
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(scaled_features[i+sequence_length, 0])  # Close price is the first column
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return X, y, scaler, feature_cols

def train_lstm_model(X, y, hidden_size=128, num_layers=2, learning_rate=0.001, epochs=20, batch_size=32, debug=False):
    """Train LSTM model on intraday data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    # Create model
    input_size = X.shape[2]
    output_size = 1
    
    model = IntradayLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    
    if debug:
        print(f"Model architecture:")
        print(model)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if debug and (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'outputs/models/intraday_lstm_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    
    if debug:
        print(f"Model saved to {model_path}")
    
    return model, model_path

def load_model(model_path, input_size, hidden_size=128, num_layers=2):
    """Load pre-trained LSTM model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IntradayLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def generate_predictions(model, data, scaler, feature_cols, sequence_length, num_predictions):
    """Generate future predictions using the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the latest sequence from data
    features = data[feature_cols].values
    scaled_features = scaler.transform(features)
    
    # Get last sequence
    last_sequence = scaled_features[-sequence_length:]
    
    # Convert to tensor
    current_input = torch.FloatTensor(last_sequence.reshape(1, sequence_length, len(feature_cols))).to(device)
    
    # Generate predictions
    predictions = []
    timestamps = []
    last_timestamp = data['Datetime'].iloc[-1]
    
    # Get interval in minutes
    if data['Datetime'].iloc[-1].minute != data['Datetime'].iloc[-2].minute:
        interval_minutes = data['Datetime'].iloc[-1].minute - data['Datetime'].iloc[-2].minute
        if interval_minutes <= 0:  # Handle end of hour
            interval_minutes = 60 + interval_minutes
    else:
        # Default to 5 minutes if can't determine
        interval_minutes = 5
    
    with torch.no_grad():
        for i in range(num_predictions):
            # Generate prediction
            pred = model(current_input)
            scaled_pred = pred.cpu().numpy()[0][0]
            predictions.append(scaled_pred)
            
            # Add timestamp
            next_timestamp = last_timestamp + timedelta(minutes=interval_minutes * (i+1))
            timestamps.append(next_timestamp)
            
            # Update sequence for next prediction
            new_input = current_input.clone()
            new_input[0, 0:-1, :] = new_input[0, 1:, :]
            
            # Create a new feature row with the prediction as Close
            last_row = scaled_features[-1].copy()
            last_row[0] = scaled_pred  # Close is first feature
            
            # Add to input sequence
            new_input[0, -1, :] = torch.FloatTensor(last_row)
            current_input = new_input
    
    # Convert scaled predictions back to original price
    dummy = np.zeros((len(predictions), len(feature_cols)))
    dummy[:, 0] = predictions  # Close is first feature
    original_predictions = scaler.inverse_transform(dummy)[:, 0]
    
    # Create prediction DataFrame
    future_df = pd.DataFrame({
        'Datetime': timestamps,
        'Predicted_Price': original_predictions
    })
    
    return future_df

def generate_trading_signals(data, predictions, ticker):
    """Generate trading signals based on technical indicators and predictions"""
    # Get current price and indicators
    current_price = data['Close'].iloc[-1]
    current_rsi = data['RSI'].iloc[-1]
    current_macd = data['MACD'].iloc[-1]
    current_signal = data['Signal'].iloc[-1]
    
    # Get predicted prices
    first_pred = predictions['Predicted_Price'].iloc[0]
    last_pred = predictions['Predicted_Price'].iloc[-1]
    
    # Calculate predicted changes
    short_term_change = (first_pred - current_price) / current_price * 100
    overall_change = (last_pred - current_price) / current_price * 100
    
    # Determine trend direction
    if overall_change > 0.5:
        trend = "BULLISH"
    elif overall_change < -0.5:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    
    # RSI signals
    if current_rsi > 70:
        rsi_signal = "OVERBOUGHT"
    elif current_rsi < 30:
        rsi_signal = "OVERSOLD"
    else:
        rsi_signal = "NEUTRAL"
    
    # MACD signals
    if current_macd > current_signal:
        macd_signal = "BULLISH"
    else:
        macd_signal = "BEARISH"
    
    # Combined trading signal
    if trend == "BULLISH" and (rsi_signal != "OVERBOUGHT" or macd_signal == "BULLISH"):
        trading_signal = "BUY"
    elif trend == "BEARISH" and (rsi_signal != "OVERSOLD" or macd_signal == "BEARISH"):
        trading_signal = "SELL"
    else:
        trading_signal = "HOLD"
    
    # Calculate confidence score (0-100)
    confidence = 50  # Base confidence
    
    # Adjust based on trend strength
    if abs(overall_change) > 2:
        confidence += 10
    if abs(overall_change) > 1:
        confidence += 5
    
    # Adjust based on indicator alignment
    if (trend == "BULLISH" and macd_signal == "BULLISH") or (trend == "BEARISH" and macd_signal == "BEARISH"):
        confidence += 10
    
    if (trend == "BULLISH" and rsi_signal == "OVERSOLD") or (trend == "BEARISH" and rsi_signal == "OVERBOUGHT"):
        confidence += 10
    
    # Adjust based on consistency of predictions
    pred_changes = np.diff(predictions['Predicted_Price'].values)
    if np.all(pred_changes > 0) or np.all(pred_changes < 0):
        confidence += 10  # Consistent direction
    
    # Cap at 100
    confidence = min(confidence, 100)
    
    # Create signals DataFrame
    signals = {
        'Ticker': ticker,
        'Current_Price': current_price,
        'Predicted_Price_Short': first_pred,
        'Predicted_Price_Long': last_pred,
        'Predicted_Change_Short': short_term_change,
        'Predicted_Change_Long': overall_change,
        'Trend': trend,
        'RSI': current_rsi,
        'RSI_Signal': rsi_signal,
        'MACD_Signal': macd_signal,
        'Trading_Signal': trading_signal,
        'Confidence': confidence
    }
    
    return signals

def visualize_predictions_and_signals(data, predictions, signals, ticker, interval):
    """Visualize predictions and trading signals"""
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Price and Predictions Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot historical prices (last 48 data points or all)
    history_len = min(48, len(data))
    historical_data = data.iloc[-history_len:]
    
    ax1.plot(historical_data['Datetime'], historical_data['Close'], 
            color='blue', linewidth=1.5, label='Historical Price')
    
    # Plot predictions
    ax1.plot(predictions['Datetime'], predictions['Predicted_Price'], 
            color='red', linestyle='--', marker='o', markersize=4,
            label='Predicted Price')
    
    # Add moving averages
    ax1.plot(historical_data['Datetime'], historical_data['MA5'], 
            color='green', linewidth=1, label='5-period MA')
    ax1.plot(historical_data['Datetime'], historical_data['MA20'], 
            color='purple', linewidth=1, label='20-period MA')
    
    # Add vertical line for current time
    ax1.axvline(x=data['Datetime'].iloc[-1], color='black', linestyle='--', alpha=0.7, 
               label='Current Time')
    
    # Add signal annotation
    signal_colors = {
        'BUY': 'green',
        'SELL': 'red',
        'HOLD': 'gray'
    }
    
    # Add trading signal annotation
    signal_text = (
        f"Signal: {signals['Trading_Signal']} ({signals['Confidence']}% confidence)\n"
        f"Trend: {signals['Trend']}\n"
        f"RSI: {signals['RSI']:.1f} ({signals['RSI_Signal']})\n"
        f"Final Predicted Change: {signals['Predicted_Change_Long']:.2f}%"
    )
    
    ax1.annotate(
        signal_text,
        xy=(predictions['Datetime'].iloc[0], predictions['Predicted_Price'].iloc[0]),
        xytext=(30, 30),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
        color=signal_colors[signals['Trading_Signal']],
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color=signal_colors[signals['Trading_Signal']])
    )
    
    # Format first chart
    ax1.set_title(f'{ticker} Intraday Trading Signals ({interval} intervals)', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot volume
    ax2.bar(historical_data['Datetime'], historical_data['Volume'], color='blue', alpha=0.5)
    ax2.axvline(x=data['Datetime'].iloc[-1], color='black', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis with dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Save and close
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{ticker}_intraday_signals_{timestamp}.png")
    plt.close()
    
    # 2. Technical indicators chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # RSI chart
    ax1.plot(historical_data['Datetime'], historical_data['RSI'], color='purple', linewidth=1.5)
    ax1.axhline(y=70, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.fill_between(historical_data['Datetime'], 70, 30, color='gray', alpha=0.1)
    ax1.axvline(x=data['Datetime'].iloc[-1], color='black', linestyle='--', alpha=0.7)
    
    # Add current RSI
    ax1.plot(data['Datetime'].iloc[-1], signals['RSI'], 'ro', markersize=8)
    
    ax1.set_title(f'{ticker} RSI (14-period)', fontsize=16)
    ax1.set_ylabel('RSI', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # MACD chart
    ax2.plot(historical_data['Datetime'], historical_data['MACD'], color='blue', linewidth=1.5, label='MACD')
    ax2.plot(historical_data['Datetime'], historical_data['Signal'], color='red', linewidth=1, label='Signal')
    ax2.bar(historical_data['Datetime'], historical_data['MACD_Histogram'], color='green', alpha=0.5, label='Histogram')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axvline(x=data['Datetime'].iloc[-1], color='black', linestyle='--', alpha=0.7)
    
    ax2.set_title(f'{ticker} MACD', fontsize=16)
    ax2.set_ylabel('MACD', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis with dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Save and close
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{ticker}_intraday_indicators_{timestamp}.png")
    plt.close()
    
    return timestamp

def save_results(predictions, signals, ticker, timestamp):
    """Save prediction and signal results to files"""
    # Create output directory
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/signals', exist_ok=True)
    
    # Save predictions to CSV
    predictions_file = f"outputs/predictions/{ticker}_intraday_predictions_{timestamp}.csv"
    predictions.to_csv(predictions_file, index=False)
    
    # Save signals to CSV
    signals_df = pd.DataFrame([signals])
    signals_file = f"outputs/signals/{ticker}_intraday_signals_{timestamp}.csv"
    signals_df.to_csv(signals_file, index=False)
    
    print(f"\nResults saved:")
    print(f"- Predictions: {predictions_file}")
    print(f"- Signals: {signals_file}")
    
    return predictions_file, signals_file

def print_trading_advice(signals):
    """Print trading advice based on signals"""
    print("\n" + "=" * 60)
    print(f"INTRADAY TRADING ADVICE FOR {signals['Ticker']}")
    print("=" * 60)
    print(f"Current Price: ${signals['Current_Price']:.2f}")
    print(f"Current RSI: {signals['RSI']:.2f} ({signals['RSI_Signal']})")
    print(f"MACD Signal: {signals['MACD_Signal']}")
    print("-" * 60)
    print(f"Trend: {signals['Trend']}")
    print(f"Predicted Change (Short-term): {signals['Predicted_Change_Short']:.2f}%")
    print(f"Predicted Change (End of period): {signals['Predicted_Change_Long']:.2f}%")
    print("-" * 60)
    print(f"Trading Signal: {signals['Trading_Signal']}")
    print(f"Confidence: {signals['Confidence']}%")
    print("-" * 60)
    
    # Detailed advice
    if signals['Trading_Signal'] == 'BUY':
        print("Recommendation: Consider BUYING or HOLDING")
        
        if signals['RSI_Signal'] == 'OVERSOLD':
            print("• Stock appears oversold according to RSI")
        
        if signals['MACD_Signal'] == 'BULLISH':
            print("• MACD indicates positive momentum")
        
        if signals['Predicted_Change_Long'] > 1:
            print(f"• Model predicts significant upside of {signals['Predicted_Change_Long']:.2f}%")
        else:
            print(f"• Model predicts modest upside of {signals['Predicted_Change_Long']:.2f}%")
            
    elif signals['Trading_Signal'] == 'SELL':
        print("Recommendation: Consider SELLING or REDUCING POSITION")
        
        if signals['RSI_Signal'] == 'OVERBOUGHT':
            print("• Stock appears overbought according to RSI")
        
        if signals['MACD_Signal'] == 'BEARISH':
            print("• MACD indicates negative momentum")
        
        if signals['Predicted_Change_Long'] < -1:
            print(f"• Model predicts significant downside of {signals['Predicted_Change_Long']:.2f}%")
        else:
            print(f"• Model predicts modest downside of {signals['Predicted_Change_Long']:.2f}%")
            
    else:  # HOLD
        print("Recommendation: HOLD or NO ACTION NEEDED")
        print("• Technical indicators and predicted trend are mixed or neutral")
        print(f"• Model predicts price change of {signals['Predicted_Change_Long']:.2f}%")
        
    print("=" * 60)
    print("DISCLAIMER: This is algorithmic trading advice and should be")
    print("used for informational purposes only. Always do your own research")
    print("before making investment decisions.")
    print("=" * 60)

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        print(f"Starting Intraday Trading Advisor for {args.ticker}")
        print(f"Using {args.interval} intervals with sequence length {args.sequence_length}")
        
        # Download and process data
        print("\nDownloading intraday data...")
        data = download_intraday_data(args.ticker, args.period, args.interval)
        
        print("Calculating technical indicators...")
        data_with_indicators = calculate_technical_indicators(data)
        
        # Prepare data for LSTM
        print("Preparing data for LSTM model...")
        X, y, scaler, feature_cols = preprocess_data_for_lstm(data_with_indicators, args.sequence_length)
        
        # Train or load model
        if args.model_path is None:
            print("Training LSTM model...")
            model, model_path = train_lstm_model(X, y, debug=args.debug)
            print(f"Model trained and saved to {model_path}")
        else:
            print(f"Loading pre-trained model from {args.model_path}...")
            model = load_model(args.model_path, len(feature_cols))
            model_path = args.model_path
        
        # Generate predictions
        print(f"Generating {args.future_predictions} future predictions...")
        predictions = generate_predictions(model, data_with_indicators, scaler, feature_cols, 
                                          args.sequence_length, args.future_predictions)
        
        # Generate trading signals
        print("Generating trading signals...")
        signals = generate_trading_signals(data_with_indicators, predictions, args.ticker)
        
        # Visualize results
        print("Creating visualizations...")
        timestamp = visualize_predictions_and_signals(data_with_indicators, predictions, signals, 
                                                    args.ticker, args.interval)
        
        # Save results
        predictions_file, signals_file = save_results(predictions, signals, args.ticker, timestamp)
        
        # Print trading advice
        print_trading_advice(signals)
        
        print("\nIntraday Trading Advisor completed successfully!")
        
    except Exception as e:
        print(f"Error in Intraday Trading Advisor: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 