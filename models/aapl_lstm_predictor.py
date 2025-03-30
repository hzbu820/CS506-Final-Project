#!/usr/bin/env python
"""
AAPL LSTM Price Predictor
Uses enhanced LSTM to predict AAPL stock price 15 minutes ahead
Uses preprocessed 15-minute interval data from CS506-Final-Project
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Add the src directory to the path so Python can find the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

def log_message(message):
    """Print a message with a timestamp and also write to a log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Write to log file
    with open('outputs/aapl_lstm_log.txt', 'a') as f:
        f.write(log_msg + '\n')

def create_advanced_lstm_model():
    """Create an enhanced LSTM model with bidirectional layers and attention mechanism"""
    class AttentionLayer(torch.nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.hidden_size = hidden_size
            self.attention = torch.nn.Linear(hidden_size, 1)
            
        def forward(self, lstm_output):
            # lstm_output shape: (batch_size, seq_len, hidden_size)
            attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            return context_vector, attention_weights
    
    class EnhancedLSTMModel(torch.nn.Module):
        """Enhanced LSTM model for stock price prediction"""
        
        def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
            super(EnhancedLSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Bidirectional LSTM layers
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            # Attention mechanism
            self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
            
            # Fully connected layers with residual connections
            self.fc1 = torch.nn.Linear(hidden_size * 2, 128)
            self.bn1 = torch.nn.BatchNorm1d(128)
            self.dropout1 = torch.nn.Dropout(0.3)
            self.fc2 = torch.nn.Linear(128, 64)
            self.bn2 = torch.nn.BatchNorm1d(64)
            self.dropout2 = torch.nn.Dropout(0.3)
            self.fc3 = torch.nn.Linear(64, 32)
            self.bn3 = torch.nn.BatchNorm1d(32)
            
            # Output layer for price prediction (single value)
            self.fc_out = torch.nn.Linear(32, 1)
            
        def forward(self, x):
            # Initialize hidden state with zeros
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 for bidirectional
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            
            # Forward propagate LSTM
            lstm_out, _ = self.lstm(x, (h0, c0))
            
            # Apply attention mechanism
            attn_output, _ = self.attention(lstm_out)
            
            # Process through fully connected layers with residual connections
            x = self.fc1(attn_output)
            x = self.bn1(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout1(x)
            
            residual = x
            x = self.fc2(x)
            x = self.bn2(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout2(x)
            
            x = self.fc3(x)
            x = self.bn3(x)
            x = torch.nn.functional.relu(x)
            
            # Output layer
            x = self.fc_out(x)
            
            return x
        
        def predict(self, x):
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                predictions = self.forward(x)
            return predictions
    
    return EnhancedLSTMModel

def load_preprocessed_data(timeframe='15m'):
    """Load preprocessed AAPL data from CS506-Final-Project"""
    log_message(f"Loading preprocessed AAPL {timeframe} data from CS506-Final-Project")
    
    # Define the path to the preprocessed data
    data_path = f"CS506-Final-Project-main/data_processed/yfinance/full/AAPL_{timeframe}_full.csv"
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Convert datetime to proper format and set as index
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        log_message(f"Loaded {len(data)} rows of preprocessed data")
        
        # Display available columns
        log_message(f"Columns available: {', '.join(data.columns.tolist())}")
        
        return data
        
    except Exception as e:
        log_message(f"Error loading preprocessed data: {e}")
        raise

def get_stock_data(ticker="AAPL", timeframe='15m'):
    """Get stock data from preprocessed file"""
    # Use the preprocessed data from CS506-Final-Project
    df = load_preprocessed_data(timeframe)
    
    # Handle column name differences - rename to match expected format
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # Apply mapping only for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    return df

def train_price_prediction_model(ticker="AAPL", sequence_length=60, hidden_size=128, num_layers=2, 
                        learning_rate=0.0005, batch_size=32, epochs=100, train_split=0.8):
    """Train an enhanced LSTM model for price prediction"""
    log_message(f"Starting price prediction training for {ticker} with 15m intervals")
    
    # Get stock data (15-minute intervals)
    df = get_stock_data(ticker, timeframe='15m')
    
    # Create target variable - next time period's close price
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()  # Remove last row which will have NaN Target
    
    # Prepare features
    # Select relevant columns for features
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'ema_9', 'sma_14', 'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        'atr_14', 'pct_change', 'log_return'
    ]
    
    # Only include columns that exist in the DataFrame
    feature_columns = [col for col in feature_cols if col in df.columns]
    features = df[feature_columns].copy()
    target = df['Target']
    
    # ADD ENHANCED TECHNICAL INDICATORS
    log_message("Adding enhanced technical indicators...")
    
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    features = df[feature_columns].copy()
    
    # Keep the most valuable indicators
    
    # 1. Stochastic Oscillator (effective for identifying overbought/oversold conditions)
    window = 14  # Standard window
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    features['stoch_k'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
    
    # 2. On-Balance Volume (effective for volume analysis)
    # Calculate OBV more efficiently
    obv = pd.Series(0, index=df.index)
    # Create masks for up and down days
    up_days = df['Close'].diff() > 0
    down_days = df['Close'].diff() < 0
    
    # Assign values using masks
    obv[up_days] = df['Volume'][up_days]
    obv[down_days] = -df['Volume'][down_days]
    
    # Calculate cumulative sum
    features['obv'] = obv.cumsum()
    features['obv_ema'] = features['obv'].ewm(span=20).mean()
    
    # 3. Money Flow Index (valuable for price/volume relationship)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    # Handle the positive and negative money flow more efficiently
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    
    # Use masks for positive and negative changes
    price_change_mask = typical_price.diff() > 0
    positive_flow[price_change_mask] = raw_money_flow[price_change_mask]
    negative_flow[~price_change_mask & (typical_price.diff() < 0)] = raw_money_flow[~price_change_mask & (typical_price.diff() < 0)]
    
    # Calculate MFI
    pos_mf_sum = positive_flow.rolling(window=14, min_periods=1).sum()
    neg_mf_sum = negative_flow.rolling(window=14, min_periods=1).sum()
    
    # Avoid division by zero
    mf_ratio = np.where(neg_mf_sum != 0, pos_mf_sum / neg_mf_sum, 100)
    features['mfi'] = 100 - (100 / (1 + mf_ratio))
    
    # 4. Rate of Change (effective momentum indicator)
    features['roc_5'] = df['Close'].pct_change(periods=5) * 100
    features['roc_10'] = df['Close'].pct_change(periods=10) * 100
    
    # 5. Bollinger Band %B (effective for volatility and trend)
    if all(col in features.columns for col in ['bollinger_upper', 'bollinger_lower']):
        # Calculate %B safely
        bb_range = features['bollinger_upper'] - features['bollinger_lower']
        # Handle division by zero
        features['bb_pct_b'] = np.where(
            bb_range != 0, 
            (df['Close'] - features['bollinger_lower']) / bb_range, 
            0.5
        )
    
    # 6. Average True Range (important volatility measure)
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': abs(df['High'] - df['Close'].shift(1)),
        'lc': abs(df['Low'] - df['Close'].shift(1))
    }).max(axis=1)
    features['atr'] = tr.rolling(window=14).mean()
    
    # 7. VWAP (Volume-Weighted Average Price - critical for intraday trading)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).cumsum()
    cum_vol = df['Volume'].cumsum()
    features['vwap'] = np.where(cum_vol != 0, cum_tp_vol / cum_vol, df['Close'])
    
    # 8. Price/Volume Ratio
    features['price_volume_ratio'] = df['Close'] / df['Volume'].rolling(window=5).mean()
    
    # 9. Williams %R (effective for momentum and reversal signals)
    features['williams_r'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    
    # 10. Relative Strength Index (if not already present)
    if 'rsi_14' not in features.columns:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        features['rsi'] = 100 - (100 / (1 + rs))
    
    # 11. Linear Regression Slope (simplified calculation)
    # Import scipy.stats locally to ensure it's available
    from scipy import stats as scipy_stats
    
    price_series = df['Close'].values
    slopes = []
    window = 10
    
    for i in range(len(df)):
        if i < window:
            slopes.append(0)
        else:
            x = np.arange(window)
            y = price_series[i-window:i]
            slope, _, _, _, _ = scipy_stats.linregress(x, y)
            slopes.append(slope)
    
    features['price_slope'] = slopes
    
    # 12. Cyclical time encoding for intraday patterns
    # Extract hour from datetime index
    if hasattr(df.index, 'hour'):
        df_hours = df.index.hour.values
        # Encode hour as sine and cosine to represent cyclical nature
        features['hour_sin'] = np.sin(2 * np.pi * df_hours / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df_hours / 24)
    
    # Fill remaining NaN values
    features = features.ffill().bfill().fillna(0)
    
    # Log the number of new features added
    new_feature_count = len(features.columns) - len(feature_columns)
    log_message(f"Added {new_feature_count} new technical indicators")
    log_message(f"Using {len(features.columns)} features: {', '.join(features.columns)}")
    
    # Ensure target is aligned with features and also has no NaNs
    target = df['Target']
    target = target.loc[features.index]
    target = target.ffill().bfill()
    
    # Verify we have data
    if len(features) == 0:
        raise ValueError("No data left after creating technical indicators. Check for excessive NaN values.")
    
    log_message(f"Dataset size after feature engineering: {len(features)} rows")
    
    # Scale features
    scaler_x = StandardScaler()
    scaled_features = scaler_x.fit_transform(features)
    log_message(f"Scaled features shape: {scaled_features.shape}")
    
    # Scale target to improve model performance
    scaler_y = StandardScaler()
    scaled_target = scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(scaled_features, scaled_target, sequence_length)
    
    # Split into training and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    log_message(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    log_message(f"X_train_tensor shape: {X_train_tensor.shape}")
    
    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    
    # Get the actual input size from the tensor shape
    # The input size should be the number of features
    input_size = X_train_tensor.shape[2]
    log_message(f"LSTM input size: {input_size} features")
    
    # Create enhanced LSTM model
    EnhancedLSTMModel = create_advanced_lstm_model()
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Use MSE loss for regression
    criterion = torch.nn.MSELoss()
    
    # Use Adam optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    
    log_message(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_message(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            
            # Save the best model
            os.makedirs('outputs/models', exist_ok=True)
            torch.save(model.state_dict(), f'outputs/models/{ticker}_price_lstm.pth')
            log_message(f"Saved model with val loss: {val_loss:.6f}")
        else:
            counter += 1
            
        if counter >= patience:
            log_message(f"Early stopping after {epoch+1} epochs")
            break
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Evaluate final model on test data
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert predictions back to original scale
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)
    
    all_preds_original = scaler_y.inverse_transform(all_preds)
    all_targets_original = scaler_y.inverse_transform(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets_original, all_preds_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets_original, all_preds_original)
    r2 = r2_score(all_targets_original, all_preds_original)
    
    # Calculate Mean Absolute Percentage Error
    mape = np.mean(np.abs((all_targets_original - all_preds_original) / all_targets_original)) * 100
    
    # Calculate accuracy - percentage of predictions within 0.5% of actual value
    accuracy_threshold = 0.005  # 0.5%
    within_threshold = np.abs((all_targets_original - all_preds_original) / all_targets_original) < accuracy_threshold
    accuracy = np.mean(within_threshold) * 100
    
    log_message("\nFinal Model Performance:")
    log_message(f"MSE: {mse:.6f}")
    log_message(f"RMSE: {rmse:.6f}")
    log_message(f"MAE: {mae:.6f}")
    log_message(f"R²: {r2:.6f}")
    log_message(f"MAPE: {mape:.4f}%")
    log_message(f"Accuracy (within 0.5%): {accuracy:.4f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 7))
    
    # Get the last 200 samples for visual clarity
    n_samples = min(200, len(all_preds_original))
    
    # Get corresponding dates - Calculate the starting index in the original dataframe
    # Since the test data is the last portion of the dataset after train_test_split
    start_idx = max(0, len(df) - len(all_preds_original))
    test_dates = df.index[start_idx:start_idx + n_samples]
    
    # Display the actual date range
    log_message(f"Data covers from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    log_message(f"Test data covers from {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
    
    # Plot with dates on x-axis
    plt.plot(test_dates, all_targets_original[-n_samples:], label='Actual Prices', color='blue', linewidth=2)
    plt.plot(test_dates, all_preds_original[-n_samples:], label='Predicted Prices', color='orange', linewidth=2, linestyle='--')
    
    # Format the date axis for better visibility
    plt.gcf().autofmt_xdate()
    import matplotlib.dates as mdates
    
    # Create date formatter and locator to ensure all months are visible
    date_fmt = mdates.DateFormatter('%b %d')  # Format as "Dec 05", "Jan 15", etc.
    
    # Set up the x-axis with major ticks at month boundaries and minor ticks at days
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # One tick per month
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[1, 15]))  # Ticks on 1st and 15th
    
    # Force x-axis limits to show the full date range
    plt.xlim(test_dates[0], test_dates[-1])
    
    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # Add a text box with the date range
    date_range_text = f"Date Range: {test_dates[0].strftime('%b %d, %Y')} to {test_dates[-1].strftime('%b %d, %Y')}"
    plt.figtext(0.5, 0.01, date_range_text, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add title and labels
    plt.title('AAPL Price Prediction', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=10)
    
    plt.savefig(f'outputs/{ticker}_price_prediction_results.png')
    
    log_message(f"Results saved to outputs/{ticker}_price_prediction_results.png")
    
    return model, feature_columns, scaler_x, scaler_y

def create_sequences(data, targets, seq_length):
    """
    Create sequences for LSTM input
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

def main():
    try:
        log_message("Starting AAPL price prediction with enhanced LSTM model using 15-minute data")
        
        # Train the price prediction model using 15-minute interval data
        model, feature_columns, scaler_x, scaler_y = train_price_prediction_model(
            ticker="AAPL",
            sequence_length=60,  # Use 60 previous 15-minute intervals
            hidden_size=128,
            num_layers=2,
            learning_rate=0.0005,
            batch_size=32,
            epochs=100,
            train_split=0.8  # 80% training, 20% testing
        )
        
        log_message("Training complete!")
        
    except Exception as e:
        import traceback
        log_message(f"Error: {str(e)}")
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main() 