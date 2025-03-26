import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(EnhancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature extraction layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Prediction layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attention_weights = self.attention(out)
        context_vector = torch.sum(attention_weights * out, dim=1)
        
        # Prediction with residual connections
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

def create_sequences(data, seq_length, future_steps=1):
    xs, ys = [], []
    for i in range(len(data) - seq_length - future_steps + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+future_steps, 0]  # Only predict the closing price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def calculate_technical_indicators(data):
    # Copy the dataframe to avoid modifying the original one
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BBup'] = df['MA20'] + 2*df['Close'].rolling(window=20).std()
    df['BBdown'] = df['MA20'] - 2*df['Close'].rolling(window=20).std()
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    df['PriceChange'] = df['Close'].pct_change()
    df['PriceMomentum'] = df['PriceChange'].rolling(window=5).mean()
    
    # Volume features
    if 'Volume' in df.columns:
        df['VolChange'] = df['Volume'].pct_change()
        df['VolMA10'] = df['Volume'].rolling(window=10).mean()
        df['Vol_Price_Corr'] = df['Close'].rolling(window=10).corr(df['Volume'])
    
    # Drop NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def download_and_prepare_data(ticker, start_date, end_date, interval, sequence_length, future_predictions, add_features=True):
    try:
        # Download data
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date} with {interval} interval")
        max_retries = 3
        for i in range(max_retries):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                if len(data) == 0:
                    if i < max_retries - 1:
                        logger.warning(f"Attempt {i+1}: No data downloaded. Retrying...")
                        continue
                    else:
                        logger.error(f"Failed to download data after {max_retries} attempts")
                        return None, None, None
                break
            except Exception as e:
                if i < max_retries - 1:
                    logger.warning(f"Attempt {i+1}: Error downloading data: {e}. Retrying...")
                else:
                    logger.error(f"Failed to download data after {max_retries} attempts: {e}")
                    return None, None, None
        
        # If columns are MultiIndex (happens sometimes with yfinance), flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        logger.info(f"Downloaded {len(data)} data points")
        
        # Handle timezones and reset index
        data = data.tz_localize(None) if data.index.tz is not None else data
        data = data.reset_index()
        
        # Add technical indicators if required
        if add_features:
            data = calculate_technical_indicators(data)
            logger.info(f"Added technical indicators, data shape: {data.shape}")
        
        # Prepare for scaling
        if add_features:
            # Separate features from datetime
            dates = data['Datetime'].values
            # Select numeric columns
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            # Ensure 'Close' is the first column for easier processing
            if 'Close' in feature_columns:
                feature_columns.remove('Close')
                feature_columns = ['Close'] + feature_columns
            features = data[feature_columns].values
        else:
            # Use only OHLC data
            dates = data['Datetime'].values
            features = data[['Close', 'Open', 'High', 'Low']].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences
        X, y = create_sequences(scaled_features, sequence_length, future_predictions)
        
        # Get last sequence for future prediction
        last_sequence = scaled_features[-sequence_length:]
        last_sequence = last_sequence.reshape(1, sequence_length, -1)

        # Get dates for future prediction
        last_date = dates[-1]
        prediction_dates = []
        
        # Convert to datetime if it's not already
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
            
        # Create dates for future predictions based on the interval
        for i in range(future_predictions):
            if interval.endswith('m'):  # Minutes
                mins = int(interval[:-1])
                prediction_dates.append(last_date + timedelta(minutes=mins * (i+1)))
            elif interval.endswith('h'):  # Hours
                hours = int(interval[:-1])
                prediction_dates.append(last_date + timedelta(hours=hours * (i+1)))
            elif interval.endswith('d'):  # Days
                days = int(interval[:-1])
                prediction_dates.append(last_date + timedelta(days=days * (i+1)))
            else:  # Default to daily
                prediction_dates.append(last_date + timedelta(days=i+1))
                
        logger.info(f"Data preparation completed, X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, last_sequence, scaler, prediction_dates, features, dates
    
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        return None, None, None, None, None, None, None

def train_model(X, y, hidden_size, num_layers, learning_rate, epochs, batch_size, model_name, ticker, interval):
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X.shape[2]  # Number of features
    output_dim = y.shape[1]  # Number of future predictions
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedLSTM(input_dim, hidden_size, num_layers, output_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 15
    
    logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            os.makedirs('outputs/models', exist_ok=True)
            model_save_path = f'outputs/models/{ticker}_{interval}_{model_name}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch,
                'input_dim': input_dim,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'output_dim': output_dim
            }, model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
            
        logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
    
    # Plot and save training history
    os.makedirs('outputs/figures', exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axhline(y=best_val_loss, color='r', linestyle='-', alpha=0.3, label=f'Best Val Loss: {best_val_loss:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{ticker} LSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    history_path = f'outputs/figures/{ticker}_{interval}_{model_name}_training_history.png'
    plt.savefig(history_path)
    plt.close()
    
    # Load best model for prediction
    model_path = f'outputs/models/{ticker}_{interval}_{model_name}.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, device, best_val_loss

def evaluate_model(model, X, y, scaler, device):
    model.eval()
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Get original scale predictions and actual values
    # Create dummy array with same number of features as the original data
    original_features = scaler.n_features_in_
    
    # Reconstruct predictions and actual values to original scale
    predictions_rescaled = []
    actual_values_rescaled = []
    
    for i in range(len(predictions)):
        # For predictions
        dummy_array = np.zeros((len(predictions[i]), original_features))
        dummy_array[:, 0] = predictions[i]  # Assuming first column is Close price
        predictions_rescaled.append(scaler.inverse_transform(dummy_array)[:, 0])
        
        # For actual values
        dummy_array = np.zeros((len(y[i]), original_features))
        dummy_array[:, 0] = y[i]  # Assuming first column is Close price
        actual_values_rescaled.append(scaler.inverse_transform(dummy_array)[:, 0])
    
    # Flatten for metrics calculation
    predictions_flat = np.concatenate(predictions_rescaled)
    actual_values_flat = np.concatenate(actual_values_rescaled)
    
    # Calculate metrics
    mse = mean_squared_error(actual_values_flat, predictions_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values_flat, predictions_flat)
    r2 = r2_score(actual_values_flat, predictions_flat)
    
    logger.info(f"Model Evaluation Metrics:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def predict_future(model, last_sequence, scaler, prediction_dates, ticker, interval, model_name, device):
    model.eval()
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(last_sequence).to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()[0]
    
    # Get original scale predictions
    original_features = scaler.n_features_in_
    dummy_array = np.zeros((len(predictions), original_features))
    dummy_array[:, 0] = predictions  # Assuming first column is Close price
    predictions_rescaled = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Create dataframe with predictions
    prediction_df = pd.DataFrame({
        'Datetime': prediction_dates,
        'Predicted_Price': predictions_rescaled
    })
    
    # Save predictions to CSV
    os.makedirs('outputs/predictions', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prediction_path = f'outputs/predictions/{ticker}_{interval}_{model_name}_{timestamp}.csv'
    prediction_df.to_csv(prediction_path, index=False)
    
    # Visualize predictions
    current_price = prediction_df['Predicted_Price'].iloc[0]
    final_price = prediction_df['Predicted_Price'].iloc[-1]
    change_pct = ((final_price / current_price) - 1) * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df['Datetime'], prediction_df['Predicted_Price'], marker='o', linestyle='-')
    
    # Add price labels
    plt.annotate(f"${final_price:.2f} ({change_pct:+.2f}%)", 
                 xy=(prediction_df['Datetime'].iloc[-1], final_price),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=12, fontweight='bold')
                 
    # Format chart
    plt.title(f'{ticker} Price Prediction ({interval} intervals)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add direction
    if change_pct > 0:
        direction = "UP"
        color = 'green'
    else:
        direction = "DOWN"
        color = 'red'
        
    plt.annotate(f"Direction: {direction}", 
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=14, fontweight='bold', color=color,
                 bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.2))
                 
    # Save figure
    figure_path = f'outputs/figures/{ticker}_{interval}_{model_name}_prediction_{timestamp}.png'
    plt.savefig(figure_path)
    plt.close()
    
    logger.info(f"Prediction results:")
    logger.info(f"Initial price: ${current_price:.2f}")
    logger.info(f"Final predicted price: ${final_price:.2f}")
    logger.info(f"Change: {change_pct:.2f}%")
    logger.info(f"Direction: {direction}")
    logger.info(f"Predictions saved to: {prediction_path}")
    logger.info(f"Figure saved to: {figure_path}")
    
    return prediction_df

def parse_arguments():
    parser = argparse.ArgumentParser(description='Enhanced LSTM Stock Price Prediction')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of history to use if start_date not specified')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for LSTM')
    parser.add_argument('--future_predictions', type=int, default=5, help='Number of future periods to predict')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', type=str, default='enhanced', help='Model name for saving')
    parser.add_argument('--add_features', action='store_true', help='Add technical indicators as features')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set up output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Set date range
    if args.start_date is None:
        end_date = datetime.now() if args.end_date is None else datetime.strptime(args.end_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=args.days)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        end_date = datetime.now().strftime('%Y-%m-%d') if args.end_date is None else args.end_date
    
    logger.info(f"Starting enhanced prediction for {args.ticker} from {start_date} to {end_date} with {args.interval} interval")
    
    # Download and prepare data
    X, y, last_sequence, scaler, prediction_dates, features, dates = download_and_prepare_data(
        args.ticker, start_date, end_date, args.interval, 
        args.sequence_length, args.future_predictions, args.add_features
    )
    
    if X is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Training or loading model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.model_path is not None:
        # Load pre-trained model
        try:
            logger.info(f"Loading pre-trained model from {args.model_path}")
            checkpoint = torch.load(args.model_path)
            input_dim = checkpoint.get('input_dim', X.shape[2])
            hidden_size = checkpoint.get('hidden_size', args.hidden_size)
            num_layers = checkpoint.get('num_layers', args.num_layers)
            output_dim = checkpoint.get('output_dim', args.future_predictions)
            
            model = EnhancedLSTM(input_dim, hidden_size, num_layers, output_dim).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            logger.info(f"Model loaded successfully, validation loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Training new model instead")
            model, device, best_val_loss = train_model(
                X, y, args.hidden_size, args.num_layers, args.learning_rate, 
                args.epochs, args.batch_size, args.model_name, args.ticker, args.interval
            )
    else:
        # Train new model
        model, device, best_val_loss = train_model(
            X, y, args.hidden_size, args.num_layers, args.learning_rate, 
            args.epochs, args.batch_size, args.model_name, args.ticker, args.interval
        )
    
    # Evaluate model
    metrics = evaluate_model(model, X, y, scaler, device)
    
    # Predict future prices
    predictions = predict_future(
        model, last_sequence, scaler, prediction_dates, 
        args.ticker, args.interval, args.model_name, device
    )
    
    logger.info("Enhanced prediction completed successfully")
    
    return metrics, predictions

if __name__ == "__main__":
    main() 