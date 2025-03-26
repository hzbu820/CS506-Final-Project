import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback

# Add src directory to the Python path
sys.path.append("src")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train an LSTM model on intraday stock data')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--interval', type=str, default='5m',
                        help='Data interval (1m, 5m, 15m, 30m, 1h) (default: 5m)')
    parser.add_argument('--period', type=str, default='5d',
                        help='Period to download (1d, 5d, 1mo) (default: 5d)')
    parser.add_argument('--sequence_length', type=int, default=12,
                        help='Sequence length for LSTM (default: 12 - 1 hour for 5m data)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of LSTM (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--future_predictions', type=int, default=None,
                        help='Number of future intervals to predict (default: same as sequence_length)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Custom name for the saved model (default: auto-generated timestamp)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    return parser.parse_args()

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

class IntradayDataLoader:
    """Data loader for intraday stock data"""
    def __init__(self, ticker, period='5d', interval='5m', sequence_length=12, debug=False):
        """
        Initialize intraday data loader
        
        Args:
            ticker: Stock ticker symbol
            period: Period to download ('1d', '5d', etc.)
            interval: Data interval ('5m', '15m', '1h', etc.)
            sequence_length: Number of time steps to use for prediction
            debug: Enable debug mode with more verbose output
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.sequence_length = sequence_length
        self.debug = debug
        self.data = None
        self.processed_data = None
        self.feature_columns = None
        self.scaler = MinMaxScaler()
        
    def download_data(self):
        """Download intraday data from Yahoo Finance"""
        print(f"Downloading {self.interval} interval data for {self.ticker} over {self.period}...")
        
        try:
            # Download data with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.data = yf.download(self.ticker, period=self.period, interval=self.interval)
                    if not self.data.empty:
                        break
                    print(f"Attempt {attempt+1}/{max_retries}: No data returned, retrying...")
                except Exception as e:
                    print(f"Attempt {attempt+1}/{max_retries}: Error downloading data: {e}")
                    if attempt < max_retries - 1:
                        print("Retrying...")
            
            if self.data is None or self.data.empty:
                raise ValueError(f"Failed to download data for {self.ticker} after {max_retries} attempts")
        
            print(f"Downloaded {len(self.data)} data points.")
            
            if self.debug:
                print("Data sample:")
                print(self.data.head())
            
            # Handle missing data
            if self.data.empty:
                raise ValueError(f"No data downloaded for {self.ticker}")
            
            # Handle MultiIndex columns if present
            if isinstance(self.data.columns, pd.MultiIndex):
                # Flatten the MultiIndex columns
                self.data.columns = [col[0] for col in self.data.columns]
                
                if self.debug:
                    print("Flattened MultiIndex columns")
                    print(f"New columns: {self.data.columns}")
                
            # Remove rows with NaN values
            self.data = self.data.dropna()
            
            # Reset index to make Date a column
            self.data = self.data.reset_index()
            
            if self.debug:
                print(f"Data shape after cleaning: {self.data.shape}")
                print(f"Data columns: {self.data.columns}")
                
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def prepare_features(self):
        """Prepare features for modeling"""
        if self.data is None:
            self.download_data()
        
        try:
            df = self.data.copy()
            
            if self.debug:
                print("Preparing features...")
                print(f"Initial data shape: {df.shape}")
            
            # Calculate technical indicators
            df['Return'] = df['Close'].pct_change()
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA_Ratio'] = df['MA5'] / df['MA20']
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Range'] = df['High'] - df['Low']
            df['Range_Pct'] = df['Range'] / df['Close']
            
            # Calculate time-based features
            df['Hour'] = df['Datetime'].dt.hour
            df['Minute'] = df['Datetime'].dt.minute
            df['DayOfWeek'] = df['Datetime'].dt.dayofweek
            
            # Adding intraday momentum indicators
            df['Price_Change'] = df['Close'].diff()
            df['Momentum'] = df['Close'].diff(5)
            
            # Drop NaN values after calculations
            df = df.dropna()
            
            if self.debug:
                print(f"Data shape after feature engineering: {df.shape}")
            
            # Select features for modeling
            self.feature_columns = ['Close', 'Open', 'High', 'Low', 'Volume', 'Return', 
                                  'MA5', 'MA20', 'MA_Ratio', 'Volume_Change', 'Range', 
                                  'Range_Pct', 'Hour', 'Minute', 'DayOfWeek', 
                                  'Price_Change', 'Momentum']
            
            self.processed_data = df
            
            if self.debug:
                print(f"Selected features: {self.feature_columns}")
                print(f"Final processed data shape: {self.processed_data.shape}")
                
            return df
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def scale_data(self):
        """Scale features to [0, 1] range"""
        if self.processed_data is None:
            self.prepare_features()
            
        try:
            if self.debug:
                print("Scaling data...")
                
            # Scale selected features
            features = self.processed_data[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            # Replace with scaled values
            for i, col in enumerate(self.feature_columns):
                self.processed_data[col] = scaled_features[:, i]
                
            if self.debug:
                print("Data scaling completed.")
                
            return self.processed_data
            
        except Exception as e:
            print(f"Error scaling data: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def create_sequences(self):
        """Create sequences for LSTM training"""
        if self.processed_data is None or 'Close' not in self.processed_data.columns:
            self.scale_data()
            
        try:
            if self.debug:
                print(f"Creating sequences with length {self.sequence_length}...")
                
            # Get scaled data
            data = self.processed_data[self.feature_columns].values
            
            # Create sequences
            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i+self.sequence_length])
                # Target is the next closing price
                y.append(data[i+self.sequence_length, 0])  # Close is the first column
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
            
            if self.debug:
                print(f"Created {len(X)} sequences.")
                print(f"X shape: {X.shape}, y shape: {y.shape}")
            
            # Split into train and test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            if self.debug:
                print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
                
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"Error creating sequences: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def prepare_prediction_data(self):
        """Prepare the most recent data for prediction"""
        if self.processed_data is None:
            self.scale_data()
            
        try:
            if self.debug:
                print("Preparing prediction data...")
                
            # Get the last sequence_length data points
            recent_data = self.processed_data[self.feature_columns].values[-self.sequence_length:]
            
            # Reshape for model input
            X_pred = recent_data.reshape(1, self.sequence_length, len(self.feature_columns))
            
            if self.debug:
                print(f"Prediction data shape: {X_pred.shape}")
                
            return X_pred
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def inverse_transform_predictions(self, scaled_predictions):
        """Convert scaled predictions back to original scale"""
        try:
            # Create a dummy array with zeros for all features
            dummy = np.zeros((len(scaled_predictions), len(self.feature_columns)))
            # Put the predictions in the first column (Close price)
            dummy[:, 0] = scaled_predictions.flatten()
            # Inverse transform
            original_predictions = self.scaler.inverse_transform(dummy)[:, 0]
            
            return original_predictions
            
        except Exception as e:
            print(f"Error inverse transforming predictions: {e}")
            traceback.print_exc()
            sys.exit(1)

def train_intraday_model(args):
    """Train an LSTM model on intraday data"""
    
    try:
        # Set future predictions to sequence length if not specified
        if args.future_predictions is None:
            args.future_predictions = args.sequence_length
            
        print("=" * 60)
        print(f"Training Intraday LSTM Model for {args.ticker} with {args.interval} intervals")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Ticker: {args.ticker}")
        print(f"  Interval: {args.interval}")
        print(f"  Period: {args.period}")
        print(f"  Sequence Length: {args.sequence_length}")
        print(f"  Hidden Size: {args.hidden_size}")
        print(f"  Num Layers: {args.num_layers}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Future Predictions: {args.future_predictions}")
        print(f"  Model Name: {args.model_name if args.model_name else 'Auto-generated'}")
        print(f"  Debug Mode: {args.debug}")
        print("=" * 60)
        
        # Create output directories
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/figures', exist_ok=True)
        os.makedirs('outputs/predictions', exist_ok=True)
        
        # Initialize data loader and download data
        data_loader = IntradayDataLoader(
            args.ticker, 
            args.period, 
            args.interval, 
            args.sequence_length,
            args.debug
        )
        
        data = data_loader.download_data()
        data_loader.prepare_features()
        data_loader.scale_data()
        
        # Create sequences
        X_train, y_train, X_test, y_test = data_loader.create_sequences()
        
        # Check if we have enough data for training
        if len(X_train) < args.batch_size:
            print(f"Warning: Not enough training data (only {len(X_train)} sequences for batch size {args.batch_size})")
            print("Try using a longer period or a shorter sequence length")
            args.batch_size = max(1, len(X_train) // 2)
            print(f"Adjusting batch size to {args.batch_size}")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        input_size = X_train.shape[2]  # Number of features
        output_size = 1  # Predicting next closing price
        
        model = IntradayLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=output_size,
            dropout=0.2
        ).to(device)
        
        if args.debug:
            print(f"Model architecture:")
            print(model)
            print(f"Input size: {input_size}, Output size: {output_size}")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print("\nTraining model...")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
            val_loss = val_loss / len(test_loader)
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if args.model_name:
                    model_save_path = f'outputs/models/{args.ticker}_intraday_{args.interval}_{args.model_name}.pth'
                else:
                    model_save_path = f'outputs/models/{args.ticker}_intraday_{args.interval}_{timestamp}.pth'
                    
                torch.save(model.state_dict(), model_save_path)
                best_model_path = model_save_path
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        print(f"Best model saved to: {best_model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{args.ticker} Intraday LSTM Training ({args.interval} intervals)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set the history path based on model name or timestamp
        if args.model_name:
            history_path = f"outputs/figures/{args.ticker}_intraday_{args.interval}_{args.model_name}_training_history.png"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_path = f"outputs/figures/{args.ticker}_intraday_{args.interval}_{timestamp}_training_history.png"
            
        plt.savefig(history_path)
        plt.close()
        
        print(f"Training history saved to: {history_path}")
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test.to(device)).cpu().numpy()
            
        # Convert predictions back to original scale
        y_test_orig = data_loader.inverse_transform_predictions(y_test.numpy())
        y_pred_orig = data_loader.inverse_transform_predictions(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        
        # Calculate direction accuracy
        if len(y_test_orig) > 1:
            direction_actual = np.diff(y_test_orig) > 0
            direction_pred = np.diff(y_pred_orig) > 0
            direction_accuracy = np.mean(direction_actual == direction_pred)
        else:
            direction_accuracy = float('nan')
        
        print("\nEvaluation Metrics:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.4f}")
        
        # Generate future predictions
        num_predictions = args.future_predictions
        
        # Get most recent data for predictions
        X_recent = data_loader.prepare_prediction_data()
        X_recent_tensor = torch.FloatTensor(X_recent).to(device)
        
        # Initialize predictions array
        predictions = []
        current_input = X_recent_tensor.clone()
        
        # Generate predictions one by one
        model.eval()
        with torch.no_grad():
            for _ in range(num_predictions):
                # Get prediction
                pred = model(current_input).cpu().numpy()[0]
                predictions.append(pred[0])
                
                # Update input sequence
                # Remove first timestep and add prediction as last timestep
                new_input = current_input.clone()
                new_input[0, 0:-1, :] = new_input[0, 1:, :]
                
                # Create a new row with the prediction as Close price
                last_row = new_input[0, -1, :].clone()
                # Convert numpy.float32 to a Python float before assigning to torch tensor
                last_row[0] = float(pred[0])  # Update Close price
                
                # Add the new row
                new_input[0, -1, :] = last_row
                current_input = new_input
        
        # Convert predictions to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        original_predictions = data_loader.inverse_transform_predictions(predictions)
        
        # Create timestamp for predictions
        last_timestamp = data_loader.processed_data['Datetime'].iloc[-1]
        timestamps = []
        for i in range(num_predictions):
            # For 5m data, add 5 minutes for each prediction
            interval_minutes = int(args.interval[:-1]) if args.interval[:-1].isdigit() else 60
            next_timestamp = last_timestamp + timedelta(minutes=interval_minutes * (i+1))
            timestamps.append(next_timestamp)
        
        # Create DataFrame with predictions
        future_df = pd.DataFrame({
            'Datetime': timestamps,
            'Predicted_Price': original_predictions
        })
        
        # Save predictions to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.model_name:
            predictions_file = f"outputs/predictions/{args.ticker}_intraday_{args.interval}_{args.model_name}.csv"
            predictions_plot = f"outputs/figures/{args.ticker}_intraday_{args.interval}_{args.model_name}.png"
            history_path = f"outputs/figures/{args.ticker}_intraday_{args.interval}_{args.model_name}_training_history.png"
        else:
            predictions_file = f"outputs/predictions/{args.ticker}_intraday_{args.interval}_{timestamp}.csv"
            predictions_plot = f"outputs/figures/{args.ticker}_intraday_{args.interval}_{timestamp}.png"
            history_path = f"outputs/figures/{args.ticker}_intraday_{args.interval}_training_history.png"
        
        future_df.to_csv(predictions_file, index=False)
        print(f"\nFuture predictions saved to {predictions_file}")
        
        # Get current price
        current_price = data_loader.data['Close'].iloc[-1]
        
        # Calculate changes
        first_pred = float(future_df['Predicted_Price'].iloc[0])
        last_pred = float(future_df['Predicted_Price'].iloc[-1])
        short_term_change = (first_pred - current_price) / current_price * 100
        overall_change = (last_pred - current_price) / current_price * 100
        
        print("\nIntraday Prediction Summary:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Next Interval Prediction: ${first_pred:.2f} ({short_term_change:.2f}%)")
        print(f"Final Prediction ({num_predictions} intervals): ${last_pred:.2f} ({overall_change:.2f}%)")
        print(f"Prediction Direction: {'UP' if overall_change > 0 else 'DOWN'}")
        
        # Plot intraday predictions
        plt.figure(figsize=(14, 7))
        
        # Get historical prices (maximum of 48 data points or all available data)
        max_historical = min(48, len(data_loader.data))
        historical_data = data_loader.data.iloc[-max_historical:]
        
        # Plot historical prices
        plt.plot(historical_data['Datetime'], historical_data['Close'], 
                color='blue', label='Historical Prices')
        
        # Plot predictions
        plt.plot(future_df['Datetime'], future_df['Predicted_Price'], 
                color='red', linestyle='--', marker='o', markersize=5,
                label='Predicted Prices')
        
        # Add vertical line for current time
        plt.axvline(x=last_timestamp, color='green', linestyle='--', alpha=0.7, 
                    label='Current Time')
        
        # Annotate predictions
        annotation_interval = max(1, num_predictions // 6)  # Limit to about 6 annotations
        for i, row in future_df.iterrows():
            if i % annotation_interval == 0:  # Avoid crowding annotations
                plt.annotate(f"${row['Predicted_Price']:.2f}",
                            xy=(row['Datetime'], row['Predicted_Price']),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            fontsize=8)
        
        # Format chart
        plt.title(f'{args.ticker} Intraday Prediction ({args.interval} intervals)', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis with times
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(predictions_plot)
        plt.close()
        
        print(f"Predictions visualization saved to {predictions_plot}")
        print("=" * 60)
        
        return {
            'predictions_file': predictions_file,
            'predictions_plot': predictions_plot,
            'model_file': best_model_path,
            'history_file': history_path,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
        }
        
    except Exception as e:
        print(f"Error training intraday model: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    train_intraday_model(args) 