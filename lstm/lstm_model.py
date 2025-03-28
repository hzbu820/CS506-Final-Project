"""
LSTM Model for Time Series Forecasting

This script implements an LSTM model for time series forecasting of stock prices.
It includes data loading, preprocessing, model creation, training, and visualization.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# For PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# For data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Extract the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMTrainer:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs, patience=5):
        best_val_loss = float('inf')
        counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load the best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        return self.train_losses, self.val_losses

def prepare_lstm_data(data, seq_length, target_col='close', forecast_horizon=1):
    """
    Prepare data for LSTM model training
    
    Args:
        data (pd.DataFrame): Input dataframe
        seq_length (int): Length of the sequence for LSTM input
        target_col (str): Column to predict
        forecast_horizon (int): Number of steps to forecast
        
    Returns:
        X (np.array): Input sequences
        y (np.array): Target values
        scaler (MinMaxScaler): Fitted scaler for inverse transformation
        features (list): List of feature names used during training
    """
    # Select features to use
    features = ['open', 'high', 'low', 'close', 'volume', 
                'ema_9', 'sma_14', 'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'atr_14']
    
    # Make sure all features are available
    features = [f for f in features if f in data.columns]
    
    # Create a copy of the dataframe with only the features we need
    df = data[features].copy()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Get the index of the target column
    target_idx = features.index(target_col)
    
    # Create input sequences and output
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - forecast_horizon + 1):
        # Input sequence
        X.append(scaled_data[i:(i + seq_length)])
        # Target value (close price)
        y.append(scaled_data[i + seq_length:i + seq_length + forecast_horizon, target_idx])
    
    return np.array(X), np.array(y), scaler, features

def forecast_future(model, last_sequence, scaler, future_steps=10, device='cpu', data=None, features=None):
    """
    Generate future predictions
    
    Args:
        model: Trained LSTM model
        last_sequence: The last sequence from the dataset
        scaler: The fitted scaler for inverse transformation
        future_steps: Number of steps to forecast
        device: Device to run the model on
        data: DataFrame used for getting column index
        features: List of feature names used during training
    
    Returns:
        predictions: Array of forecasted values
    """
    model.eval()
    curr_seq = last_sequence.clone()
    predictions = []
    target_idx = features.index('close')  # Use features list instead of data DataFrame
    
    with torch.no_grad():
        for _ in range(future_steps):
            # Get prediction for the next step
            curr_pred = model(curr_seq.unsqueeze(0).to(device)).cpu().numpy()[0][0]
            predictions.append(curr_pred)
            
            # Create a new sequence by dropping the first entry and adding the new prediction
            new_step = curr_seq[-1].clone()
            # Update only the target column with the new prediction
            new_step = new_step.unsqueeze(0)
            curr_seq = torch.cat((curr_seq[1:], new_step), dim=0)
    
    # Inverse transform the predictions
    dummy = np.zeros((len(predictions), len(features)))  # Use features list length
    dummy[:, target_idx] = predictions
    pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    return pred_transformed

def plot_results(test_dates, actual_transformed, forecast_dates=None, forecasted_values=None):
    """Plot the actual vs predicted prices and optionally forecast"""
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_transformed, label='Actual Price')
    
    if forecast_dates is not None and forecasted_values is not None:
        plt.plot(forecast_dates, forecasted_values, 'r--', label='Forecasted Prices')
    
    plt.title('AAPL Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model(actual, predicted):
    """Calculate and print performance metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    return mse, rmse, mae, r2

def moving_average_baseline(data, window_size=24, target_col='close'):
    """
    Creates a simple moving average baseline for comparison
    
    Args:
        data: DataFrame with time series data
        window_size: Size of the moving average window
        target_col: Target column to predict
        
    Returns:
        ma_predictions: Moving average predictions
    """
    # Get the target column data
    target_data = data[target_col].values
    
    # Create moving average predictions (shifted by 1 to predict next value)
    ma_predictions = np.zeros_like(target_data)
    
    # Fill initial values with the original data (can't calculate MA yet)
    ma_predictions[:window_size] = target_data[:window_size]
    
    # Calculate moving average for the rest
    for i in range(window_size, len(target_data)):
        ma_predictions[i] = np.mean(target_data[i-window_size:i])
    
    return ma_predictions

def evaluate_against_baseline(actual, model_predicted, baseline_predicted, start_idx=0):
    """
    Evaluate model performance against a baseline
    
    Args:
        actual: Array of actual values
        model_predicted: Array of model predictions
        baseline_predicted: Array of baseline predictions
        start_idx: Index to start evaluation (to skip initial values)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Skip the first few predictions where baseline can't predict
    actual = actual[start_idx:]
    model_predicted = model_predicted[start_idx:]
    baseline_predicted = baseline_predicted[start_idx:]
    
    # Calculate metrics for model
    model_mse = mean_squared_error(actual, model_predicted)
    model_rmse = np.sqrt(model_mse)
    model_mae = mean_absolute_error(actual, model_predicted)
    model_r2 = r2_score(actual, model_predicted)
    
    # Calculate metrics for baseline
    baseline_mse = mean_squared_error(actual, baseline_predicted)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mae = mean_absolute_error(actual, baseline_predicted)
    baseline_r2 = r2_score(actual, baseline_predicted)
    
    # Calculate improvement over baseline
    mse_improvement = (baseline_mse - model_mse) / baseline_mse * 100
    rmse_improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    mae_improvement = (baseline_mae - model_mae) / baseline_mae * 100
    
    # Print comparison
    print("\n--- Model vs Baseline Comparison ---")
    print(f"MSE  - Model: {model_mse:.4f}, Baseline: {baseline_mse:.4f}, Improvement: {mse_improvement:.2f}%")
    print(f"RMSE - Model: {model_rmse:.4f}, Baseline: {baseline_rmse:.4f}, Improvement: {rmse_improvement:.2f}%")
    print(f"MAE  - Model: {model_mae:.4f}, Baseline: {baseline_mae:.4f}, Improvement: {mae_improvement:.2f}%")
    print(f"R²   - Model: {model_r2:.4f}, Baseline: {baseline_r2:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='black')
    plt.plot(model_predicted, label='LSTM Model', color='blue')
    plt.plot(baseline_predicted, label='Moving Average Baseline', color='red', linestyle='--')
    plt.title('Model vs Baseline Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_vs_baseline.png')
    
    return {
        'model_metrics': {
            'mse': model_mse,
            'rmse': model_rmse,
            'mae': model_mae,
            'r2': model_r2
        },
        'baseline_metrics': {
            'mse': baseline_mse,
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': baseline_r2
        },
        'improvements': {
            'mse': mse_improvement,
            'rmse': rmse_improvement,
            'mae': mae_improvement
        }
    }

def out_of_sample_test(model, seq_length, scaler, features, target_col, device='cpu'):
    """
    Test the model on out-of-sample data (a different stock or time period)
    
    Args:
        model: Trained LSTM model
        seq_length: Sequence length used for training
        scaler: Scaler used for training data
        features: List of features used for training
        target_col: Target column name
        device: Device to run the model on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    try:
        # Try to load a different stock (MSFT) as out-of-sample data
        file_path = 'CS506-Final-Project-main/data_processed/yfinance/full/MSFT_15m_full.csv'
        oos_data = pd.read_csv(file_path)
        
        # Process the new data similar to the training data
        date_col = 'datetime' if 'datetime' in oos_data.columns else 'date'
        oos_data[date_col] = pd.to_datetime(oos_data[date_col])
        oos_data.set_index(date_col, inplace=True)
        
        # Filter for features that were used in training
        available_features = [f for f in features if f in oos_data.columns]
        if len(available_features) != len(features):
            print(f"Warning: Only {len(available_features)}/{len(features)} features available in out-of-sample data")
        
        # Select a subset of the data for testing (last 300 entries)
        oos_data = oos_data.tail(300).copy()
        
        # Create sequences from the out-of-sample data
        oos_df = oos_data[available_features].copy()
        
        # Get the target column data for evaluation
        target_idx = available_features.index(target_col)
        target_data = oos_df[target_col].values
        
        # Scale the data using the original scaler
        scaled_data = scaler.transform(oos_df)
        
        # Create sequences
        X_oos = []
        for i in range(len(scaled_data) - seq_length):
            X_oos.append(scaled_data[i:i+seq_length])
        
        X_oos = np.array(X_oos)
        X_oos_tensor = torch.FloatTensor(X_oos)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X_oos_tensor.to(device)).cpu().numpy()
        
        # Inverse transform predictions
        dummy = np.zeros((len(predictions), len(available_features)))
        dummy[:, target_idx] = predictions.flatten()
        pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
        
        # Get actual values for comparison
        actual = target_data[seq_length:]
        
        # Calculate metrics
        mse = mean_squared_error(actual, pred_transformed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, pred_transformed)
        r2 = r2_score(actual, pred_transformed)
        
        # Print results
        print("\n--- Out-of-Sample Testing Results (MSFT) ---")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(oos_data.index[seq_length:], actual, label='Actual MSFT Price')
        plt.plot(oos_data.index[seq_length:], pred_transformed, label='Predicted MSFT Price')
        plt.title('Out-of-Sample Testing on MSFT')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('out_of_sample_test.png')
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
    except Exception as e:
        print(f"Error in out-of-sample testing: {e}")
        return None

def walk_forward_validation(data, features, target_col, seq_length, hidden_size=64, num_layers=2, 
                           window_size=300, step_size=60, n_windows=3, batch_size=32, epochs=20):
    """
    Perform walk-forward validation
    
    Args:
        data: Full DataFrame with all data
        features: List of features to use
        target_col: Target column to predict
        seq_length: Sequence length for LSTM
        hidden_size: Hidden size for LSTM
        num_layers: Number of LSTM layers
        window_size: Size of each validation window
        step_size: Steps to move forward for next window
        n_windows: Number of windows to evaluate
        batch_size: Batch size for training
        epochs: Number of epochs for each window
        
    Returns:
        metrics: List of performance metrics for each window
    """
    print("\n--- Walk-Forward Validation ---")
    
    # Select only the required features
    df = data[features].copy()
    
    # Get the total length of data available
    total_len = len(df)
    
    # Check if we have enough data for the requested validation
    required_len = window_size + (n_windows - 1) * step_size
    if total_len < required_len:
        print(f"Not enough data for {n_windows} windows. Need {required_len} but have {total_len}")
        n_windows = (total_len - window_size) // step_size + 1
        print(f"Reducing to {n_windows} windows")
    
    # Initialize result containers
    window_metrics = []
    
    # Prepare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loop through each window
    for i in range(n_windows):
        print(f"\nTraining on window {i+1}/{n_windows}")
        
        # Calculate window start and end indices
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > total_len:
            print(f"Not enough data for window {i+1}. Skipping.")
            break
        
        # Get the window data
        window_data = df.iloc[start_idx:end_idx].copy()
        
        # Calculate train/test split indices within this window
        train_size = int(0.8 * len(window_data))
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(window_data)
        
        # Get target column index
        target_idx = features.index(target_col)
        
        # Create sequences
        X, y = [], []
        for j in range(len(scaled_data) - seq_length):
            X.append(scaled_data[j:(j + seq_length)])
            y.append(scaled_data[j + seq_length, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        X_train, X_test = X[:train_size-seq_length], X[train_size-seq_length:]
        y_train, y_test = y[:train_size-seq_length], y[train_size-seq_length:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        
        # Create DataLoader for batched training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the model
        input_size = X.shape[2]  # Number of features
        output_size = 1
        dropout = 0.2
        
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Initialize the trainer
        trainer = LSTMTrainer(model, criterion, optimizer, device)
        
        # Train the model for this window
        train_losses, val_losses = trainer.train(train_loader, test_loader, epochs, patience=5)
        
        # Make predictions on the test set
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor.to(device)).cpu().numpy()
        
        # Prepare for inverse transform
        dummy = np.zeros((len(predictions), len(features)))
        dummy[:, target_idx] = predictions.flatten()
        pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
        
        # Get actual values
        dummy = np.zeros((len(y_test), len(features)))
        dummy[:, target_idx] = y_test
        actual_transformed = scaler.inverse_transform(dummy)[:, target_idx]
        
        # Calculate metrics
        mse = mean_squared_error(actual_transformed, pred_transformed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_transformed, pred_transformed)
        r2 = r2_score(actual_transformed, pred_transformed)
        
        # Store the metrics
        window_metrics.append({
            'window': i+1,
            'date_range': (data.index[start_idx], data.index[end_idx-1]),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        # Print metrics for this window
        print(f"Window {i+1} Results:")
        print(f"Date range: {data.index[start_idx]} to {data.index[end_idx-1]}")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Plot the metrics across windows
    if window_metrics:
        plt.figure(figsize=(12, 8))
        
        windows = [m['window'] for m in window_metrics]
        mses = [m['mse'] for m in window_metrics]
        rmses = [m['rmse'] for m in window_metrics]
        maes = [m['mae'] for m in window_metrics]
        r2s = [m['r2'] for m in window_metrics]
        
        plt.subplot(2, 2, 1)
        plt.plot(windows, mses, marker='o')
        plt.title('MSE over Windows')
        plt.xlabel('Window')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(windows, rmses, marker='o')
        plt.title('RMSE over Windows')
        plt.xlabel('Window')
        plt.ylabel('RMSE')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(windows, maes, marker='o')
        plt.title('MAE over Windows')
        plt.xlabel('Window')
        plt.ylabel('MAE')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(windows, r2s, marker='o')
        plt.title('R² over Windows')
        plt.xlabel('Window')
        plt.ylabel('R²')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('walk_forward_validation.png')
    
    return window_metrics

def hidden_data_validation(data_path, test_period_days=10, seq_length=24, hidden_size=64, 
                          num_layers=2, batch_size=32, epochs=30, plot_results=True):
    """
    Performs validation by hiding the most recent portion of data, training on the rest,
    and then predicting the hidden portion to compare with actual values.
    
    Args:
        data_path: Path to the historical data file
        test_period_days: Number of days to hide/use for testing
        seq_length: Sequence length for LSTM input
        hidden_size: Hidden size for LSTM model
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        epochs: Number of training epochs
        plot_results: Whether to plot the results
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\n--- Hidden Data Validation ---")
    print(f"Loading data from {data_path}")
    
    # Load the data
    data = pd.read_csv(data_path)
    
    # Convert date column
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    
    # Drop any rows with NaN values
    data = data.dropna()
    
    # Calculate how many rows to hide (assuming 15-minute data, 4*24*days)
    intervals_per_day = 24 * 4  # 15-minute intervals in a day
    hidden_rows = test_period_days * intervals_per_day
    
    if hidden_rows >= len(data):
        hidden_rows = int(len(data) * 0.2)  # Use 20% if requested period is too large
        print(f"Requested test period too large. Using {hidden_rows} rows instead.")
    
    # Split data into visible (training) and hidden (testing) portions
    visible_data = data.iloc[:-hidden_rows].copy()
    hidden_data = data.iloc[-hidden_rows:].copy()
    
    print(f"Data split:")
    print(f"- Visible data: {visible_data.index.min()} to {visible_data.index.max()} ({len(visible_data)} rows)")
    print(f"- Hidden data: {hidden_data.index.min()} to {hidden_data.index.max()} ({len(hidden_data)} rows)")
    
    # Prepare visible data for LSTM - match the function signature
    X, y, scaler, selected_features = prepare_lstm_data(visible_data, seq_length, 'close', 1)
    
    # Split visible data into train and validation
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create DataLoader for batched training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = X.shape[2]  # Number of features
    output_size = 1
    dropout = 0.2
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize the trainer
    trainer = LSTMTrainer(model, criterion, optimizer, device)
    
    # Train the model
    print("\nTraining model on visible data...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs, patience=10)
    
    # Now prepare the hidden data sequences for prediction
    print("\nPredicting hidden data...")
    
    # We need some overlap from the visible data to make the first prediction
    overlap_data = visible_data.iloc[-seq_length:].copy()
    full_test_data = pd.concat([overlap_data, hidden_data])
    
    # Prepare the full test data using the existing scaler
    X_test, _, _, _ = prepare_lstm_data(full_test_data, seq_length, 'close', 1)
    
    # Get the actual values for comparison
    target_idx = selected_features.index('close')
    actual_values = hidden_data['close'].values
    
    # Make predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.FloatTensor(X_test[i:i+1]).to(device)
            output = model(x).cpu().numpy().flatten()[0]
            predictions.append(output)
            
            # Stop when we've generated enough predictions for the hidden data
            if len(predictions) >= len(hidden_data):
                break
    
    # Inverse transform predictions
    dummy = np.zeros((len(predictions), len(selected_features)))
    dummy[:, target_idx] = predictions
    pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Trim to match hidden data length
    pred_transformed = pred_transformed[:len(hidden_data)]
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, pred_transformed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, pred_transformed)
    r2 = r2_score(actual_values, pred_transformed)
    
    # Print results
    print("\n--- Hidden Data Prediction Results ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Calculate percent error
    avg_price = np.mean(actual_values)
    pct_mae = (mae / avg_price) * 100
    print(f"Percent MAE: {pct_mae:.2f}%")
    
    # Plot the results
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(hidden_data.index, actual_values, label='Actual Prices', color='blue')
        plt.plot(hidden_data.index, pred_transformed, label='Predicted Prices', color='red', linestyle='--')
        plt.title('Prediction of Hidden Data')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('hidden_data_validation.png')
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pct_mae': pct_mae,
        'predictions': pred_transformed,
        'actual': actual_values,
        'dates': hidden_data.index
    }

def main():
    # Load the data
    try:
        # Try to load from the data_processed directory
        file_path = 'CS506-Final-Project-main/data_processed/yfinance/full/AAPL_15m_full.csv'
        data = pd.read_csv(file_path)
    except:
        # If the file doesn't exist, use our sample data
        file_path = 'LSTM.csv'
        data = pd.read_csv(file_path)
        
    # Display the first few rows
    print(f"Loaded data from {file_path}")
    print(data.head())
    
    # Filter for our target symbol and set date as index
    if 'symbol' in data.columns:
        data = data[data['symbol'] == 'AAPL']
    
    # Convert datetime to proper format
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    
    # Drop any rows with NaN values
    data = data.dropna()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Perform hidden data validation (hide last 5 days of data)
    hidden_validation_results = hidden_data_validation(
        file_path,
        test_period_days=5,  # Hide last 5 days
        seq_length=24,
        hidden_size=64,
        num_layers=2,
        batch_size=32,
        epochs=30
    )
    
    # Save the results to the summary file
    with open('model_evaluation_summary.txt', 'a') as f:
        f.write("\nHidden Data Validation (Last 5 Days)\n")
        f.write("----------------------------------\n")
        f.write(f"MSE: {hidden_validation_results['mse']:.4f}\n")
        f.write(f"RMSE: {hidden_validation_results['rmse']:.4f}\n")
        f.write(f"MAE: {hidden_validation_results['mae']:.4f}\n")
        f.write(f"R²: {hidden_validation_results['r2']:.4f}\n")
        f.write(f"Percent MAE: {hidden_validation_results['pct_mae']:.2f}%\n\n")
    
    print("\nHidden data validation complete. Results added to 'model_evaluation_summary.txt'")
    
    # Continue with the existing code...
    # Add these lines to keep the main functionality in place
    seq_length = 24  # Use 24 time steps (6 hours of 15-minute data)
    forecast_horizon = 1  # Predict just the next step
    X, y, scaler, features = prepare_lstm_data(data, seq_length, 'close', forecast_horizon)
    
    # Split into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nContinuing with regular model training...")
    print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create DataLoader for batched training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    output_size = forecast_horizon
    dropout = 0.2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize the trainer
    trainer = LSTMTrainer(model, criterion, optimizer, device)
    
    # Train the model
    epochs = 50
    train_losses, val_losses = trainer.train(train_loader, test_loader, epochs, patience=10)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    # Make predictions on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device)).cpu().numpy()
    
    # Prepare for inverse transform
    target_idx = features.index('close')  # Use features list instead of data.columns
    y_test_flat = y_test.reshape(-1)
    pred_flat = predictions.reshape(-1)
    
    # Inverse transform the predictions and actual values
    dummy = np.zeros((len(pred_flat), len(features)))  # Use features list length
    dummy[:, target_idx] = pred_flat
    pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    dummy = np.zeros((len(y_test_flat), len(features)))  # Use features list length
    dummy[:, target_idx] = y_test_flat
    actual_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(actual_transformed, label='Actual Price')
    plt.plot(pred_transformed, label='Predicted Price', alpha=0.7)
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    
    # Calculate performance metrics
    metrics = evaluate_model(actual_transformed, pred_transformed)
    
    # Create moving average baseline and compare
    # Get the right portion of data for comparison
    test_start_idx = train_size + seq_length
    test_portion = data.iloc[test_start_idx:test_start_idx+len(actual_transformed)].copy()
    
    # Create moving average predictions
    ma_predictions = moving_average_baseline(test_portion, window_size=seq_length, target_col='close')
    
    # Compare model performance against baseline
    comparison_metrics = evaluate_against_baseline(
        actual_transformed, 
        pred_transformed, 
        ma_predictions,
        start_idx=seq_length
    )
    
    # Perform out-of-sample testing
    oos_metrics = out_of_sample_test(model, seq_length, scaler, features, 'close', device)
    
    # Perform walk-forward validation (using smaller epochs for speed)
    wf_metrics = walk_forward_validation(
        data, 
        features,
        'close', 
        seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        window_size=400,  # Use 400 data points per window
        step_size=100,    # Step 100 data points forward each time
        n_windows=3,      # Evaluate 3 windows
        batch_size=batch_size,
        epochs=20         # Fewer epochs for speed
    )
    
    # Forecast future values
    last_sequence = X_test_tensor[-1]
    future_steps = 10
    forecasted_values = forecast_future(model, last_sequence, scaler, future_steps, device, data, features)  # Pass features list
    
    # Prepare the x-axis for plotting
    test_start_idx = train_size + seq_length
    test_dates = data.index[test_start_idx:test_start_idx+len(actual_transformed)]
    
    # Create future dates for forecasting
    last_date = test_dates[-1]
    forecast_dates = pd.date_range(start=last_date, periods=future_steps+1, freq='15min')[1:]
    
    # Plot the historical prices and the forecasted prices
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_transformed, label='Historical Prices')
    plt.plot(forecast_dates, forecasted_values, 'r--', label='Forecasted Prices')
    plt.title('AAPL Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forecast_results.png')
    plt.show()
    
    # Save summary to a file
    with open('model_evaluation_summary.txt', 'w') as f:
        f.write("LSTM MODEL EVALUATION SUMMARY\n")
        f.write("============================\n\n")
        
        f.write("Basic Model Performance\n")
        f.write("----------------------\n")
        f.write(f"MSE: {metrics[0]:.4f}\n")
        f.write(f"RMSE: {metrics[1]:.4f}\n")
        f.write(f"MAE: {metrics[2]:.4f}\n")
        f.write(f"R²: {metrics[3]:.4f}\n\n")
        
        f.write("Baseline Comparison\n")
        f.write("------------------\n")
        f.write(f"Model MSE: {comparison_metrics['model_metrics']['mse']:.4f}\n")
        f.write(f"Baseline MSE: {comparison_metrics['baseline_metrics']['mse']:.4f}\n")
        f.write(f"MSE Improvement: {comparison_metrics['improvements']['mse']:.2f}%\n\n")
        
        f.write(f"Model RMSE: {comparison_metrics['model_metrics']['rmse']:.4f}\n")
        f.write(f"Baseline RMSE: {comparison_metrics['baseline_metrics']['rmse']:.4f}\n")
        f.write(f"RMSE Improvement: {comparison_metrics['improvements']['rmse']:.2f}%\n\n")
        
        f.write(f"Model MAE: {comparison_metrics['model_metrics']['mae']:.4f}\n")
        f.write(f"Baseline MAE: {comparison_metrics['baseline_metrics']['mae']:.4f}\n")
        f.write(f"MAE Improvement: {comparison_metrics['improvements']['mae']:.2f}%\n\n")
        
        f.write(f"Model R²: {comparison_metrics['model_metrics']['r2']:.4f}\n")
        f.write(f"Baseline R²: {comparison_metrics['baseline_metrics']['r2']:.4f}\n\n")
        
        if oos_metrics:
            f.write("Out-of-Sample Testing (MSFT)\n")
            f.write("----------------------------\n")
            f.write(f"MSE: {oos_metrics['mse']:.4f}\n")
            f.write(f"RMSE: {oos_metrics['rmse']:.4f}\n")
            f.write(f"MAE: {oos_metrics['mae']:.4f}\n")
            f.write(f"R²: {oos_metrics['r2']:.4f}\n\n")
        
        f.write("Walk-Forward Validation\n")
        f.write("----------------------\n")
        for i, metrics in enumerate(wf_metrics):
            f.write(f"Window {i+1}:\n")
            f.write(f"  Date range: {metrics['date_range'][0]} to {metrics['date_range'][1]}\n")
            f.write(f"  MSE: {metrics['mse']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"  MAE: {metrics['mae']:.4f}\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n\n")
    
    print("\nEvaluation complete. Summary saved to 'model_evaluation_summary.txt'")

if __name__ == "__main__":
    main() 