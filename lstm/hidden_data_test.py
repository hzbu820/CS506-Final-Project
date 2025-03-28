import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the LSTM model and training classes from lstm_model.py
from lstm_model import LSTMModel, LSTMTrainer, prepare_lstm_data

def hidden_data_test(data_path, hidden_percentage=0.2, seq_length=24, 
                    hidden_size=64, num_layers=2, batch_size=32, epochs=30):
    """
    Test LSTM model by hiding a portion of the data and predicting it
    
    Args:
        data_path: Path to the data file
        hidden_percentage: Percentage of data to hide for testing (0.0-1.0)
        seq_length: Sequence length for LSTM
        hidden_size: Hidden size for LSTM model
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        None, but saves results to file and displays plots
    """
    print(f"Loading data from {data_path}")
    
    # Load the data
    data = pd.read_csv(data_path)
    
    # Convert date column
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    
    # Drop any rows with NaN values
    data = data.dropna()
    
    # Calculate the number of rows to hide
    total_rows = len(data)
    hidden_rows = int(total_rows * hidden_percentage)
    
    print(f"Total data rows: {total_rows}")
    print(f"Hiding {hidden_rows} rows ({hidden_percentage*100:.1f}% of data)")
    
    # Split data into visible and hidden portions
    visible_data = data.iloc[:-hidden_rows].copy()
    hidden_data = data.iloc[-hidden_rows:].copy()
    
    print(f"Visible data range: {visible_data.index.min()} to {visible_data.index.max()}")
    print(f"Hidden data range: {hidden_data.index.min()} to {hidden_data.index.max()}")
    
    # Prepare data for LSTM
    X, y, scaler, features = prepare_lstm_data(visible_data, seq_length, 'close', 1)
    
    # Split into training and validation sets (80% train, 20% validation)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
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
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = X.shape[2]  # Number of features
    output_size = 1
    dropout = 0.2
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize trainer
    trainer = LSTMTrainer(model, criterion, optimizer, device)
    
    # Train the model
    print("\nTraining model on visible data...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs, patience=10)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('hidden_data_training_loss.png')
    plt.close()
    
    # Now we need to predict the hidden data
    print("\nPredicting hidden data...")
    
    # We need some overlap from the visible data to create the first sequence
    overlap_rows = seq_length
    overlap_data = visible_data.iloc[-overlap_rows:].copy()
    
    # Combine overlap with hidden data to create continuous sequences
    combined_data = pd.concat([overlap_data, hidden_data])
    
    # Process this combined data
    X_test, _, _, _ = prepare_lstm_data(combined_data, seq_length, 'close', 1)
    
    # Get target column index
    target_idx = features.index('close')
    
    # Create a simpler approach - use first sequence to predict each next point one at a time
    model.eval()
    predictions = []
    
    target_len = len(hidden_data)
    print(f"Need to predict {target_len} data points")
    
    with torch.no_grad():
        try:
            # Use the initial sequence from the visible data
            sequence = torch.FloatTensor(X_test[0:1]).to(device)
            
            # For each time step in the hidden data
            for i in range(target_len):
                # Make a prediction
                pred = model(sequence).cpu().numpy()[0][0]
                predictions.append(pred)
                
                # Create new sequence by sliding window forward
                if i < target_len - 1:
                    # Get the next row of features from the test data (except the target)
                    next_features = X_test[i+1, -1].copy()
                    
                    # Create a new sequence by shifting and appending
                    new_sequence = sequence.clone()
                    
                    # Shift all rows up one
                    new_sequence[0, :-1] = new_sequence[0, 1:]
                    
                    # Update the last row with next features and our prediction
                    new_sequence[0, -1] = torch.FloatTensor(next_features)
                    
                    # Set the target column to our prediction
                    new_sequence[0, -1, target_idx] = pred
                    
                    # Update sequence for next iteration
                    sequence = new_sequence
                    
            print(f"Successfully generated {len(predictions)} predictions")
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
    
    # Check that we have correct number of predictions
    if len(predictions) != target_len:
        print(f"WARNING: Prediction count mismatch. Got {len(predictions)}, expected {target_len}")
        
        # If not enough predictions, duplicate the last one
        if len(predictions) < target_len:
            print("Padding predictions with last value")
            last_pred = predictions[-1] if predictions else 0.0
            while len(predictions) < target_len:
                predictions.append(last_pred)
        else:
            # If too many predictions, truncate
            print("Truncating extra predictions")
            predictions = predictions[:target_len]
    
    # Verify that our prediction array now matches expected length
    print(f"Final prediction count: {len(predictions)}, hidden data count: {target_len}")
    
    # Inverse transform predictions
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, target_idx] = predictions
    pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Get actual values
    actual_values = hidden_data['close'].values
    
    print(f"Prediction array shape: {pred_transformed.shape}, Actual array shape: {actual_values.shape}")
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, pred_transformed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, pred_transformed)
    r2 = r2_score(actual_values, pred_transformed)
    
    # Calculate percentage error
    avg_price = np.mean(actual_values)
    pct_mae = (mae / avg_price) * 100
    
    # Print results
    print("\n--- Hidden Data Prediction Results ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Percentage MAE: {pct_mae:.2f}%")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(hidden_data.index, actual_values, label='Actual Prices', color='blue')
    plt.plot(hidden_data.index, pred_transformed, label='Predicted Prices', color='red', linestyle='--')
    plt.title('Prediction of Hidden Data')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hidden_data_prediction.png')
    
    # Plot full dataset with highlighted prediction area
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['close'], label='Full Data', color='gray', alpha=0.7)
    plt.plot(visible_data.index, visible_data['close'], label='Training Data', color='blue')
    plt.plot(hidden_data.index, hidden_data['close'], label='Actual Future Data', color='green')
    plt.plot(hidden_data.index, pred_transformed, label='Predicted Future Data', color='red', linestyle='--')
    
    # Add a vertical line where visible data ends
    split_date = visible_data.index[-1]
    plt.axvline(x=split_date, color='black', linestyle='--')
    plt.annotate('Training End / Prediction Start', 
                 xy=(split_date, data['close'].min()),
                 xytext=(split_date, data['close'].min() - (data['close'].max() - data['close'].min()) * 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center')
    
    plt.title('AAPL Stock Price - Training vs Hidden Data Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('full_data_with_prediction.png')
    
    # Save results to a file
    with open('hidden_data_results.txt', 'w') as f:
        f.write("LSTM HIDDEN DATA PREDICTION RESULTS\n")
        f.write("==================================\n\n")
        f.write(f"Data file: {data_path}\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Hidden rows: {hidden_rows} ({hidden_percentage*100:.1f}%)\n\n")
        
        f.write(f"Visible data range: {visible_data.index.min()} to {visible_data.index.max()}\n")
        f.write(f"Hidden data range: {hidden_data.index.min()} to {hidden_data.index.max()}\n\n")
        
        f.write("Model Parameters:\n")
        f.write(f"- Sequence Length: {seq_length}\n")
        f.write(f"- Hidden Size: {hidden_size}\n")
        f.write(f"- Number of Layers: {num_layers}\n")
        f.write(f"- Training Epochs: {epochs}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"- Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"- Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"- R-squared (R²): {r2:.4f}\n")
        f.write(f"- Percentage MAE: {pct_mae:.2f}%\n\n")
        
        # Sample of predictions vs actual
        f.write("Sample Predictions vs Actual:\n")
        f.write("Date                  | Actual Price | Predicted Price | Error\n")
        f.write("--------------------- | ------------ | --------------- | -----\n")
        
        # Sample 10 evenly spaced rows
        sample_indices = np.linspace(0, len(hidden_data)-1, 10, dtype=int)
        for idx in sample_indices:
            date = hidden_data.index[idx]
            actual = actual_values[idx]
            pred = pred_transformed[idx]
            error = actual - pred
            f.write(f"{date} | ${actual:.2f} | ${pred:.2f} | ${error:.2f}\n")
    
    print("\nResults saved to 'hidden_data_results.txt'")
    print("Plots saved to 'hidden_data_training_loss.png', 'hidden_data_prediction.png', and 'full_data_with_prediction.png'")
    return mse, rmse, mae, r2, pct_mae


if __name__ == "__main__":
    # Path to data file
    data_path = 'CS506-Final-Project-main/data_processed/yfinance/full/AAPL_15m_full.csv'
    
    # Hide 20% of the data for testing
    hidden_percentage = 0.2
    
    # Run the test
    hidden_data_test(
        data_path=data_path,
        hidden_percentage=hidden_percentage,
        seq_length=24,
        hidden_size=64,
        num_layers=2,
        batch_size=32,
        epochs=30
    ) 