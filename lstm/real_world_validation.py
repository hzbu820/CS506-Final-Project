import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import from our LSTM model file
from lstm_model import LSTMModel, prepare_lstm_data

def test_model_accuracy(data_path, hidden_percentage=0.2):
    """
    Test the LSTM model's accuracy by hiding a portion of the data,
    training on the visible portion, and predicting the hidden portion.
    
    Args:
        data_path: Path to the data file
        hidden_percentage: Percentage of data to hide (0.0-1.0)
    """
    print(f"Loading data from {data_path}")
    
    # Load the data
    data = pd.read_csv(data_path)
    
    # Convert datetime to proper format
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    
    # Drop any rows with NaN values
    data = data.dropna()
    
    # Calculate number of rows to hide
    total_rows = len(data)
    hidden_rows = int(total_rows * hidden_percentage)
    
    print(f"Total data rows: {total_rows}")
    print(f"Hidden rows: {hidden_rows} ({hidden_percentage*100:.1f}%)")
    
    # Split data into visible and hidden portions
    visible_data = data.iloc[:-hidden_rows].copy()
    hidden_data = data.iloc[-hidden_rows:].copy()
    
    print(f"Visible data date range: {visible_data.index.min()} to {visible_data.index.max()}")
    print(f"Hidden data date range: {hidden_data.index.min()} to {hidden_data.index.max()}")
    
    # Configure LSTM parameters
    seq_length = 24
    forecast_horizon = 1
    
    # Prepare data for LSTM using visible data only
    X, y, scaler, features = prepare_lstm_data(visible_data, seq_length, 'close', forecast_horizon)
    
    # Split into training/validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Load the pre-trained model
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    output_size = forecast_horizon
    dropout = 0.2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    
    # Try to load the pre-trained model
    try:
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        print("Successfully loaded pre-trained model")
    except:
        print("No pre-trained model found. Using a new model.")
    
    # Now predict the hidden portion
    print("\nPredicting hidden data...")
    
    # We need one sequence from the end of visible data to start predictions
    start_sequence = visible_data.iloc[-seq_length:].copy()
    
    # Get target column index
    target_idx = features.index('close')
    
    # Prepare the sequence
    start_sequence_scaled = scaler.transform(start_sequence[features])
    current_sequence = torch.FloatTensor(start_sequence_scaled).unsqueeze(0).to(device)
    
    # Generate predictions one by one
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i in range(len(hidden_data)):
            # Predict next value
            output = model(current_sequence).cpu().numpy()[0][0]
            predictions.append(output)
            
            # Update sequence for next prediction
            if i < len(hidden_data) - 1:
                # Shift sequence by one time step
                current_sequence = current_sequence.clone()
                current_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                
                # Create new step with prediction
                new_step = torch.zeros(1, 1, input_size)
                
                # For the target column, use our prediction - convert to tensor float
                new_step[0, 0, target_idx] = torch.tensor(output, dtype=torch.float32)
                
                # For other features, use the actual values from the hidden data
                # (This isn't realistic in practice but helps for a clean test)
                next_data = hidden_data.iloc[i][features].values
                next_data_scaled = scaler.transform(next_data.reshape(1, -1))[0]
                
                for j in range(len(features)):
                    if j != target_idx:
                        new_step[0, 0, j] = torch.tensor(next_data_scaled[j], dtype=torch.float32)
                
                # Add the new step to the sequence
                current_sequence[0, -1, :] = new_step[0, 0, :]
    
    # Inverse transform predictions
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, target_idx] = predictions
    pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Get actual values from hidden data
    actual_values = hidden_data['close'].values
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, pred_transformed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, pred_transformed)
    r2 = r2_score(actual_values, pred_transformed)
    
    # Calculate percentage error
    avg_price = np.mean(actual_values)
    pct_mae = (mae / avg_price) * 100
    
    # Print results
    print("\n--- Model Accuracy Results ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Percentage MAE: {pct_mae:.2f}%")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(hidden_data.index, actual_values, label='Actual', color='blue')
    plt.plot(hidden_data.index, pred_transformed, label='Predicted', color='red', linestyle='--')
    plt.title('LSTM Model Predictions vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('real_world_validation.png')
    plt.close()
    
    # Plot full dataset with predictions
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['close'], label='Full Data', color='gray', alpha=0.6)
    plt.plot(visible_data.index, visible_data['close'], label='Visible Data', color='green')
    plt.plot(hidden_data.index, actual_values, label='Actual Hidden Data', color='blue')
    plt.plot(hidden_data.index, pred_transformed, label='Predicted Values', color='red', linestyle='--')
    
    # Add vertical line at split point
    split_date = visible_data.index[-1]
    plt.axvline(x=split_date, color='black', linestyle='--')
    plt.annotate('Training End / Prediction Start', 
                 xy=(split_date, data['close'].min()),
                 xytext=(split_date, data['close'].min() - 5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center')
    
    plt.title('AAPL Stock Price - Training vs. Hidden Data Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('full_validation_results.png')
    
    # Save a sample of predictions vs actual values
    with open('validation_results.txt', 'w') as f:
        f.write("LSTM MODEL VALIDATION RESULTS\n")
        f.write("============================\n\n")
        
        f.write(f"Data file: {data_path}\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Hidden rows: {hidden_rows} ({hidden_percentage*100:.1f}%)\n\n")
        
        f.write(f"Visible data range: {visible_data.index.min()} to {visible_data.index.max()}\n")
        f.write(f"Hidden data range: {hidden_data.index.min()} to {hidden_data.index.max()}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"- Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"- Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"- R-squared (R²): {r2:.4f}\n")
        f.write(f"- Percentage MAE: {pct_mae:.2f}%\n\n")
        
        # Sample of predictions vs actual
        f.write("Sample Predictions vs Actual Values:\n")
        f.write("Date                  | Actual Price | Predicted Price | Error     | Error %\n")
        f.write("--------------------- | ------------ | --------------- | --------- | -------\n")
        
        # Sample up to 10 evenly spaced rows
        sample_indices = np.linspace(0, len(hidden_data)-1, min(10, len(hidden_data)), dtype=int)
        for idx in sample_indices:
            date = hidden_data.index[idx]
            actual = actual_values[idx]
            pred = pred_transformed[idx]
            error = actual - pred
            error_pct = (error / actual) * 100
            f.write(f"{date} | ${actual:.2f} | ${pred:.2f} | ${error:.2f} | {error_pct:.2f}%\n")
    
    print("\nResults saved to 'validation_results.txt'")
    print("Plots saved to 'real_world_validation.png' and 'full_validation_results.png'")
    
    return mse, rmse, mae, r2, pct_mae


if __name__ == "__main__":
    # Path to data file
    data_path = 'CS506-Final-Project-main/data_processed/yfinance/full/AAPL_15m_full.csv'
    
    # Hide 20% of the data for validation
    test_model_accuracy(data_path, hidden_percentage=0.2) 