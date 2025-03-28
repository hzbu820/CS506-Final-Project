import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import LSTMModel, prepare_lstm_data

def run_single_validation(data_path, test_size=0.2):
    """
    Run a single validation test of the LSTM model to check accuracy
    
    Args:
        data_path: Path to the data file
        test_size: Percentage of data to use for testing (0.0-1.0)
    """
    print(f"Running single validation test with {test_size*100}% test data")
    print(f"Loading data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data = data.dropna()
    
    # Split data
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Testing data: {len(test_data)} rows")
    
    # Set LSTM parameters
    seq_length = 24
    hidden_size = 64
    num_layers = 2
    
    # Prepare data
    X, y, scaler, features = prepare_lstm_data(train_data, seq_length, 'close', 1)
    
    # Get target index
    target_idx = features.index('close')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = X.shape[2]  # Number of features
    output_size = 1
    dropout = 0.2
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    
    # Load pre-trained model if available
    try:
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        print("Using pre-trained model")
    except:
        print("No pre-trained model found - using randomly initialized model")
    
    # Generate predictions
    # First create a sequence with the last part of training data
    overlap = seq_length
    last_seq = train_data.iloc[-overlap:].copy()
    combined_data = pd.concat([last_seq, test_data])
    
    # Scale the data
    X_test_full = combined_data[features].values
    X_test_scaled = scaler.transform(X_test_full)
    
    # Get the first sequence
    current_sequence = torch.FloatTensor(X_test_scaled[:seq_length]).unsqueeze(0).to(device)
    
    # Make predictions
    predictions = []
    model.eval()
    
    print(f"Generating {len(test_data)} predictions...")
    
    with torch.no_grad():
        for i in range(len(test_data)):
            # Predict next value
            output = model(current_sequence).cpu().numpy()[0][0]
            predictions.append(output)
            
            # Update sequence for next prediction (if not last step)
            if i < len(test_data) - 1:
                # Create new sequence by shifting
                current_sequence = current_sequence.clone()
                current_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                
                # Get next step from the data (except target)
                next_step = X_test_scaled[seq_length + i]
                
                # Use our prediction for target column
                next_step[target_idx] = output
                
                # Set as last step in sequence
                current_sequence[0, -1, :] = torch.FloatTensor(next_step)
    
    # Inverse transform predictions
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, target_idx] = predictions
    pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Get actual values
    actual_values = test_data['close'].values
    
    # Calculate metrics
    mse = mean_squared_error(actual_values, pred_transformed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, pred_transformed)
    r2 = r2_score(actual_values, pred_transformed)
    
    # Calculate percentage MAE
    avg_price = np.mean(actual_values)
    pct_mae = (mae / avg_price) * 100
    
    # Display results
    print("\n--- Model Performance Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Percentage MAE: {pct_mae:.2f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, actual_values, label='Actual')
    plt.plot(test_data.index, pred_transformed, label=f'Predicted (R²={r2:.4f})')
    plt.title('LSTM Model Prediction vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('single_validation_result.png')
    plt.close()
    
    # Plot relative error
    plt.figure(figsize=(12, 4))
    error_pct = np.abs(actual_values - pred_transformed) / actual_values * 100
    plt.plot(test_data.index, error_pct)
    plt.title('Prediction Error Percentage')
    plt.xlabel('Date')
    plt.ylabel('Error %')
    plt.grid(True)
    plt.savefig('single_validation_error.png')
    
    print(f"\nAverage Error: {np.mean(error_pct):.2f}%")
    print(f"Maximum Error: {np.max(error_pct):.2f}%")
    print(f"Minimum Error: {np.min(error_pct):.2f}%")
    
    return mse, rmse, mae, r2, pct_mae

if __name__ == "__main__":
    # Path to data file
    data_path = 'CS506-Final-Project-main/data_processed/yfinance/full/AAPL_15m_full.csv'
    
    # Run validation
    run_single_validation(data_path, test_size=0.2) 