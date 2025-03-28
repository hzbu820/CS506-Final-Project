import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import LSTMModel, prepare_lstm_data

def quick_model_validation(data_path, test_sizes=[0.1, 0.2, 0.3]):
    """
    Run a quick validation of the LSTM model with different test sizes
    
    Args:
        data_path: Path to the data file
        test_sizes: List of test sizes to try (as percentage of data)
    """
    print(f"Loading data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data = data.dropna()
    
    print(f"Data loaded: {len(data)} rows from {data.index.min()} to {data.index.max()}")
    
    # Results storage
    results = []
    
    # Check if model is trained
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set parameters
    seq_length = 24
    hidden_size = 64
    num_layers = 2
    
    # Try different test sizes
    for test_size in test_sizes:
        print(f"\nValidating with {test_size*100:.1f}% test data...")
        
        # Split data
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
        
        print(f"Train data: {len(train_data)} rows, Test data: {len(test_data)} rows")
        
        # Prepare data
        X, y, scaler, features = prepare_lstm_data(train_data, seq_length, 'close', 1)
        
        # Get target index
        target_idx = features.index('close')
        
        # Initialize model
        input_size = X.shape[2]  # Number of features
        output_size = 1
        dropout = 0.2
        
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
        
        # Try to load pre-trained model
        try:
            model.load_state_dict(torch.load('best_lstm_model.pth'))
            print("Using pre-trained model")
        except:
            print("No pre-trained model found")
        
        # Create a sequence from the end of train data to start predictions
        overlap = seq_length
        last_seq = train_data.iloc[-overlap:].copy()
        combined_data = pd.concat([last_seq, test_data])
        
        # Scale the data
        X_test_full = combined_data[features].values
        X_test_scaled = scaler.transform(X_test_full)
        
        # Get the first sequence
        current_sequence = torch.FloatTensor(X_test_scaled[:seq_length]).unsqueeze(0).to(device)
        
        # Generate predictions
        predictions = []
        actual_values = test_data['close'].values
        model.eval()
        
        print(f"Generating {len(test_data)} predictions...")
        
        with torch.no_grad():
            for i in range(len(test_data)):
                # Predict next step
                output = model(current_sequence).cpu().numpy()[0][0]
                predictions.append(output)
                
                # Update sequence for next prediction (if not last step)
                if i < len(test_data) - 1:
                    # Shift sequence
                    current_sequence = current_sequence.clone()
                    
                    # For auto-regressive prediction, use the prediction as next input
                    # Move all time steps forward by 1
                    current_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                    
                    # For the last step, use actual values except for the target
                    next_step = X_test_scaled[seq_length + i]
                    # Replace the target value with our prediction
                    next_step[target_idx] = output
                    
                    # Set this as the new last step in the sequence
                    current_sequence[0, -1, :] = torch.FloatTensor(next_step)
        
        # Inverse transform predictions
        dummy = np.zeros((len(predictions), len(features)))
        dummy[:, target_idx] = predictions
        pred_transformed = scaler.inverse_transform(dummy)[:, target_idx]
        
        # Calculate metrics
        mse = mean_squared_error(actual_values, pred_transformed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, pred_transformed)
        r2 = r2_score(actual_values, pred_transformed)
        
        # Calculate percentage MAE
        avg_price = np.mean(actual_values)
        pct_mae = (mae / avg_price) * 100
        
        # Display results
        print(f"Test size: {test_size*100:.1f}%")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE %: {pct_mae:.2f}%")
        
        # Store results
        results.append({
            'test_size': test_size,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pct_mae': pct_mae
        })
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, actual_values, label='Actual')
        plt.plot(test_data.index, pred_transformed, label=f'Predicted (R²={r2:.4f})')
        plt.title(f'LSTM Prediction with {test_size*100:.1f}% Test Size')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'quick_validation_test_{int(test_size*100)}pct.png')
        plt.close()
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    
    # Plot metrics vs test size
    test_sizes_pct = [r['test_size'] * 100 for r in results]
    
    plt.subplot(2, 2, 1)
    plt.plot(test_sizes_pct, [r['r2'] for r in results], 'o-', color='blue')
    plt.xlabel('Test Size (%)')
    plt.ylabel('R² Score')
    plt.title('R² Score by Test Size')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(test_sizes_pct, [r['mae'] for r in results], 'o-', color='red')
    plt.xlabel('Test Size (%)')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error by Test Size')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(test_sizes_pct, [r['pct_mae'] for r in results], 'o-', color='green')
    plt.xlabel('Test Size (%)')
    plt.ylabel('MAE %')
    plt.title('Percentage MAE by Test Size')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(test_sizes_pct, [r['rmse'] for r in results], 'o-', color='purple')
    plt.xlabel('Test Size (%)')
    plt.ylabel('RMSE')
    plt.title('RMSE by Test Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quick_validation_summary.png')
    
    # Save results to CSV
    pd.DataFrame(results).to_csv('quick_validation_results.csv', index=False)
    
    print("\nQuick validation complete. Results saved to CSV and PNG files.")
    print(f"\nAverage R² score: {np.mean([r['r2'] for r in results]):.4f}")
    print(f"Average MAE: {np.mean([r['mae'] for r in results]):.4f}")
    print(f"Average MAE %: {np.mean([r['pct_mae'] for r in results]):.2f}%")

if __name__ == "__main__":
    # Path to data file
    data_path = 'CS506-Final-Project-main/data_processed/yfinance/full/AAPL_15m_full.csv'
    
    # Run validation with different test sizes
    quick_model_validation(data_path, test_sizes=[0.1, 0.2, 0.3]) 