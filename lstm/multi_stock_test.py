import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import LSTMModel, prepare_lstm_data

def test_multiple_stocks(stock_symbols=['AAPL', 'AMZN', 'GOOG', 'META'], interval='15m', test_size=0.2):
    """
    Test the LSTM model's performance across multiple stocks
    
    Args:
        stock_symbols: List of stock symbols to test
        interval: Data interval to use (15m, 1h, 1d, etc.)
        test_size: Portion of data to use for testing
    """
    print(f"Testing LSTM model on multiple stocks with {interval} data")
    
    # Store results
    results = []
    
    # Test on each stock
    for symbol in stock_symbols:
        print(f"\n=== Testing on {symbol} ===")
        
        # Construct file path
        file_path = f'CS506-Final-Project-main/data_processed/yfinance/full/{symbol}_{interval}_full.csv'
        
        try:
            # Load data
            data = pd.read_csv(file_path)
            date_col = 'datetime' if 'datetime' in data.columns else 'date'
            data[date_col] = pd.to_datetime(data[date_col])
            data.set_index(date_col, inplace=True)
            data = data.dropna()
            
            print(f"Loaded {len(data)} rows from {data.index.min()} to {data.index.max()}")
            
            # Split data
            train_size = int(len(data) * (1 - test_size))
            train_data = data.iloc[:train_size].copy()
            test_data = data.iloc[train_size:].copy()
            
            # Parameters
            seq_length = 24
            hidden_size = 64
            num_layers = 2
            
            # Prepare data
            X, y, scaler, features = prepare_lstm_data(train_data, seq_length, 'close', 1)
            
            # Get target index
            target_idx = features.index('close')
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                continue  # Skip if no model
            
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
            
            # Store results
            results.append({
                'symbol': symbol,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pct_mae': pct_mae,
                'avg_price': avg_price
            })
            
            # Plot predictions
            plt.figure(figsize=(12, 6))
            plt.plot(test_data.index, actual_values, label='Actual')
            plt.plot(test_data.index, pred_transformed, label=f'Predicted (R²={r2:.4f})')
            plt.title(f'{symbol} Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{symbol}_prediction.png')
            plt.close()
            
            # Display results
            print(f"{symbol} Results:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"MAE %: {pct_mae:.2f}%")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Create summary of results
    if results:
        results_df = pd.DataFrame(results)
        
        # Print summary
        print("\n=== Summary of Results ===")
        for symbol in stock_symbols:
            symbol_results = results_df[results_df['symbol'] == symbol]
            if not symbol_results.empty:
                print(f"{symbol}: R² = {symbol_results['r2'].values[0]:.4f}, MAE % = {symbol_results['pct_mae'].values[0]:.2f}%")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: R² by stock
        plt.subplot(2, 2, 1)
        plt.bar(results_df['symbol'], results_df['r2'])
        plt.title('R² by Stock')
        plt.xlabel('Stock Symbol')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)  # R² normally between 0 and 1
        
        # Plot 2: MAE% by stock
        plt.subplot(2, 2, 2)
        plt.bar(results_df['symbol'], results_df['pct_mae'])
        plt.title('Percentage MAE by Stock')
        plt.xlabel('Stock Symbol')
        plt.ylabel('MAE %')
        
        # Plot 3: RMSE by stock
        plt.subplot(2, 2, 3)
        plt.bar(results_df['symbol'], results_df['rmse'])
        plt.title('RMSE by Stock')
        plt.xlabel('Stock Symbol')
        plt.ylabel('RMSE')
        
        # Plot 4: Normalized RMSE by stock (RMSE / Avg Price)
        plt.subplot(2, 2, 4)
        normalized_rmse = results_df['rmse'] / results_df['avg_price'] * 100
        plt.bar(results_df['symbol'], normalized_rmse)
        plt.title('Normalized RMSE by Stock (% of Avg Price)')
        plt.xlabel('Stock Symbol')
        plt.ylabel('RMSE %')
        
        plt.tight_layout()
        plt.savefig('multi_stock_results.png')
        
        # Save results to CSV
        results_df.to_csv('multi_stock_results.csv', index=False)
        
        print("\nMulti-stock testing complete. Results saved to CSV and PNG files.")
        
    else:
        print("No results to report.")

if __name__ == "__main__":
    # Test on multiple stocks with 15-minute data
    test_multiple_stocks(stock_symbols=['AAPL', 'AMZN', 'GOOG', 'META'], interval='15m', test_size=0.2) 