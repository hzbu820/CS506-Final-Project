import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from real_world_validation import test_model_accuracy

# Run model validation with different configuration parameters
# to assess consistency and robustness

def run_multiple_validations():
    """Run the model validation multiple times with different parameters"""
    print("Running multiple LSTM model validations...")

    # Data path
    data_path = 'CS506-Final-Project-main/data_processed/yfinance/full/AAPL_15m_full.csv'
    
    # Results storage
    all_results = []
    
    # Test 1: Different hidden data percentages
    print("\n=== TEST 1: Different hidden data percentages ===")
    hidden_percentages = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    for pct in hidden_percentages:
        print(f"\nValidating with {pct*100:.1f}% hidden data...")
        mse, rmse, mae, r2, pct_mae = test_model_accuracy(
            data_path=data_path,
            hidden_percentage=pct
        )
        
        all_results.append({
            'test_type': 'hidden_percentage',
            'parameter': pct,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pct_mae': pct_mae
        })
        
    # Test 2: Different time periods (using different start points)
    print("\n=== TEST 2: Different time periods ===")
    
    # Load the data
    data = pd.read_csv(data_path)
    date_col = 'datetime' if 'datetime' in data.columns else 'date'
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data = data.dropna()
    
    # Create temporary files with different start points
    total_rows = len(data)
    test_periods = [0.0, 0.2, 0.4, 0.6]  # Start at different points in the data
    
    for period_start in test_periods:
        # Calculate start row
        start_row = int(total_rows * period_start)
        # Make sure we have enough data
        if start_row > total_rows - 300:
            continue
            
        # Create subset of data
        subset_data = data.iloc[start_row:].copy()
        temp_file = f'temp_data_start_{period_start}.csv'
        subset_data.to_csv(temp_file)
        
        print(f"\nValidating time period starting at {subset_data.index[0]}...")
        mse, rmse, mae, r2, pct_mae = test_model_accuracy(
            data_path=temp_file,
            hidden_percentage=0.2
        )
        
        all_results.append({
            'test_type': 'time_period',
            'parameter': str(subset_data.index[0]),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pct_mae': pct_mae
        })
    
    # Visualize the results
    print("\n=== Results Summary ===")
    
    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(all_results)
    
    # Summary statistics
    print("\nOverall Statistics:")
    print(f"Average R² score: {results_df['r2'].mean():.4f}")
    print(f"R² standard deviation: {results_df['r2'].std():.4f}")
    print(f"Average MAE: {results_df['mae'].mean():.4f}")
    print(f"Average MAE %: {results_df['pct_mae'].mean():.2f}%")
    
    # Plot R² scores for different hidden percentages
    plt.figure(figsize=(12, 8))
    
    # Plot 1: R² vs hidden percentage
    plt.subplot(2, 2, 1)
    hidden_pct_df = results_df[results_df['test_type'] == 'hidden_percentage']
    plt.plot(hidden_pct_df['parameter'] * 100, hidden_pct_df['r2'], 'o-', color='blue')
    plt.xlabel('Hidden Data Percentage (%)')
    plt.ylabel('R² Score')
    plt.title('R² Score vs. Hidden Data Percentage')
    plt.grid(True)
    
    # Plot 2: MAE vs hidden percentage
    plt.subplot(2, 2, 2)
    plt.plot(hidden_pct_df['parameter'] * 100, hidden_pct_df['mae'], 'o-', color='red')
    plt.xlabel('Hidden Data Percentage (%)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE vs. Hidden Data Percentage')
    plt.grid(True)
    
    # Plot 3: R² for different time periods
    plt.subplot(2, 2, 3)
    time_period_df = results_df[results_df['test_type'] == 'time_period']
    x_values = range(len(time_period_df))
    plt.bar(x_values, time_period_df['r2'], color='green')
    plt.xticks(x_values, [p.split()[0] for p in time_period_df['parameter']], rotation=45)
    plt.xlabel('Starting Date')
    plt.ylabel('R² Score')
    plt.title('R² Score for Different Time Periods')
    plt.grid(True)
    
    # Plot 4: MAE for different time periods
    plt.subplot(2, 2, 4)
    plt.bar(x_values, time_period_df['mae'], color='orange')
    plt.xticks(x_values, [p.split()[0] for p in time_period_df['parameter']], rotation=45)
    plt.xlabel('Starting Date')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE for Different Time Periods')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multi_validation_results.png')
    
    # Save all results to CSV
    results_df.to_csv('multi_validation_results.csv', index=False)
    
    print("\nMultiple validation complete. Results saved to:")
    print("- multi_validation_results.png")
    print("- multi_validation_results.csv")

if __name__ == "__main__":
    run_multiple_validations() 