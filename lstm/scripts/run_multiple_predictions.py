import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from datetime import datetime
import glob
import argparse

# Add src directory to Python path
sys.path.append("src")

# Import predictor
from models.stock_predictor import StockPredictor

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run multiple stock predictions with different seeds')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--runs', type=int, default=50,
                        help='Number of prediction runs to perform (default: 50)')
    parser.add_argument('--future_days', type=int, default=14,
                        help='Number of days to predict into the future (default: 14)')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Sequence length for LSTM (default: 30)')
    parser.add_argument('--hidden_size_min', type=int, default=128,
                        help='Minimum hidden size for LSTM (default: 128)')
    parser.add_argument('--hidden_size_max', type=int, default=256,
                        help='Maximum hidden size for LSTM (default: 256)')
    parser.add_argument('--num_layers_min', type=int, default=2,
                        help='Minimum number of LSTM layers (default: 2)')
    parser.add_argument('--num_layers_max', type=int, default=3,
                        help='Maximum number of LSTM layers (default: 3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--use_existing', action='store_true',
                        help='Use existing models instead of training new ones')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    return parser.parse_args()

def train_and_predict(ticker, run_id, seed, args, hidden_size, num_layers):
    """Train a model and make predictions with specified parameters"""
    try:
        # Set random seed
        set_random_seed(seed)
        
        # Create output directories
        os.makedirs('outputs/ensemble_runs', exist_ok=True)
        
        # Initialize predictor
        predictor = StockPredictor(
            ticker=ticker,
            sequence_length=args.sequence_length
        )
        
        if args.use_existing:
            # Find existing model
            existing_model = predictor.find_latest_model()
            if existing_model:
                print(f"Run {run_id}: Using existing model {existing_model}")
                predictor.load_model(existing_model)
            else:
                print(f"Run {run_id}: No existing model found, training new model")
                # Train model
                history = predictor.train_model(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    epochs=args.epochs
                )
        else:
            # Train model
            print(f"Run {run_id}: Training model (hidden_size={hidden_size}, num_layers={num_layers}, seed={seed})")
            history = predictor.train_model(
                hidden_size=hidden_size,
                num_layers=num_layers,
                epochs=args.epochs
            )
        
        # Predict future prices
        print(f"Run {run_id}: Predicting future prices for {args.future_days} days")
        future_predictions = predictor.predict_future(days=args.future_days)
        
        # Save predictions to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'outputs/ensemble_runs/{ticker}_run_{run_id}_{timestamp}.csv'
        future_predictions.to_csv(output_path, index=False)
        
        print(f"Run {run_id}: Predictions saved to {output_path}")
        
        # Get model info
        # Ensure final_price is a scalar, not a Series
        final_price = future_predictions['Predicted_Price'].iloc[-1]
        if hasattr(final_price, 'item'):  # Check if it's a numpy value or tensor
            final_price = final_price.item()
        else:
            final_price = float(final_price)  # Ensure it's a float
            
        model_info = {
            'run_id': run_id,
            'seed': seed,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'prediction_file': output_path,
            'final_price': final_price,
            'success': True
        }
        
        return model_info
        
    except Exception as e:
        import traceback
        print(f"Error in run {run_id}: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return {
            'run_id': run_id,
            'seed': seed,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'success': False,
            'error': str(e)
        }

def analyze_predictions(ticker, runs_info):
    """Analyze the predictions from multiple runs"""
    print("\nAnalyzing predictions from multiple runs...")
    
    # Create output directories
    os.makedirs('outputs/ensemble_analysis', exist_ok=True)
    
    successful_runs = [run for run in runs_info if run['success']]
    print(f"Successfully completed {len(successful_runs)} out of {len(runs_info)} runs")
    
    if len(successful_runs) == 0:
        print("No successful runs to analyze")
        # Try to use existing files in outputs/ensemble_runs
        print("Trying to use existing files from outputs/ensemble_runs")
        prediction_files = glob.glob(f'outputs/ensemble_runs/{ticker}_run_*.csv')
        
        if not prediction_files:
            # Also check outputs/predictions
            prediction_files = glob.glob(f'outputs/predictions/{ticker}_*.csv')
            
        if not prediction_files:
            print("No prediction files found, nothing to analyze")
            return
            
        print(f"Found {len(prediction_files)} existing prediction files to analyze")
        
        # Create dummy successful_runs from files
        for i, file in enumerate(prediction_files):
            filename = os.path.basename(file)
            try:
                # Try to extract run_id from filename
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == 'run':
                    run_id = int(parts[2])
                else:
                    run_id = i + 1
                    
                # Create dummy run info
                successful_runs.append({
                    'run_id': run_id,
                    'seed': run_id + 42,
                    'hidden_size': 128,  # Default
                    'num_layers': 2,     # Default
                    'prediction_file': file,
                    'success': True
                })
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        if len(successful_runs) == 0:
            print("Still no valid runs to analyze")
            return
    
    # Load all predictions
    all_predictions = []
    for run in successful_runs:
        try:
            pred_df = pd.read_csv(run['prediction_file'])
            
            # Check for Date or Datetime column
            if 'Date' in pred_df.columns:
                pred_df['Date'] = pd.to_datetime(pred_df['Date'])
                # Standardize to timezone-naive
                if hasattr(pred_df['Date'].dt, 'tz') and pred_df['Date'].dt.tz is not None:
                    pred_df['Date'] = pred_df['Date'].dt.tz_localize(None)
            elif 'Datetime' in pred_df.columns:
                pred_df.rename(columns={'Datetime': 'Date'}, inplace=True)
                pred_df['Date'] = pd.to_datetime(pred_df['Date'])
                # Standardize to timezone-naive
                if hasattr(pred_df['Date'].dt, 'tz') and pred_df['Date'].dt.tz is not None:
                    pred_df['Date'] = pred_df['Date'].dt.tz_localize(None)
            else:
                print(f"No date column found in {run['prediction_file']}, skipping")
                continue
                
            # Check for predicted price column
            if 'Predicted_Price' not in pred_df.columns:
                # Try to find price column
                price_cols = [c for c in pred_df.columns if 'price' in c.lower() or 'pred' in c.lower()]
                if price_cols:
                    pred_df.rename(columns={price_cols[0]: 'Predicted_Price'}, inplace=True)
                else:
                    print(f"No price column found in {run['prediction_file']}, skipping")
                    continue
                
            pred_df['run_id'] = run['run_id']
            pred_df['hidden_size'] = run.get('hidden_size', 128)  # Default if not present
            pred_df['num_layers'] = run.get('num_layers', 2)     # Default if not present
            all_predictions.append(pred_df)
        except Exception as e:
            print(f"Error loading predictions for run {run['run_id']}: {e}")
    
    # Combine all predictions
    if not all_predictions:
        print("Could not load any prediction files")
        return
        
    combined_df = pd.concat(all_predictions)
    
    # Ensure all dates are timezone-naive for consistent grouping
    if hasattr(combined_df['Date'].dt, 'tz') and combined_df['Date'].dt.tz is not None:
        combined_df['Date'] = combined_df['Date'].dt.tz_localize(None)
    
    # Group by date and calculate statistics
    stats_df = combined_df.groupby('Date')['Predicted_Price'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Calculate confidence intervals (95%)
    stats_df['ci_lower'] = stats_df['mean'] - 1.96 * stats_df['std']
    stats_df['ci_upper'] = stats_df['mean'] + 1.96 * stats_df['std']
    
    # Calculate coefficient of variation (CV) to measure relative variability
    stats_df['cv'] = stats_df['std'] / stats_df['mean'] * 100
    
    # Save statistics to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stats_path = f'outputs/ensemble_analysis/{ticker}_ensemble_stats_{timestamp}.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Ensemble statistics saved to {stats_path}")
    
    # Calculate overall statistics for final prices
    # Extract final price from each run safely
    final_prices = []
    for run in successful_runs:
        if 'final_price' in run and run['final_price'] is not None:
            # Use final_price from run info if available
            final_prices.append(run['final_price'])
        else:
            # Try to extract from prediction file
            try:
                pred_df = next((df for df in all_predictions if df['run_id'].iloc[0] == run['run_id']), None)
                if pred_df is not None:
                    final_price = pred_df['Predicted_Price'].iloc[-1]
                    if hasattr(final_price, 'item'):  # Check if it's a numpy value or tensor
                        final_price = final_price.item()
                    else:
                        final_price = float(final_price)
                    final_prices.append(final_price)
            except Exception as e:
                print(f"Could not extract final price for run {run['run_id']}: {e}")
    
    if not final_prices:
        print("No final prices available for analysis")
        return
    
    mean_final = np.mean(final_prices)
    median_final = np.median(final_prices)
    std_final = np.std(final_prices)
    min_final = np.min(final_prices)
    max_final = np.max(final_prices)
    
    # Calculate 95% confidence interval for final price
    ci_lower = mean_final - 1.96 * std_final
    ci_upper = mean_final + 1.96 * std_final
    
    # Create visualizations
    # 1. Line plot with confidence intervals
    plt.figure(figsize=(14, 10))
    
    # First subplot: All predictions
    plt.subplot(2, 1, 1)
    
    # Plot individual predictions with low opacity
    for pred_df in all_predictions:
        plt.plot(pred_df['Date'], pred_df['Predicted_Price'], 'b-', alpha=0.2)
    
    # Plot mean and confidence intervals
    plt.plot(stats_df['Date'], stats_df['mean'], 'r-', linewidth=2, label='Mean Prediction')
    plt.fill_between(stats_df['Date'], stats_df['ci_lower'], stats_df['ci_upper'], 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.title(f'{ticker} Price Predictions - {len(successful_runs)} Model Runs', fontsize=16)
    plt.ylabel('Predicted Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Second subplot: Variability
    plt.subplot(2, 1, 2)
    plt.plot(stats_df['Date'], stats_df['cv'], 'g-', linewidth=2)
    plt.title('Prediction Variability (Coefficient of Variation)', fontsize=16)
    plt.ylabel('CV (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'outputs/ensemble_analysis/{ticker}_ensemble_plot_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Ensemble plot saved to {plot_path}")
    
    # 2. Histogram of final predictions
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=15, alpha=0.7, color='blue')
    plt.axvline(mean_final, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_final:.2f}')
    plt.axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label=f'95% CI: ${ci_lower:.2f} - ${ci_upper:.2f}')
    plt.axvline(ci_upper, color='orange', linestyle='--', linewidth=2)
    
    plt.title(f'{ticker} Final Price Distribution - {len(final_prices)} Model Runs', fontsize=16)
    plt.xlabel('Predicted Final Price ($)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save histogram
    hist_path = f'outputs/ensemble_analysis/{ticker}_final_price_hist_{timestamp}.png'
    plt.savefig(hist_path)
    plt.close()
    print(f"Final price histogram saved to {hist_path}")
    
    # Print summary statistics
    print("\nFinal Price Statistics:")
    print(f"Mean: ${mean_final:.2f}")
    print(f"Median: ${median_final:.2f}")
    print(f"Standard Deviation: ${std_final:.2f}")
    print(f"Range: ${min_final:.2f} - ${max_final:.2f}")
    print(f"95% Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}")
    
    # Calculate trend information - use a more robust approach
    # Calculate average first day price
    first_day_prices = []
    for pred_df in all_predictions:
        try:
            first_price = pred_df['Predicted_Price'].iloc[0]
            if hasattr(first_price, 'item'):
                first_price = first_price.item()
            else:
                first_price = float(first_price)
            first_day_prices.append(first_price)
        except Exception as e:
            print(f"Error extracting first day price: {e}")
    
    if not first_day_prices:
        print("Could not calculate trend information")
        up_percentage = 50  # Default neutral
    else:
        avg_first_price = np.mean(first_day_prices)
        up_count = sum(1 for price in final_prices if price > avg_first_price)
        total_count = len(final_prices)
        up_percentage = (up_count / total_count) * 100
        
    down_percentage = 100 - up_percentage
        
    print("\nTrend Analysis:")
    print(f"Models predicting UP: {up_count} ({up_percentage:.1f}%)")
    print(f"Models predicting DOWN: {total_count - up_count} ({down_percentage:.1f}%)")
    
    if up_percentage > 70:
        print("Strong consensus: UP")
    elif up_percentage < 30:
        print("Strong consensus: DOWN")
    else:
        print("Mixed consensus, no clear direction")
    
    # Calculate average price path and final price
    avg_change_pct = ((mean_final / np.mean(first_day_prices)) - 1) * 100 if first_day_prices else 0
    
    print(f"\nAverage predicted change: {avg_change_pct:.2f}%")
    
    # Export summary to text file
    summary_path = f'outputs/ensemble_analysis/{ticker}_ensemble_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write(f"{ticker} Ensemble Analysis Summary\n")
        f.write(f"==============================\n")
        f.write(f"Number of successful runs: {len(successful_runs)}\n")
        f.write(f"Date of analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Final Price Statistics:\n")
        f.write(f"Mean: ${mean_final:.2f}\n")
        f.write(f"Median: ${median_final:.2f}\n")
        f.write(f"Standard Deviation: ${std_final:.2f}\n")
        f.write(f"Range: ${min_final:.2f} - ${max_final:.2f}\n")
        f.write(f"95% Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}\n\n")
        
        f.write(f"Trend Analysis:\n")
        f.write(f"Models predicting UP: {up_count} ({up_percentage:.1f}%)\n")
        f.write(f"Models predicting DOWN: {total_count - up_count} ({down_percentage:.1f}%)\n")
        f.write(f"Average predicted change: {avg_change_pct:.2f}%\n\n")
        
        if up_percentage > 70:
            f.write("Strong consensus: UP\n")
        elif up_percentage < 30:
            f.write("Strong consensus: DOWN\n")
        else:
            f.write("Mixed consensus, no clear direction\n")
            
        f.write("\nVariability Analysis:\n")
        f.write(f"Average Coefficient of Variation: {stats_df['cv'].mean():.2f}%\n")
        f.write(f"Maximum Coefficient of Variation: {stats_df['cv'].max():.2f}%\n")
        f.write(f"Minimum Coefficient of Variation: {stats_df['cv'].min():.2f}%\n\n")
        
        # Use get() to safely access potentially missing keys
        hidden_sizes = [run.get('hidden_size', 128) for run in successful_runs]
        num_layers = [run.get('num_layers', 2) for run in successful_runs]
        
        f.write("Model Parameters Distribution:\n")
        f.write(f"Hidden Sizes: {sorted(set(hidden_sizes))}\n")
        f.write(f"Number of Layers: {sorted(set(num_layers))}\n")
    
    print(f"Ensemble summary saved to {summary_path}")
    
    # Also create a direction ensemble
    try:
        print("\nCreating direction ensemble analysis...")
        import importlib.util
        spec = importlib.util.spec_from_file_location("ensemble_from_multiple_runs", "scripts/ensemble_from_multiple_runs.py")
        ensemble_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble_module)
        
        # Call create_direction_ensemble with our timestamp
        ensemble_module.create_direction_ensemble(ticker, timestamp, 0.6, debug=False)
    except Exception as e:
        print(f"Error creating direction ensemble: {e}")
    
    return stats_df, all_predictions

def main():
    """Main function to run multiple predictions and analyze results"""
    args = parse_arguments()
    ticker = args.ticker
    
    print(f"Running {args.runs} predictions for {ticker}")
    print(f"Future days: {args.future_days}")
    
    # Create output directories
    os.makedirs('outputs/ensemble_runs', exist_ok=True)
    os.makedirs('outputs/ensemble_analysis', exist_ok=True)
    
    # Run predictions
    runs_info = []
    
    for run_id in range(1, args.runs + 1):
        # Generate seed for this run
        seed = run_id + 42
        
        # Set random hyperparameters within specified ranges
        hidden_size = random.randint(args.hidden_size_min, args.hidden_size_max)
        num_layers = random.randint(args.num_layers_min, args.num_layers_max)
        
        # Train and predict
        run_info = train_and_predict(ticker, run_id, seed, args, hidden_size, num_layers)
        runs_info.append(run_info)
        
        # Print progress
        print(f"Completed {run_id}/{args.runs} runs")
    
    # Analyze results
    analyze_predictions(ticker, runs_info)
    
    # Final summary
    successful_runs = [run for run in runs_info if run['success']]
    print(f"\nCompleted {args.runs} runs with {len(successful_runs)} successful runs")
    
    # Print command to run ensemble analysis separately
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\nTo create a direction ensemble from these runs, run:")
    print(f"python scripts/ensemble_from_multiple_runs.py --ticker {ticker} --timestamp {timestamp}")

if __name__ == "__main__":
    main() 