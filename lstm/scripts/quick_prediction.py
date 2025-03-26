"""
Quick Prediction Script - Uses existing models to make predictions without heavy training
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
import glob

# Add src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple dependency checking
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')  # Use non-interactive backend
    import yfinance as yf
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Quick Stock Price Prediction')
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL, MSFT)')
    
    parser.add_argument('--future_days', type=int, default=14,
                        help='Number of days to predict into the future (default: 14)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed error messages')
    
    return parser.parse_args()

def get_current_price(ticker):
    """
    Get the current price for a ticker
    """
    try:
        data = yf.download(ticker, period="1d")
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def load_existing_predictions(ticker, debug=False):
    """
    Load existing prediction files
    """
    prediction_files = glob.glob(f"outputs/predictions/{ticker}_*.csv")
    
    if not prediction_files:
        print(f"No prediction files found for {ticker}")
        return None
    
    if debug:
        print(f"Found {len(prediction_files)} prediction files:")
        for f in prediction_files:
            print(f"  - {f}")
    
    all_predictions = []
    
    for file_path in prediction_files:
        try:
            # Skip files that are too old (e.g., modified more than 30 days ago)
            if os.path.getmtime(file_path) < (datetime.now().timestamp() - 30 * 86400):
                if debug:
                    print(f"Skipping old file: {file_path}")
                continue
                
            # Load the prediction
            pred_df = pd.read_csv(file_path)
            
            # Ensure the prediction has the necessary columns
            if 'Date' in pred_df.columns or 'Datetime' in pred_df.columns:
                # Standardize date column name
                if 'Date' in pred_df.columns:
                    pred_df['Datetime'] = pd.to_datetime(pred_df['Date'])
                else:
                    pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'])
                
                # Ensure it has predicted price column
                if 'Predicted_Price' in pred_df.columns:
                    model_name = os.path.basename(file_path).replace('.csv', '')
                    
                    if debug:
                        print(f"Loaded prediction from {model_name} with {len(pred_df)} rows")
                    
                    all_predictions.append((pred_df, model_name))
                    
        except Exception as e:
            if debug:
                print(f"Error loading prediction from {file_path}: {e}")
    
    if not all_predictions:
        print("No valid prediction files could be loaded")
        return None
        
    print(f"Successfully loaded {len(all_predictions)} prediction sets")
    return all_predictions

def normalize_predictions(predictions, current_price, debug=False):
    """
    Normalize predictions to align with current price
    """
    normalized_predictions = []
    
    for pred_df, model_name in predictions:
        try:
            # Make a copy to avoid SettingWithCopyWarning
            df_copy = pred_df.copy()
            
            # Get the first predicted price
            first_pred = df_copy['Predicted_Price'].iloc[0]
            
            # Calculate scaling factor
            price_diff = abs((first_pred - current_price) / current_price)
            
            if price_diff > 0.1:  # More than 10% different from current price
                if debug:
                    print(f"Normalizing {model_name}: First prediction ${first_pred:.2f} vs current ${current_price:.2f}")
                
                # Apply scaling correction
                scaling_factor = current_price / first_pred
                df_copy.loc[:, 'Predicted_Price'] = df_copy['Predicted_Price'] * scaling_factor
                
                if debug:
                    print(f"  After normalization: ${df_copy['Predicted_Price'].iloc[0]:.2f}")
            
            normalized_predictions.append((df_copy, model_name))
            
        except Exception as e:
            if debug:
                print(f"Error normalizing predictions from {model_name}: {e}")
    
    return normalized_predictions

def ensemble_predictions(normalized_predictions, future_days=14, debug=False):
    """
    Create an ensemble prediction from multiple models
    """
    if not normalized_predictions:
        return None
    
    try:
        # Get the earliest and latest dates across all predictions
        all_dates = set()
        for pred_df, _ in normalized_predictions:
            all_dates.update(pred_df['Datetime'].tolist())
        
        all_dates = sorted(list(all_dates))
        
        if debug:
            print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
            print(f"Total prediction days available: {len(all_dates)}")
        
        # Create date range for the ensemble
        ensemble_dates = all_dates[:min(future_days, len(all_dates))]
        
        if debug:
            print(f"Using dates for ensemble: {len(ensemble_dates)} days")
        
        # Initialize ensemble DataFrame
        ensemble_df = pd.DataFrame({
            'Datetime': ensemble_dates,
            'Mean_Price': 0.0,
            'Model_Count': 0
        })
        
        # Process each prediction set
        for pred_df, model_name in normalized_predictions:
            for date in ensemble_dates:
                # Find matching date in the prediction
                matching_rows = pred_df[pred_df['Datetime'] == date]
                
                if not matching_rows.empty:
                    # Update ensemble with this prediction
                    date_idx = ensemble_df.index[ensemble_df['Datetime'] == date].tolist()[0]
                    
                    # Get the predicted price
                    pred_price = matching_rows['Predicted_Price'].values[0]
                    
                    # Update weighted average
                    current_count = ensemble_df.loc[date_idx, 'Model_Count']
                    current_price = ensemble_df.loc[date_idx, 'Mean_Price']
                    
                    if current_count == 0:
                        ensemble_df.loc[date_idx, 'Mean_Price'] = pred_price
                    else:
                        ensemble_df.loc[date_idx, 'Mean_Price'] = (
                            (current_price * current_count + pred_price) / (current_count + 1)
                        )
                    
                    # Increment model count
                    ensemble_df.loc[date_idx, 'Model_Count'] += 1
        
        return ensemble_df
        
    except Exception as e:
        if debug:
            print(f"Error creating ensemble: {e}")
            traceback.print_exc()
        return None

def main():
    """
    Main function for quick stock price prediction
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("=" * 50)
    print(f"Quick Stock Price Prediction for {args.ticker}")
    print("=" * 50)
    
    try:
        # Get current price
        print(f"Getting current price for {args.ticker}...")
        current_price = get_current_price(args.ticker)
        
        if current_price is None:
            print("Could not retrieve current price. Exiting.")
            return 1
            
        print(f"Current price: ${current_price:.2f}")
        
        # Load existing predictions
        print("\nLoading existing predictions...")
        predictions = load_existing_predictions(args.ticker, args.debug)
        
        if not predictions:
            print("No usable predictions found. Exiting.")
            return 1
        
        # Normalize predictions
        print("\nNormalizing predictions to current price...")
        normalized_predictions = normalize_predictions(predictions, current_price, args.debug)
        
        # Create ensemble
        print("\nCreating ensemble prediction...")
        ensemble_df = ensemble_predictions(normalized_predictions, args.future_days, args.debug)
        
        if ensemble_df is None or ensemble_df.empty:
            print("Could not create ensemble prediction. Exiting.")
            return 1
        
        # Save ensemble prediction
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_file = f"outputs/ensemble/{args.ticker}_quick_ensemble_{timestamp}.csv"
        
        # Create directory if it doesn't exist
        os.makedirs('outputs/ensemble', exist_ok=True)
        
        # Save to CSV
        ensemble_df.to_csv(ensemble_file, index=False)
        print(f"Ensemble prediction saved to {ensemble_file}")
        
        # Print summary
        print("\nEnsemble Prediction Summary:")
        if len(ensemble_df) > 0:
            first_price = ensemble_df['Mean_Price'].iloc[0]
            last_price = ensemble_df['Mean_Price'].iloc[-1]
            change_pct = ((last_price - first_price) / first_price) * 100
            
            print(f"Predicted price in {len(ensemble_df)} days: ${last_price:.2f}")
            print(f"Predicted change from now: {change_pct:.2f}%")
            
            # Direction analysis
            up_days = (ensemble_df['Mean_Price'].diff() > 0).sum()
            down_days = (ensemble_df['Mean_Price'].diff() < 0).sum()
            neutral_days = len(ensemble_df) - 1 - up_days - down_days
            
            print("\nDirection Analysis:")
            print(f"Up days: {up_days} ({up_days/(len(ensemble_df)-1)*100:.1f}%)")
            print(f"Down days: {down_days} ({down_days/(len(ensemble_df)-1)*100:.1f}%)")
            print(f"Neutral days: {neutral_days} ({neutral_days/(len(ensemble_df)-1)*100:.1f}%)")
            
            # Trading signal
            if change_pct > 5:
                signal = "BUY - Strong upward trend predicted"
            elif change_pct > 2:
                signal = "BUY - Moderate upward trend predicted"
            elif change_pct > 0:
                signal = "HOLD/BUY - Slight upward trend predicted"
            elif change_pct > -2:
                signal = "HOLD/NEUTRAL - Minimal change predicted"
            elif change_pct > -5:
                signal = "HOLD/SELL - Moderate downward trend predicted"
            else:
                signal = "SELL - Strong downward trend predicted"
                
            print(f"\nTrading Signal: {signal}")
        
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            print("\nDetailed traceback:")
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 