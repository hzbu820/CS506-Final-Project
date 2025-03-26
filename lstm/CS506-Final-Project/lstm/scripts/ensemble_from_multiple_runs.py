import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
from datetime import datetime, timezone
import yfinance as yf

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create ensemble prediction from multiple model runs')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp to identify specific runs (optional)')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                        help='Confidence threshold for trading signals (default: 0.6)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    return parser.parse_args()

def get_current_price(ticker):
    """Get the current price of a stock"""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        return current_price
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
        return None

def create_direction_ensemble(ticker, timestamp=None, confidence_threshold=0.6, debug=False):
    """Create a direction-based ensemble from multiple model runs"""
    # Create output directories
    os.makedirs('outputs/ensemble_analysis', exist_ok=True)
    
    # Get prediction files
    if timestamp:
        prediction_files = glob.glob(f'outputs/ensemble_runs/{ticker}_run_*_{timestamp}.csv')
        # If no files found with timestamp, try without requiring timestamp
        if not prediction_files:
            print(f"No files found with timestamp {timestamp}, using all available files")
            prediction_files = glob.glob(f'outputs/ensemble_runs/{ticker}_run_*.csv')
    else:
        prediction_files = glob.glob(f'outputs/ensemble_runs/{ticker}_run_*.csv')
    
    # If still no files, check predictions directory
    if not prediction_files:
        print("No files found in ensemble_runs directory, checking predictions directory")
        prediction_files = glob.glob(f'outputs/predictions/{ticker}_*.csv')
    
    if not prediction_files:
        print(f"No prediction files found for {ticker}" + 
              (f" with timestamp {timestamp}" if timestamp else ""))
        return
    
    print(f"Found {len(prediction_files)} prediction files for {ticker}")
    
    # Get current price
    current_price = get_current_price(ticker)
    if current_price is None:
        print("Could not fetch current price, using first prediction as reference")
    else:
        print(f"Current price of {ticker}: ${current_price:.2f}")
    
    # Load all predictions
    all_predictions = []
    
    for file_path in prediction_files:
        try:
            # Extract run ID from filename
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            if len(parts) >= 3 and parts[1] == 'run':
                run_id = int(parts[2])
            else:
                # If not a standard run file, use index as run_id
                run_id = len(all_predictions) + 1
            
            # Load predictions
            pred_df = pd.read_csv(file_path)
            
            # Ensure Date column is datetime
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
                print(f"No date column found in {file_path}, skipping")
                continue
            
            # Ensure there's a Predicted_Price column
            if 'Predicted_Price' not in pred_df.columns:
                # Try to find price column
                price_cols = [c for c in pred_df.columns if 'price' in c.lower() or 'pred' in c.lower()]
                if price_cols:
                    pred_df.rename(columns={price_cols[0]: 'Predicted_Price'}, inplace=True)
                else:
                    print(f"No price column found in {file_path}, skipping")
                    continue
            
            # Add run ID
            pred_df['run_id'] = run_id
            
            all_predictions.append(pred_df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_predictions:
        print("No valid prediction files loaded")
        return
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions)
    
    # Get all unique dates
    all_dates = sorted(combined_df['Date'].unique())
    
    # Initialize ensemble DataFrame with dates
    ensemble_df = pd.DataFrame({'Date': all_dates})
    
    # Initialize direction counts
    for date in all_dates:
        # Get all predictions for this date
        date_preds = combined_df[combined_df['Date'] == date]
        
        # Reference price - either the previous day's predictions or current price for first day
        if date == all_dates[0]:
            if current_price is not None:
                reference_price = current_price
            else:
                # Use the mean of the first day predictions
                reference_price = date_preds['Predicted_Price'].mean()
        else:
            # Get the previous day
            prev_date = all_dates[all_dates.index(date) - 1]
            # Get mean prediction for previous day
            prev_preds = combined_df[combined_df['Date'] == prev_date]
            reference_price = prev_preds['Predicted_Price'].mean()
        
        # Count up/down predictions
        up_votes = (date_preds['Predicted_Price'] > reference_price).sum()
        down_votes = (date_preds['Predicted_Price'] < reference_price).sum()
        same_votes = (date_preds['Predicted_Price'] == reference_price).sum()
        
        # Calculate average predicted price
        mean_price = date_preds['Predicted_Price'].mean()
        median_price = date_preds['Predicted_Price'].median()
        std_dev = date_preds['Predicted_Price'].std()
        
        # Calculate confidence
        total_votes = up_votes + down_votes + same_votes
        if total_votes > 0:
            up_confidence = up_votes / total_votes
            down_confidence = down_votes / total_votes
            same_confidence = same_votes / total_votes
        else:
            up_confidence = down_confidence = same_confidence = 0
        
        # Determine direction
        if up_votes > down_votes:
            direction = 'UP'
            confidence = up_confidence
        elif down_votes > up_votes:
            direction = 'DOWN'
            confidence = down_confidence
        else:
            direction = 'NEUTRAL'
            confidence = same_confidence
        
        # Calculate 95% confidence interval
        ci_lower = mean_price - 1.96 * std_dev
        ci_upper = mean_price + 1.96 * std_dev
        
        # Add to ensemble DataFrame
        ensemble_df.loc[ensemble_df['Date'] == date, 'Mean_Price'] = mean_price
        ensemble_df.loc[ensemble_df['Date'] == date, 'Median_Price'] = median_price
        ensemble_df.loc[ensemble_df['Date'] == date, 'Std_Dev'] = std_dev
        ensemble_df.loc[ensemble_df['Date'] == date, 'CI_Lower'] = ci_lower
        ensemble_df.loc[ensemble_df['Date'] == date, 'CI_Upper'] = ci_upper
        ensemble_df.loc[ensemble_df['Date'] == date, 'Up_Votes'] = up_votes
        ensemble_df.loc[ensemble_df['Date'] == date, 'Down_Votes'] = down_votes
        ensemble_df.loc[ensemble_df['Date'] == date, 'Same_Votes'] = same_votes
        ensemble_df.loc[ensemble_df['Date'] == date, 'Total_Votes'] = total_votes
        ensemble_df.loc[ensemble_df['Date'] == date, 'Direction'] = direction
        ensemble_df.loc[ensemble_df['Date'] == date, 'Confidence'] = confidence
    
    # Calculate daily returns
    ensemble_df['Pct_Change'] = ensemble_df['Mean_Price'].pct_change()
    ensemble_df.loc[0, 'Pct_Change'] = (ensemble_df.loc[0, 'Mean_Price'] / current_price - 1) if current_price else 0
    
    # Calculate cumulative returns
    if current_price is not None:
        ensemble_df['Cumulative_Return'] = ensemble_df['Mean_Price'] / current_price - 1
    else:
        first_price = ensemble_df.loc[0, 'Mean_Price']
        ensemble_df['Cumulative_Return'] = ensemble_df['Mean_Price'] / first_price - 1
    
    # Generate trading signals
    ensemble_df['Signal'] = 'HOLD'
    ensemble_df.loc[(ensemble_df['Direction'] == 'UP') & (ensemble_df['Confidence'] >= confidence_threshold), 'Signal'] = 'BUY'
    ensemble_df.loc[(ensemble_df['Direction'] == 'DOWN') & (ensemble_df['Confidence'] >= confidence_threshold), 'Signal'] = 'SELL'
    
    # Create visualizations
    create_ensemble_visualizations(ticker, ensemble_df, current_price)
    
    # Save ensemble DataFrame
    curr_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ensemble_path = f'outputs/ensemble_analysis/{ticker}_direction_ensemble_{curr_timestamp}.csv'
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"Direction ensemble saved to {ensemble_path}")
    
    # Print summary statistics
    final_day = ensemble_df.iloc[-1]
    first_day = ensemble_df.iloc[0]
    
    print("\nEnsemble Direction Prediction Summary:")
    print(f"Time period: {first_day['Date'].strftime('%Y-%m-%d')} to {final_day['Date'].strftime('%Y-%m-%d')}")
    print(f"Number of models: {len(prediction_files)}")
    
    if current_price is not None:
        print(f"Current price: ${current_price:.2f}")
    
    print(f"Final mean predicted price: ${final_day['Mean_Price']:.2f}")
    print(f"Final prediction range: ${final_day['CI_Lower']:.2f} - ${final_day['CI_Upper']:.2f}")
    
    if current_price is not None:
        print(f"Predicted change: {final_day['Cumulative_Return']*100:.2f}%")
    
    # Count days for each direction
    direction_counts = ensemble_df['Direction'].value_counts()
    print("\nDirection Distribution:")
    for direction, count in direction_counts.items():
        print(f"{direction}: {count} days ({count/len(ensemble_df)*100:.1f}%)")
    
    # Count signals
    signal_counts = ensemble_df['Signal'].value_counts()
    print("\nTrading Signal Distribution:")
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} days ({count/len(ensemble_df)*100:.1f}%)")
    
    # Overall recommendation
    up_days = ensemble_df[ensemble_df['Direction'] == 'UP'].shape[0]
    down_days = ensemble_df[ensemble_df['Direction'] == 'DOWN'].shape[0]
    high_confidence_up = ensemble_df[(ensemble_df['Direction'] == 'UP') & (ensemble_df['Confidence'] >= confidence_threshold)].shape[0]
    high_confidence_down = ensemble_df[(ensemble_df['Direction'] == 'DOWN') & (ensemble_df['Confidence'] >= confidence_threshold)].shape[0]
    
    print("\nOverall Recommendation:")
    
    if high_confidence_up > high_confidence_down and final_day['Direction'] == 'UP' and final_day['Confidence'] >= confidence_threshold:
        print("STRONG BUY - Consistent upward trend with high confidence")
    elif high_confidence_down > high_confidence_up and final_day['Direction'] == 'DOWN' and final_day['Confidence'] >= confidence_threshold:
        print("STRONG SELL - Consistent downward trend with high confidence")
    elif up_days > down_days and final_day['Direction'] == 'UP':
        print("BUY - Moderately bullish outlook")
    elif down_days > up_days and final_day['Direction'] == 'DOWN':
        print("SELL - Moderately bearish outlook")
    else:
        print("HOLD - Mixed or neutral signals")
    
    print(f"\nNOTE: Analysis based on {len(prediction_files)} model runs")

def create_ensemble_visualizations(ticker, ensemble_df, current_price):
    """Create visualizations for the ensemble analysis"""
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Price prediction with confidence bands
    plt.figure(figsize=(14, 10))
    
    # First subplot: Price prediction
    plt.subplot(2, 1, 1)
    
    # Plot mean price
    plt.plot(ensemble_df['Date'], ensemble_df['Mean_Price'], 'b-', linewidth=2, label='Mean Prediction')
    
    # Plot confidence intervals
    plt.fill_between(ensemble_df['Date'], ensemble_df['CI_Lower'], ensemble_df['CI_Upper'], 
                     color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # Add current price line
    if current_price is not None:
        plt.axhline(y=current_price, color='black', linestyle='--', alpha=0.5, 
                   label=f'Current Price: ${current_price:.2f}')
    
    # Add markers for trading signals
    buy_signals = ensemble_df[ensemble_df['Signal'] == 'BUY']
    sell_signals = ensemble_df[ensemble_df['Signal'] == 'SELL']
    
    if not buy_signals.empty:
        plt.scatter(buy_signals['Date'], buy_signals['Mean_Price'], 
                   marker='^', color='green', s=100, label='BUY Signal')
    
    if not sell_signals.empty:
        plt.scatter(sell_signals['Date'], sell_signals['Mean_Price'], 
                   marker='v', color='red', s=100, label='SELL Signal')
    
    plt.title(f'{ticker} Ensemble Price Prediction', fontsize=16)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Second subplot: Direction confidence
    plt.subplot(2, 1, 2)
    
    # Create a bar for each day
    for i, row in ensemble_df.iterrows():
        if row['Direction'] == 'UP':
            color = 'green'
        elif row['Direction'] == 'DOWN':
            color = 'red'
        else:
            color = 'gray'
            
        plt.bar(row['Date'], row['Confidence'], color=color, alpha=0.7)
    
    # Add threshold line
    plt.axhline(y=0.6, color='black', linestyle='--', alpha=0.5, label='Confidence Threshold (0.6)')
    
    plt.title('Direction Prediction Confidence', fontsize=16)
    plt.ylabel('Confidence', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    price_path = f'outputs/ensemble_analysis/{ticker}_ensemble_price_{timestamp}.png'
    plt.savefig(price_path)
    plt.close()
    print(f"Ensemble price visualization saved to {price_path}")
    
    # 2. Vote distribution chart
    plt.figure(figsize=(14, 8))
    
    # Create stacked bar chart of votes
    plt.bar(ensemble_df['Date'], ensemble_df['Up_Votes'], color='green', alpha=0.7, label='Up Votes')
    plt.bar(ensemble_df['Date'], ensemble_df['Down_Votes'], bottom=ensemble_df['Up_Votes'], 
           color='red', alpha=0.7, label='Down Votes')
    
    if (ensemble_df['Same_Votes'] > 0).any():
        plt.bar(ensemble_df['Date'], ensemble_df['Same_Votes'], 
               bottom=ensemble_df['Up_Votes'] + ensemble_df['Down_Votes'],
               color='gray', alpha=0.7, label='Neutral Votes')
    
    plt.title(f'{ticker} Direction Vote Distribution', fontsize=16)
    plt.ylabel('Number of Votes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    votes_path = f'outputs/ensemble_analysis/{ticker}_vote_distribution_{timestamp}.png'
    plt.savefig(votes_path)
    plt.close()
    print(f"Vote distribution visualization saved to {votes_path}")
    
    # 3. Cumulative return chart
    plt.figure(figsize=(14, 8))
    
    # Plot cumulative return
    plt.plot(ensemble_df['Date'], ensemble_df['Cumulative_Return'] * 100, 'b-', linewidth=2)
    
    # Shade based on direction
    for i in range(len(ensemble_df) - 1):
        if ensemble_df.iloc[i]['Direction'] == 'UP':
            color = 'green'
            alpha = min(ensemble_df.iloc[i]['Confidence'] + 0.2, 0.5)
        elif ensemble_df.iloc[i]['Direction'] == 'DOWN':
            color = 'red'
            alpha = min(ensemble_df.iloc[i]['Confidence'] + 0.2, 0.5)
        else:
            color = 'gray'
            alpha = 0.3
            
        plt.axvspan(ensemble_df.iloc[i]['Date'], ensemble_df.iloc[i+1]['Date'], 
                   color=color, alpha=alpha)
    
    plt.title(f'{ticker} Predicted Cumulative Return', fontsize=16)
    plt.ylabel('Return (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    return_path = f'outputs/ensemble_analysis/{ticker}_cumulative_return_{timestamp}.png'
    plt.savefig(return_path)
    plt.close()
    print(f"Cumulative return visualization saved to {return_path}")

def main():
    """Main function"""
    args = parse_arguments()
    create_direction_ensemble(
        args.ticker, 
        args.timestamp, 
        args.confidence_threshold,
        args.debug
    )

if __name__ == "__main__":
    main() 