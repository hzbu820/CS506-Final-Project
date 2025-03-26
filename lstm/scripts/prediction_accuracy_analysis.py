import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import traceback

# Add scripts directory to the Python path
sys.path.append("scripts")

# Import intraday data visualizer functions
from scripts.intraday_data_visualizer import download_intraday_data, calculate_technical_indicators

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze LSTM prediction accuracy against actual prices')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--interval', type=str, default='5m',
                        help='Data interval (1m, 5m, 15m, 30m, 1h) (default: 5m)')
    parser.add_argument('--prediction_file', type=str, required=True,
                        help='Path to the prediction CSV file')
    parser.add_argument('--validation_period', type=str, default='1d',
                        help='Period to validate predictions (default: 1d)')
    
    return parser.parse_args()

def download_actual_data(ticker, start_date, end_date, interval):
    """Download the actual price data for comparison"""
    try:
        # Convert dates to strings if they are datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        print(f"Downloading actual data for {ticker} from {start_date} to {end_date} with {interval} interval...")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            print("Warning: No actual data downloaded. This could be due to a weekend or holiday.")
            return None
            
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        # Reset index to make Datetime a column
        data = data.reset_index()
        
        print(f"Downloaded {len(data)} data points for actual prices.")
        return data
        
    except Exception as e:
        print(f"Error downloading actual data: {e}")
        traceback.print_exc()
        return None

def load_predictions(prediction_file):
    """Load the predicted prices from a CSV file"""
    try:
        print(f"Loading predictions from {prediction_file}...")
        predictions = pd.read_csv(prediction_file)
        
        # Ensure datetime column is datetime type
        predictions['Datetime'] = pd.to_datetime(predictions['Datetime'])
        
        print(f"Loaded {len(predictions)} predictions.")
        return predictions
        
    except Exception as e:
        print(f"Error loading predictions: {e}")
        traceback.print_exc()
        return None

def match_predictions_with_actual(predictions, actual_data):
    """Match predicted prices with actual prices based on datetime"""
    try:
        # Convert datetime to string format for easier matching
        merged_data = None
        
        # If we have both datasets
        if predictions is not None and actual_data is not None:
            # Standardize datetime format
            predictions['Date_Key'] = predictions['Datetime'].dt.strftime('%Y-%m-%d %H:%M:00')
            actual_data['Date_Key'] = actual_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:00')
            
            # Merge on the datetime key
            merged_data = pd.merge(
                predictions, 
                actual_data[['Date_Key', 'Close']], 
                on='Date_Key', 
                how='left',
                suffixes=('', '_Actual')
            )
            
            # Rename columns for clarity
            merged_data = merged_data.rename(columns={
                'Predicted_Price': 'Price_Predicted',
                'Close': 'Price_Actual'
            })
            
            # Calculate prediction error
            if 'Price_Actual' in merged_data.columns:
                merged_data['Error'] = merged_data['Price_Predicted'] - merged_data['Price_Actual']
                merged_data['Error_Pct'] = (merged_data['Error'] / merged_data['Price_Actual']) * 100
                
                # Calculate predicted and actual direction
                merged_data['Direction_Predicted'] = merged_data['Price_Predicted'].diff() > 0
                merged_data['Direction_Actual'] = merged_data['Price_Actual'].diff() > 0
                
                # Calculate if direction prediction was correct
                merged_data['Direction_Correct'] = merged_data['Direction_Predicted'] == merged_data['Direction_Actual']
                
                print(f"Matched {merged_data['Price_Actual'].notna().sum()} predictions with actual data.")
                
            else:
                print("Warning: No actual prices matched with predictions.")
                
        return merged_data
        
    except Exception as e:
        print(f"Error matching predictions with actual data: {e}")
        traceback.print_exc()
        return None

def calculate_metrics(matched_data):
    """Calculate accuracy metrics for predictions"""
    metrics = {}
    
    try:
        if matched_data is None or matched_data.empty:
            print("No data available to calculate metrics.")
            return metrics
            
        # Filter out rows without actual data
        valid_data = matched_data.dropna(subset=['Price_Actual'])
        
        if valid_data.empty:
            print("No valid data with actual prices available.")
            return metrics
            
        # Calculate error metrics
        predictions = valid_data['Price_Predicted'].values
        actuals = valid_data['Price_Actual'].values
        
        metrics['MSE'] = mean_squared_error(actuals, predictions)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(actuals, predictions)
        metrics['R2'] = r2_score(actuals, predictions)
        
        # Direction accuracy
        if 'Direction_Correct' in valid_data.columns and len(valid_data) > 1:
            direction_accuracy = valid_data['Direction_Correct'].mean()
            metrics['Direction_Accuracy'] = direction_accuracy
            
        # Calculate average percentage error
        metrics['Mean_Pct_Error'] = valid_data['Error_Pct'].mean()
        metrics['Mean_Abs_Pct_Error'] = valid_data['Error_Pct'].abs().mean()
        
        # Calculate profitability potential
        # Assume a simple strategy: buy when prediction is up, sell when prediction is down
        valid_data = valid_data.copy()
        valid_data['Strategy_Return'] = 0.0
        
        for i in range(1, len(valid_data)):
            if valid_data['Direction_Predicted'].iloc[i-1]:  # If predicted up
                # Return is the actual percentage change
                valid_data.loc[valid_data.index[i], 'Strategy_Return'] = (
                    (valid_data['Price_Actual'].iloc[i] / valid_data['Price_Actual'].iloc[i-1]) - 1
                ) * 100
            else:  # If predicted down
                # Return is the negative of actual percentage change (to simulate shorting)
                valid_data.loc[valid_data.index[i], 'Strategy_Return'] = (
                    1 - (valid_data['Price_Actual'].iloc[i] / valid_data['Price_Actual'].iloc[i-1])
                ) * 100
                
        metrics['Strategy_Return'] = valid_data['Strategy_Return'].sum()
        metrics['Strategy_Win_Rate'] = (valid_data['Strategy_Return'] > 0).mean()
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        traceback.print_exc()
        return metrics

def visualize_accuracy(matched_data, metrics, ticker, interval):
    """Visualize prediction accuracy vs actual prices"""
    try:
        if matched_data is None or matched_data.empty:
            print("No data available to visualize.")
            return
            
        # Create output directory
        os.makedirs('outputs/analysis', exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Filter out rows without actual data
        valid_data = matched_data.dropna(subset=['Price_Actual'])
        
        if valid_data.empty:
            print("No valid data with actual prices available for visualization.")
            return
            
        # 1. Predictions vs Actual Prices
        plt.figure(figsize=(14, 8))
        
        plt.plot(valid_data['Datetime'], valid_data['Price_Actual'], 
                color='blue', linewidth=2, label='Actual Price')
        plt.plot(valid_data['Datetime'], valid_data['Price_Predicted'], 
                color='red', linestyle='--', marker='o', markersize=4, 
                label='Predicted Price')
        
        # Add error band
        error_std = valid_data['Error'].std()
        plt.fill_between(
            valid_data['Datetime'],
            valid_data['Price_Predicted'] - error_std,
            valid_data['Price_Predicted'] + error_std,
            color='red', alpha=0.2,
            label=f'Error Band (±1 SD: ${error_std:.2f})'
        )
        
        # Format chart
        plt.title(f'{ticker} Prediction Accuracy Analysis ({interval} intervals)', fontsize=16)
        plt.xlabel('Date/Time', fontsize=14)
        plt.ylabel('Price ($)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Format x-axis with dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Add metrics annotation
        metrics_text = "\n".join([
            f"RMSE: ${metrics.get('RMSE', 0):.3f}",
            f"MAE: ${metrics.get('MAE', 0):.3f}",
            f"Mean % Error: {metrics.get('Mean_Pct_Error', 0):.2f}%",
            f"Direction Accuracy: {metrics.get('Direction_Accuracy', 0)*100:.1f}%",
            f"Strategy Return: {metrics.get('Strategy_Return', 0):.2f}%",
            f"Win Rate: {metrics.get('Strategy_Win_Rate', 0)*100:.1f}%"
        ])
        
        # Place the metrics box in the upper right corner
        plt.annotate(
            metrics_text,
            xy=(0.97, 0.97),
            xycoords='axes fraction',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
            fontsize=10
        )
        
        # Save figure
        accuracy_plot = f"outputs/analysis/{ticker}_accuracy_analysis_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(accuracy_plot)
        plt.close()
        
        # 2. Error distribution
        plt.figure(figsize=(14, 6))
        
        plt.hist(valid_data['Error'], bins=20, alpha=0.7, color='blue')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.axvline(x=valid_data['Error'].mean(), color='green', linestyle='-', linewidth=1,
                 label=f'Mean Error: ${valid_data["Error"].mean():.3f}')
        
        plt.title(f'{ticker} Prediction Error Distribution', fontsize=16)
        plt.xlabel('Prediction Error ($)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        error_plot = f"outputs/analysis/{ticker}_error_distribution_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(error_plot)
        plt.close()
        
        # 3. Direction accuracy over time
        plt.figure(figsize=(14, 6))
        
        # Create a rolling window of direction accuracy
        window_size = min(5, len(valid_data) - 1)
        valid_data['Direction_Accuracy_Rolling'] = valid_data['Direction_Correct'].rolling(window=window_size).mean()
        
        # Plot direction accuracy
        plt.plot(valid_data['Datetime'][window_size-1:], valid_data['Direction_Accuracy_Rolling'][window_size-1:], 
                color='blue', linewidth=2)
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random Guess (50%)')
        plt.axhline(y=metrics.get('Direction_Accuracy', 0), color='green', linestyle='-', linewidth=1,
                 label=f'Average Accuracy: {metrics.get("Direction_Accuracy", 0)*100:.1f}%')
        
        plt.title(f'{ticker} Direction Prediction Accuracy Over Time', fontsize=16)
        plt.xlabel('Date/Time', fontsize=14)
        plt.ylabel('Direction Accuracy (Rolling Average)', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis with dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Save figure
        direction_plot = f"outputs/analysis/{ticker}_direction_accuracy_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(direction_plot)
        plt.close()
        
        print(f"Visualizations saved to:")
        print(f"- {accuracy_plot}")
        print(f"- {error_plot}")
        print(f"- {direction_plot}")
        
        # Save matched data to CSV
        matched_data_file = f"outputs/analysis/{ticker}_accuracy_data_{timestamp}.csv"
        valid_data.to_csv(matched_data_file, index=False)
        print(f"- Accuracy data: {matched_data_file}")
        
    except Exception as e:
        print(f"Error visualizing accuracy: {e}")
        traceback.print_exc()

def print_accuracy_summary(metrics, ticker):
    """Print a summary of prediction accuracy metrics"""
    print("\n" + "=" * 60)
    print(f"PREDICTION ACCURACY ANALYSIS FOR {ticker}")
    print("=" * 60)
    
    if not metrics:
        print("No metrics available. Analysis could not be completed.")
        return
        
    print(f"Error Metrics:")
    print(f"- Root Mean Squared Error (RMSE): ${metrics.get('RMSE', 0):.3f}")
    print(f"- Mean Absolute Error (MAE): ${metrics.get('MAE', 0):.3f}")
    print(f"- R-squared (R²): {metrics.get('R2', 0):.3f}")
    print(f"- Mean Percentage Error: {metrics.get('Mean_Pct_Error', 0):.2f}%")
    print(f"- Mean Absolute Percentage Error: {metrics.get('Mean_Abs_Pct_Error', 0):.2f}%")
    
    print("\nDirection Prediction:")
    print(f"- Direction Accuracy: {metrics.get('Direction_Accuracy', 0)*100:.1f}%")
    
    print("\nTrading Strategy Simulation:")
    print(f"- Cumulative Return: {metrics.get('Strategy_Return', 0):.2f}%")
    print(f"- Win Rate: {metrics.get('Strategy_Win_Rate', 0)*100:.1f}%")
    
    # Determine the quality assessment
    quality_threshold = {
        'excellent': {'RMSE': 0.5, 'Direction_Accuracy': 0.65, 'Strategy_Return': 2.0},
        'good': {'RMSE': 1.0, 'Direction_Accuracy': 0.55, 'Strategy_Return': 0.5},
        'fair': {'RMSE': 2.0, 'Direction_Accuracy': 0.5, 'Strategy_Return': 0},
        'poor': {'RMSE': float('inf'), 'Direction_Accuracy': 0, 'Strategy_Return': float('-inf')}
    }
    
    quality = 'poor'
    for level in ['excellent', 'good', 'fair', 'poor']:
        thresholds = quality_threshold[level]
        if (metrics.get('RMSE', float('inf')) <= thresholds['RMSE'] and 
            metrics.get('Direction_Accuracy', 0) >= thresholds['Direction_Accuracy'] and
            metrics.get('Strategy_Return', float('-inf')) >= thresholds['Strategy_Return']):
            quality = level
            break
    
    print("\nOverall Assessment:")
    print(f"Based on the metrics, the prediction quality is {quality.upper()}.")
    
    print("=" * 60)
    
    # Provide insights
    print("\nInsights:")
    
    if metrics.get('Direction_Accuracy', 0) > 0.5:
        print("• The model is better than random guessing at predicting price direction.")
    else:
        print("• The model is not reliable for predicting price direction (worse than random).")
        
    if metrics.get('Strategy_Return', 0) > 0:
        print("• The simulated trading strategy would have been profitable during this period.")
    else:
        print("• The simulated trading strategy would have lost money during this period.")
        
    if metrics.get('Mean_Abs_Pct_Error', 0) < 1.0:
        print("• The model has high precision in predicting the exact price values.")
    elif metrics.get('Mean_Abs_Pct_Error', 0) < 2.0:
        print("• The model has reasonable precision in predicting price values.")
    else:
        print("• The model has significant errors in predicting exact price values.")
        
    print(f"• On average, predictions are off by ${metrics.get('MAE', 0):.2f} from the actual price.")
    
    # Recommendation based on quality
    print("\nRecommendation:")
    if quality in ['excellent', 'good']:
        print("This model demonstrates good predictive power and could be considered for real trading,")
        print("but should be used in conjunction with other analysis methods and proper risk management.")
    elif quality == 'fair':
        print("This model shows some predictive ability but needs improvement before being relied upon.")
        print("Consider using it as one of multiple signals in a trading system.")
    else:
        print("This model does not show sufficient predictive power for trading decisions.")
        print("Further training, feature engineering, or model architecture changes are recommended.")
    
    print("=" * 60)

def main():
    """Main function to analyze prediction accuracy"""
    args = parse_arguments()
    
    try:
        # Load predictions
        predictions = load_predictions(args.prediction_file)
        if predictions is None:
            print("Could not load predictions. Exiting.")
            return
            
        # Determine date range for actual data
        min_date = predictions['Datetime'].min()
        max_date = predictions['Datetime'].max()
        print(f"Prediction period: {min_date} to {max_date}")
        
        # Download actual data for the prediction period
        # Add a buffer to ensure we get enough data
        buffer_days = 2  # Add 2 days buffer
        start_date = min_date - timedelta(days=buffer_days)
        end_date = max_date + timedelta(days=buffer_days)
        
        actual_data = download_actual_data(args.ticker, start_date, end_date, args.interval)
        if actual_data is None:
            print("Could not download actual data. Exiting.")
            return
            
        # Match predictions with actual data
        matched_data = match_predictions_with_actual(predictions, actual_data)
        if matched_data is None or matched_data['Price_Actual'].notna().sum() == 0:
            print("Could not match predictions with actual data. Exiting.")
            return
            
        # Calculate accuracy metrics
        metrics = calculate_metrics(matched_data)
        
        # Visualize accuracy
        visualize_accuracy(matched_data, metrics, args.ticker, args.interval)
        
        # Print accuracy summary
        print_accuracy_summary(metrics, args.ticker)
        
    except Exception as e:
        print(f"Error in prediction accuracy analysis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 