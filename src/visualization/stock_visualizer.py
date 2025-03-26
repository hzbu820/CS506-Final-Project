import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import traceback

def visualize_with_correct_scaling():
    """
    Create a visualization of AAPL predictions with correctly scaled historical data
    """
    try:
        # Parameters
        ticker = 'AAPL'
        days_to_visualize = 90  # How many days of historical data to show
        
        print("Starting visualization process...")
        
        # Get latest prediction data
        predictions_file = "outputs/predictions/AAPL_continued_predictions.csv"
        if not os.path.exists(predictions_file):
            print(f"Predictions file {predictions_file} not found.")
            
            # Try looking for other prediction files
            prediction_files = [f for f in os.listdir("outputs/predictions") if f.endswith('.csv')]
            if prediction_files:
                print(f"Found alternative prediction files: {prediction_files}")
                predictions_file = os.path.join("outputs/predictions", prediction_files[0])
                print(f"Using alternative file: {predictions_file}")
            else:
                print("No prediction files found.")
                return
                
        # Load predictions
        print(f"Loading predictions from {predictions_file}...")
        predictions = pd.read_csv(predictions_file)
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        print(f"Loaded predictions shape: {predictions.shape}")
        print(f"First few predictions:\n{predictions.head()}")
        
        # Get historical data from yfinance (properly scaled)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_to_visualize)).strftime('%Y-%m-%d')
        
        print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
        historical_data = yf.download(ticker, start=start_date, end=end_date)
        if historical_data.empty:
            print("Warning: No historical data downloaded. Using placeholder data.")
            # Create placeholder data if download fails
            historical_data = pd.DataFrame({
                'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
                'Close': [225.0] * (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            })
        else:
            # Convert index to datetime for plotting
            print(f"Historical data shape: {historical_data.shape}")
            historical_data.reset_index(inplace=True)
            historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        print("Creating visualization...")
        # Create the visualization
        plt.figure(figsize=(14, 8))
        
        # Plot historical prices
        plt.plot(historical_data['Date'], historical_data['Close'], 
                color='blue', linewidth=2, label='Historical Price')
        
        # Add vertical line for today/prediction start
        today = datetime.now()
        plt.axvline(x=today, color='green', linestyle='--', alpha=0.7, 
                    label='Prediction Start')
        
        # Plot predicted prices
        plt.plot(predictions['Date'], predictions['Predicted_Price'], 
                color='red', linestyle='--', linewidth=2, label='Predicted Price')
        
        # Add shading for prediction period
        hist_min = float(historical_data['Close'].min())
        hist_max = float(historical_data['Close'].max())
        pred_min = float(predictions['Predicted_Price'].min())
        pred_max = float(predictions['Predicted_Price'].max())
        
        min_price = min(hist_min, pred_min) * 0.95
        max_price = max(hist_max, pred_max) * 1.05
        
        plt.fill_between(
            predictions['Date'], 
            min_price, max_price,
            color='gray', alpha=0.1
        )
        
        # Annotate key points - convert Series to float
        latest_close = float(historical_data['Close'].iloc[-1])
        latest_date = historical_data['Date'].iloc[-1]
        future_close = float(predictions['Predicted_Price'].iloc[-1])
        future_date = predictions['Date'].iloc[-1]
        
        # Calculate percent change
        percent_change = (future_close - latest_close) / latest_close * 100
        
        plt.annotate(f'${latest_close:.2f}', 
                    xy=(latest_date, latest_close),
                    xytext=(10, -30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=12)
                    
        plt.annotate(f'${future_close:.2f} ({percent_change:.2f}%)', 
                    xy=(future_date, future_close),
                    xytext=(10, 30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=12)
        
        # Format chart
        plt.title(f'{ticker} Stock Price Forecast', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price ($)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Format x-axis with dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        # Set y-axis to show the actual price range
        plt.ylim(min_price, max_price)
        
        # Save and show the plot
        output_path = f"outputs/figures/{ticker}_scaled_forecast.png"
        print(f"Saving visualization to {output_path}...")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        plt.close()
        
        print("Visualization complete.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    visualize_with_correct_scaling() 