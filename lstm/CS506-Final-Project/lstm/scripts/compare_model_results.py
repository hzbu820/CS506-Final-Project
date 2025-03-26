import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import os

def compare_predictions():
    """
    Compare the predictions from the short-term model with the previous model
    """
    # Load prediction files
    short_term_file = "outputs/predictions/AAPL_short_term_20250325_173342.csv"
    detailed_file = "outputs/predictions/AAPL_detailed_forecast.csv"
    
    if not os.path.exists(short_term_file) or not os.path.exists(detailed_file):
        print("Error: Prediction files not found.")
        return
    
    # Load prediction data
    short_term = pd.read_csv(short_term_file)
    detailed = pd.read_csv(detailed_file)
    
    # Convert dates to datetime
    short_term['Date'] = pd.to_datetime(short_term['Date'])
    detailed['Date'] = pd.to_datetime(detailed['Date'])
    
    # Fetch current AAPL price
    print("Fetching current AAPL price...")
    aapl_data = yf.download('AAPL', period='5d')
    current_price = float(aapl_data['Close'].iloc[-1])
    print(f"Current price: ${current_price:.2f}")
    
    # Create comparison data
    comparison = pd.DataFrame({
        'Date': short_term['Date'],
        'Short_Term_Model': short_term['Predicted_Price'],
        'Original_Model': detailed['Original_Price']
    })
    
    # Calculate key metrics
    short_term_final = float(short_term['Predicted_Price'].iloc[-1])
    detailed_final = float(detailed['Original_Price'].iloc[-1])
    
    short_term_change = (short_term_final - current_price) / current_price * 100
    detailed_change = (detailed_final - current_price) / current_price * 100
    
    short_term_direction = "UP" if short_term_change > 0 else "DOWN"
    detailed_direction = "UP" if detailed_change > 0 else "DOWN"
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print("=" * 50)
    print(f"Current AAPL Price: ${current_price:.2f}")
    print("-" * 50)
    print(f"Short-Term Model (14-day sequence length):")
    print(f"  Final Price: ${short_term_final:.2f}")
    print(f"  Change: {short_term_change:.2f}%")
    print(f"  Direction: {short_term_direction}")
    print(f"  Model Architecture: 128 hidden units, 2 layers")
    print(f"  Training Loss: 0.0127, Validation Loss: 0.0558")
    print("-" * 50)
    print(f"Original Enhanced Model (30-day sequence length):")
    print(f"  Final Price: ${detailed_final:.2f}")
    print(f"  Change: {detailed_change:.2f}%")
    print(f"  Direction: {detailed_direction}")
    print(f"  Model Architecture: 256 hidden units, 3 layers")
    print("-" * 50)
    print(f"Difference between models: ${detailed_final - short_term_final:.2f} ({detailed_change - short_term_change:.2f}%)")
    
    # Create comparison visualization
    plt.figure(figsize=(14, 8))
    
    # Plot both predictions
    plt.plot(comparison['Date'], comparison['Short_Term_Model'], 
             color='red', linestyle='-', linewidth=2, marker='o', 
             markersize=5, label='Short-Term Model (14-day)')
    
    plt.plot(comparison['Date'], comparison['Original_Model'], 
             color='blue', linestyle='-', linewidth=2, marker='s', 
             markersize=5, label='Original Enhanced Model (30-day)')
    
    # Add horizontal line for current price
    plt.axhline(y=current_price, color='green', linestyle='--', alpha=0.7,
                label=f'Current Price (${current_price:.2f})')
    
    # Add annotations for final predictions
    plt.annotate(f'${short_term_final:.2f} ({short_term_change:.2f}%)',
                xy=(comparison['Date'].iloc[-1], short_term_final),
                xytext=(15, -10),
                textcoords='offset points',
                color='red',
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.annotate(f'${detailed_final:.2f} ({detailed_change:.2f}%)',
                xy=(comparison['Date'].iloc[-1], detailed_final),
                xytext=(15, 10),
                textcoords='offset points',
                color='blue',
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    # Format chart
    plt.title('AAPL Prediction Model Comparison', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    # Set y-axis range with padding
    min_price = min(comparison['Short_Term_Model'].min(), comparison['Original_Model'].min(), current_price) * 0.995
    max_price = max(comparison['Short_Term_Model'].max(), comparison['Original_Model'].max(), current_price) * 1.005
    plt.ylim(min_price, max_price)
    
    # Add text box with key differences
    plt.figtext(0.15, 0.02, 
               f"Key Differences:\n"
               f"• Short-Term Model: 14-day sequence, 128 hidden units, 2 layers\n"
               f"• Original Model: 30-day sequence, 256 hidden units, 3 layers\n"
               f"• Final Price Difference: ${detailed_final - short_term_final:.2f} ({detailed_change - short_term_change:.2f}%)",
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
               fontsize=10)
    
    # Save visualization
    output_path = "outputs/figures/model_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nComparison visualization saved to {output_path}")
    
    # Generate daily comparison table
    print("\nDaily Price Comparison:")
    print("-" * 65)
    print(f"{'Date':<12} {'Short-Term':<12} {'Enhanced':<12} {'Diff':<8} {'Diff %':<8}")
    print("-" * 65)
    
    for i, row in comparison.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        short_term_price = row['Short_Term_Model']
        detailed_price = row['Original_Model']
        diff = detailed_price - short_term_price
        diff_pct = (diff / short_term_price) * 100
        print(f"{date:<12} ${short_term_price:<10.2f} ${detailed_price:<10.2f} ${diff:<6.2f} {diff_pct:<6.2f}%")
    
    # Save comparison data
    comparison.to_csv("outputs/predictions/model_comparison.csv", index=False)
    print("\nComparison data saved to outputs/predictions/model_comparison.csv")
    
if __name__ == "__main__":
    compare_predictions() 