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
    parser = argparse.ArgumentParser(description='Create normalized ensemble prediction from multiple models')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
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

def create_normalized_ensemble(ticker, confidence_threshold=0.6, debug=False):
    """Create a normalized ensemble from multiple model predictions"""
    # Create output directories
    os.makedirs('outputs/normalized_ensemble', exist_ok=True)
    
    # Get all prediction files for this ticker
    prediction_files = glob.glob(f'outputs/predictions/{ticker}_*.csv')
    
    # Also check ensemble_runs directory
    ensemble_files = glob.glob(f'outputs/ensemble_runs/{ticker}_run_*.csv')
    prediction_files.extend(ensemble_files)
    
    if not prediction_files:
        print(f"No prediction files found for {ticker}")
        return
    
    print(f"Found {len(prediction_files)} prediction files for {ticker}")
    
    # Get current price
    current_price = get_current_price(ticker)
    if current_price is None:
        print("Could not fetch current price. Using most recent known price.")
        try:
            # Try to get recent price from historical data
            recent_data = yf.download(ticker, period="1d")
            current_price = recent_data['Close'].iloc[-1]
            print(f"Using recent price: ${current_price:.2f}")
        except:
            print("Failed to get recent price. Using $100 as reference price.")
            current_price = 100.0
    else:
        print(f"Current price of {ticker}: ${current_price:.2f}")
    
    # Load all prediction files
    all_predictions = []
    
    for file_path in prediction_files:
        try:
            # Get model name from filename
            model_name = os.path.basename(file_path).replace('.csv', '')
            
            # Load predictions
            pred_df = pd.read_csv(file_path)
            
            # Ensure we have a date column
            date_column = None
            for col in pred_df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_column = col
                    break
            
            if date_column is None:
                print(f"No date column found in {file_path}, skipping")
                continue
            
            # Standardize date column name
            pred_df.rename(columns={date_column: 'Date'}, inplace=True)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            
            # Standardize to timezone-naive
            if hasattr(pred_df['Date'].dt, 'tz') and pred_df['Date'].dt.tz is not None:
                pred_df['Date'] = pred_df['Date'].dt.tz_localize(None)
            
            # Find price column
            price_column = None
            for col in pred_df.columns:
                if 'price' in col.lower() or 'pred' in col.lower() or 'forecast' in col.lower():
                    price_column = col
                    break
            
            if price_column is None:
                print(f"No price column found in {file_path}, skipping")
                continue
            
            # Standardize price column name
            pred_df.rename(columns={price_column: 'Predicted_Price'}, inplace=True)
            
            # Normalize prices to current price
            first_price = pred_df['Predicted_Price'].iloc[0]
            scaling_factor = current_price / first_price
            
            # Apply normalization
            pred_df['Original_Price'] = pred_df['Predicted_Price']
            pred_df['Predicted_Price'] = pred_df['Predicted_Price'] * scaling_factor
            
            # Add model name
            pred_df['Model'] = model_name
            
            # Compute daily returns
            pred_df['Daily_Return'] = pred_df['Predicted_Price'].pct_change()
            pred_df.loc[0, 'Daily_Return'] = (pred_df.loc[0, 'Predicted_Price'] / current_price) - 1
            
            # Compute cumulative returns
            pred_df['Cumulative_Return'] = (pred_df['Predicted_Price'] / current_price) - 1
            
            if debug:
                print(f"Model: {model_name}")
                print(f"First price: ${first_price:.2f}, Scaling factor: {scaling_factor:.4f}")
                print(f"Normalized first price: ${pred_df['Predicted_Price'].iloc[0]:.2f}")
                print(f"Final price: ${pred_df['Predicted_Price'].iloc[-1]:.2f}")
                print(f"Final return: {pred_df['Cumulative_Return'].iloc[-1]*100:.2f}%")
                print()
            
            all_predictions.append(pred_df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_predictions:
        print("No valid predictions could be loaded")
        return
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions)
    
    # Get all unique dates
    all_dates = sorted(combined_df['Date'].unique())
    
    # Create a date range for analysis
    date_df = pd.DataFrame({'Date': all_dates})
    
    # Calculate statistics for each date
    results = []
    
    for date in all_dates:
        # Get all predictions for this date
        date_preds = combined_df[combined_df['Date'] == date]
        
        # Calculate statistics
        mean_price = date_preds['Predicted_Price'].mean()
        median_price = date_preds['Predicted_Price'].median()
        std_price = date_preds['Predicted_Price'].std()
        min_price = date_preds['Predicted_Price'].min()
        max_price = date_preds['Predicted_Price'].max()
        count = len(date_preds)
        
        # Calculate 95% confidence interval
        ci_lower = mean_price - 1.96 * std_price
        ci_upper = mean_price + 1.96 * std_price
        
        # Calculate direction votes
        if date == all_dates[0]:
            reference_price = current_price
        else:
            # Use previous day's mean price
            prev_date_idx = all_dates.index(date) - 1
            prev_date = all_dates[prev_date_idx]
            prev_result = next((r for r in results if r['Date'] == prev_date), None)
            reference_price = prev_result['Mean_Price'] if prev_result else current_price
        
        up_votes = (date_preds['Predicted_Price'] > reference_price).sum()
        down_votes = (date_preds['Predicted_Price'] < reference_price).sum()
        neutral_votes = (date_preds['Predicted_Price'] == reference_price).sum()
        
        # Calculate direction confidence
        total_votes = up_votes + down_votes + neutral_votes
        if total_votes > 0:
            if up_votes > down_votes:
                direction = 'UP'
                confidence = up_votes / total_votes
            elif down_votes > up_votes:
                direction = 'DOWN'
                confidence = down_votes / total_votes
            else:
                direction = 'NEUTRAL'
                confidence = neutral_votes / total_votes
        else:
            direction = 'NEUTRAL'
            confidence = 0.0
        
        # Calculate trading signal
        if direction == 'UP' and confidence >= confidence_threshold:
            signal = 'BUY'
        elif direction == 'DOWN' and confidence >= confidence_threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Calculate return metrics
        daily_return = (mean_price / reference_price) - 1
        cumulative_return = (mean_price / current_price) - 1
        
        # Add to results
        results.append({
            'Date': date,
            'Mean_Price': mean_price,
            'Median_Price': median_price,
            'Std_Dev': std_price,
            'Min_Price': min_price,
            'Max_Price': max_price,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Model_Count': count,
            'Up_Votes': up_votes,
            'Down_Votes': down_votes,
            'Neutral_Votes': neutral_votes,
            'Direction': direction,
            'Confidence': confidence,
            'Signal': signal,
            'Daily_Return': daily_return,
            'Cumulative_Return': cumulative_return
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization: Price Prediction with Confidence Bands
    plt.figure(figsize=(14, 10))
    
    # First subplot: Price prediction
    plt.subplot(2, 1, 1)
    
    # Plot price prediction
    plt.plot(results_df['Date'], results_df['Mean_Price'], 'b-', linewidth=2, label='Mean Prediction')
    
    # Plot confidence band
    plt.fill_between(results_df['Date'], results_df['CI_Lower'], results_df['CI_Upper'], 
                    color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # Add current price line
    plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.5, 
               label=f'Current Price: ${current_price:.2f}')
    
    # Add buy/sell signals
    buy_signals = results_df[results_df['Signal'] == 'BUY']
    sell_signals = results_df[results_df['Signal'] == 'SELL']
    
    if not buy_signals.empty:
        plt.scatter(buy_signals['Date'], buy_signals['Mean_Price'], color='green', marker='^', 
                   s=100, label='BUY Signal')
        
    if not sell_signals.empty:
        plt.scatter(sell_signals['Date'], sell_signals['Mean_Price'], color='red', marker='v', 
                   s=100, label='SELL Signal')
    
    # Add model prices with low opacity for context
    for model_name in combined_df['Model'].unique():
        model_df = combined_df[combined_df['Model'] == model_name]
        plt.plot(model_df['Date'], model_df['Predicted_Price'], alpha=0.15)
    
    plt.title(f'{ticker} Normalized Ensemble Prediction', fontsize=16)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Second subplot: Direction confidence
    plt.subplot(2, 1, 2)
    
    # Plot direction confidence as bars
    for i, row in results_df.iterrows():
        if row['Direction'] == 'UP':
            color = 'green'
        elif row['Direction'] == 'DOWN':
            color = 'red'
        else:
            color = 'gray'
        
        plt.bar(row['Date'], row['Confidence'], color=color, alpha=0.7)
    
    plt.axhline(y=confidence_threshold, color='black', linestyle='--', alpha=0.5, 
               label=f'Confidence Threshold ({confidence_threshold})')
    
    plt.title('Direction Prediction Confidence', fontsize=16)
    plt.ylabel('Confidence', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = f'outputs/normalized_ensemble/{ticker}_normalized_ensemble_{timestamp}.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"Normalized ensemble visualization saved to {fig_path}")
    
    # Create visualization: Return Chart
    plt.figure(figsize=(14, 7))
    
    # Plot cumulative return
    plt.plot(results_df['Date'], results_df['Cumulative_Return'] * 100, 'b-', linewidth=2, 
            label='Cumulative Return')
    
    # Color background based on direction
    for i in range(len(results_df) - 1):
        if results_df.iloc[i]['Direction'] == 'UP':
            color = 'green'
            alpha = min(0.1 + results_df.iloc[i]['Confidence'] * 0.3, 0.4)
        elif results_df.iloc[i]['Direction'] == 'DOWN':
            color = 'red'
            alpha = min(0.1 + results_df.iloc[i]['Confidence'] * 0.3, 0.4)
        else:
            color = 'gray'
            alpha = 0.1
        
        plt.axvspan(results_df.iloc[i]['Date'], results_df.iloc[i+1]['Date'], 
                   color=color, alpha=alpha)
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add return values
    first_return = results_df.iloc[0]['Cumulative_Return'] * 100
    last_return = results_df.iloc[-1]['Cumulative_Return'] * 100
    
    plt.annotate(f"{first_return:.2f}%", 
                xy=(results_df.iloc[0]['Date'], first_return),
                xytext=(5, 0), textcoords='offset points',
                fontsize=10)
    
    plt.annotate(f"{last_return:.2f}%", 
                xy=(results_df.iloc[-1]['Date'], last_return),
                xytext=(-40, 0), textcoords='offset points',
                fontsize=10)
    
    plt.title(f'{ticker} Predicted Cumulative Return', fontsize=16)
    plt.ylabel('Return (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    return_path = f'outputs/normalized_ensemble/{ticker}_return_chart_{timestamp}.png'
    plt.savefig(return_path)
    plt.close()
    print(f"Return chart saved to {return_path}")
    
    # Create visualization: Individual Models
    plt.figure(figsize=(14, 7))
    
    # Plot each model's normalized prediction
    for model_name in combined_df['Model'].unique():
        model_df = combined_df[combined_df['Model'] == model_name]
        
        # Calculate final return
        final_return = model_df['Cumulative_Return'].iloc[-1] * 100
        
        # Determine if it's UP or DOWN overall
        direction = 'UP' if final_return > 0 else 'DOWN'
        
        # Format model name to be shorter
        display_name = model_name.split('_')[-2:]
        display_name = '_'.join(display_name) if len(display_name) > 1 else model_name.split('_')[-1]
        
        # Create label with return
        label = f"{display_name}: {direction} ({final_return:.2f}%)"
        
        plt.plot(model_df['Date'], model_df['Predicted_Price'], linewidth=1.5, label=label)
    
    # Add current price line
    plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.5, 
               label=f'Current Price: ${current_price:.2f}')
    
    plt.title(f'{ticker} Individual Model Predictions (Normalized)', fontsize=16)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    models_path = f'outputs/normalized_ensemble/{ticker}_model_comparison_{timestamp}.png'
    plt.savefig(models_path)
    plt.close()
    print(f"Model comparison saved to {models_path}")
    
    # Save results to CSV
    results_path = f'outputs/normalized_ensemble/{ticker}_normalized_ensemble_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Normalized ensemble results saved to {results_path}")
    
    # Save individual model predictions
    for model_name in combined_df['Model'].unique():
        model_df = combined_df[combined_df['Model'] == model_name]
        model_path = f'outputs/normalized_ensemble/{ticker}_{model_name}_normalized_{timestamp}.csv'
        model_df.to_csv(model_path, index=False)
    
    # Print summary
    final_row = results_df.iloc[-1]
    
    print("\nNormalized Ensemble Summary:")
    print(f"Current price: ${current_price:.2f}")
    print(f"Prediction period: {results_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {results_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Final predicted price: ${final_row['Mean_Price']:.2f} (Range: ${final_row['Min_Price']:.2f} - ${final_row['Max_Price']:.2f})")
    print(f"Predicted change: {final_row['Cumulative_Return']*100:.2f}%")
    
    # Count of each direction
    direction_counts = results_df['Direction'].value_counts()
    print("\nDirection Distribution:")
    for direction, count in direction_counts.items():
        percentage = count / len(results_df) * 100
        print(f"{direction}: {count} days ({percentage:.1f}%)")
    
    # Trading recommendation
    final_confidence = final_row['Confidence']
    
    print("\nTrading Recommendation:")
    
    if final_row['Direction'] == 'UP' and final_confidence >= confidence_threshold:
        print(f"BUY - {final_confidence*100:.1f}% confidence in upward movement")
    elif final_row['Direction'] == 'DOWN' and final_confidence >= confidence_threshold:
        print(f"SELL - {final_confidence*100:.1f}% confidence in downward movement")
    else:
        print(f"HOLD - {final_confidence*100:.1f}% confidence (below threshold)")
    
    print(f"\nNOTE: This analysis is based on {len(combined_df['Model'].unique())} different models")
    return results_df

def main():
    """Main function"""
    args = parse_arguments()
    create_normalized_ensemble(
        args.ticker,
        args.confidence_threshold,
        args.debug
    )

if __name__ == "__main__":
    main() 