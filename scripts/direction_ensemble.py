import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timezone
import yfinance as yf

def load_predictions(file_path):
    """Load predictions from a file"""
    try:
        predictions = pd.read_csv(file_path)
        date_cols = [col for col in predictions.columns if 'date' in col.lower() or 'time' in col.lower() or col == 'Date' or col == 'Datetime']
        
        if date_cols:
            date_col = date_cols[0]
            predictions[date_col] = pd.to_datetime(predictions[date_col])
            
            # Standardize column names
            if date_col != 'Datetime':
                predictions.rename(columns={date_col: 'Datetime'}, inplace=True)
                
            # Standardize timezones
            if predictions['Datetime'].dt.tz is not None:
                predictions['Datetime'] = predictions['Datetime'].dt.tz_convert('UTC')
            else:
                predictions['Datetime'] = predictions['Datetime'].dt.tz_localize('UTC')
        else:
            print(f"No date column found in {file_path}")
            return None
            
        # Find price column
        price_cols = [col for col in predictions.columns if 'price' in col.lower() or 'pred' in col.lower() or 'forecast' in col.lower()]
        if price_cols:
            if 'Predicted_Price' not in predictions.columns:
                predictions.rename(columns={price_cols[0]: 'Predicted_Price'}, inplace=True)
        else:
            print(f"No price column found in {file_path}")
            return None
            
        return predictions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_current_price(ticker):
    """Get the current price of a stock"""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        return current_price
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
        return None

def ensemble_direction_prediction(ticker='AAPL', future_days=14):
    """Create an ensemble direction prediction focusing just on future predictions"""
    # Create output directories
    os.makedirs('outputs/ensemble', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Get prediction files
    prediction_files = glob.glob(f'outputs/predictions/{ticker}_*.csv')
    
    if not prediction_files:
        print(f"No prediction files found for {ticker}")
        return
        
    print(f"Found {len(prediction_files)} prediction files for {ticker}")
    
    # Get current price
    current_price = get_current_price(ticker)
    
    if current_price is None:
        print("Could not fetch current price. Exiting.")
        return
        
    print(f"Current price of {ticker}: ${current_price:.2f}")
    
    # Get current time in UTC
    now_utc = datetime.now(timezone.utc)
    
    # Collect future predictions from all models
    all_future_predictions = []
    
    for file_path in prediction_files:
        predictions = load_predictions(file_path)
        if predictions is None:
            continue
            
        model_name = os.path.basename(file_path).replace('.csv', '')
        
        # Get future predictions
        future_predictions = predictions[predictions['Datetime'] > now_utc]
        
        if future_predictions.empty:
            print(f"No future predictions found in {model_name}")
            continue
            
        # Ensure predictions are reasonable
        first_pred = future_predictions['Predicted_Price'].iloc[0]
        price_diff_pct = abs((first_pred - current_price) / current_price)
        
        if price_diff_pct > 0.2:  # More than 20% different from current price
            print(f"Warning: {model_name} predictions start at ${first_pred:.2f}, which is {price_diff_pct*100:.1f}% from current price")
            # Apply scaling correction - create a copy to avoid SettingWithCopyWarning
            future_predictions = future_predictions.copy()
            scaling_factor = current_price / first_pred
            future_predictions.loc[:, 'Predicted_Price'] = future_predictions['Predicted_Price'] * scaling_factor
            print(f"Applied scaling correction. New first prediction: ${future_predictions['Predicted_Price'].iloc[0]:.2f}")
        
        # Calculate the predicted direction (up or down)
        pct_changes = future_predictions['Predicted_Price'].pct_change().fillna(0)
        up_days = (pct_changes > 0).sum()
        down_days = (pct_changes < 0).sum()
        direction = 'UP' if up_days > down_days else 'DOWN'
        
        # Calculate the overall change
        first_price = future_predictions['Predicted_Price'].iloc[0]
        last_price = future_predictions['Predicted_Price'].iloc[-1]
        overall_change = ((last_price / first_price) - 1) * 100
        
        print(f"Model {model_name}: Direction {direction}, Change {overall_change:.2f}%")
        
        all_future_predictions.append((future_predictions, model_name, direction, overall_change))
    
    if not all_future_predictions:
        print("No valid future predictions found")
        return
        
    # Combine all predictions into a single dataframe
    all_dates = set()
    for pred, _, _, _ in all_future_predictions:
        all_dates.update(pred['Datetime'].tolist())
        
    all_dates = sorted(all_dates)
    
    # Create date range for consistency
    date_range = pd.date_range(start=min(all_dates), end=max(all_dates), freq='D')
    
    # Initialize ensemble DataFrame
    ensemble_df = pd.DataFrame({
        'Datetime': date_range,
        'Up_Vote': 0,
        'Down_Vote': 0,
        'Direction_Confidence': 0.0,
        'Weighted_Price': 0.0,
        'Model_Count': 0
    })
    
    # Fill ensemble DataFrame
    for predictions, model_name, model_direction, overall_change in all_future_predictions:
        # Weight based on how reasonable the overall change is
        # More extreme changes get less weight
        weight = 1.0
        if abs(overall_change) > 10:
            weight = 0.5  # Reduce weight for predictions with >10% change
        
        # Map dates to ensemble DataFrame
        for i, row in predictions.iterrows():
            # Find the closest date in the ensemble DataFrame
            closest_date_idx = abs(ensemble_df['Datetime'] - row['Datetime']).idxmin()
            
            # Update weighted price
            current_count = ensemble_df.loc[closest_date_idx, 'Model_Count']
            current_price = ensemble_df.loc[closest_date_idx, 'Weighted_Price']
            
            if current_count == 0:
                ensemble_df.loc[closest_date_idx, 'Weighted_Price'] = row['Predicted_Price']
            else:
                ensemble_df.loc[closest_date_idx, 'Weighted_Price'] = (
                    (current_price * current_count + row['Predicted_Price'] * weight) / 
                    (current_count + weight)
                )
            
            # Update direction votes
            if i > 0 and i < len(predictions) - 1:  # Skip first and last points
                prev_price = predictions['Predicted_Price'].iloc[i-1]
                daily_change = (row['Predicted_Price'] - prev_price) / prev_price
                
                if daily_change > 0:
                    ensemble_df.loc[closest_date_idx, 'Up_Vote'] += weight
                else:
                    ensemble_df.loc[closest_date_idx, 'Down_Vote'] += weight
            
            # Increment model count
            ensemble_df.loc[closest_date_idx, 'Model_Count'] += weight
    
    # Calculate direction confidence
    ensemble_df['Total_Votes'] = ensemble_df['Up_Vote'] + ensemble_df['Down_Vote']
    ensemble_df['Direction_Confidence'] = np.where(
        ensemble_df['Total_Votes'] > 0,
        np.maximum(ensemble_df['Up_Vote'], ensemble_df['Down_Vote']) / ensemble_df['Total_Votes'],
        0
    )
    
    # Calculate final direction
    ensemble_df['Direction'] = np.where(
        ensemble_df['Up_Vote'] > ensemble_df['Down_Vote'],
        'UP',
        'DOWN'
    )
    
    # Calculate percentage change from current price
    ensemble_df['Pct_Change'] = ((ensemble_df['Weighted_Price'] / current_price) - 1) * 100
    
    # Plot ensemble direction prediction
    plt.figure(figsize=(14, 8))
    
    # Plot price prediction
    plt.subplot(2, 1, 1)
    plt.plot(ensemble_df['Datetime'], ensemble_df['Weighted_Price'], 'b-', linewidth=2, label='Ensemble Price')
    
    # Add confidence bands
    for i, (dt, price, conf, direction) in enumerate(zip(
        ensemble_df['Datetime'], ensemble_df['Weighted_Price'], 
        ensemble_df['Direction_Confidence'], ensemble_df['Direction']
    )):
        color = 'green' if direction == 'UP' else 'red'
        alpha = min(1, conf + 0.2)  # Minimum alpha of 0.2
        
        plt.plot(dt, price, 'o', color=color, alpha=alpha, markersize=8)
        
        # Add arrow to show direction
        if i > 0:
            prev_price = ensemble_df['Weighted_Price'].iloc[i-1]
            if direction == 'UP':
                plt.annotate('', xy=(dt, price), xytext=(dt, prev_price),
                            arrowprops=dict(facecolor=color, alpha=alpha, width=2))
            else:
                plt.annotate('', xy=(dt, price), xytext=(dt, prev_price),
                            arrowprops=dict(facecolor=color, alpha=alpha, width=2))
    
    # Add horizontal line for current price
    plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.3, 
                label=f'Current: ${current_price:.2f}')
    
    plt.title(f'{ticker} Ensemble Direction Prediction', fontsize=16)
    plt.ylabel('Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot direction confidence
    plt.subplot(2, 1, 2)
    
    # Plot direction confidence
    for i, (dt, conf, direction) in enumerate(zip(
        ensemble_df['Datetime'], ensemble_df['Direction_Confidence'], ensemble_df['Direction']
    )):
        color = 'green' if direction == 'UP' else 'red'
        plt.bar(dt, conf, color=color, alpha=0.7)
    
    plt.title(f'Direction Prediction Confidence', fontsize=16)
    plt.ylabel('Confidence', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'outputs/ensemble/{ticker}_direction_ensemble_{timestamp}.png')
    
    # Save ensemble DataFrame
    ensemble_df.to_csv(f'outputs/ensemble/{ticker}_direction_ensemble_{timestamp}.csv', index=False)
    
    # Print summary
    final_direction = ensemble_df['Direction'].value_counts().idxmax()  # Most common direction
    avg_confidence = ensemble_df['Direction_Confidence'].mean()
    final_price = ensemble_df['Weighted_Price'].iloc[-1]
    price_change = (final_price - current_price) / current_price * 100
    
    print("\nEnsemble Direction Prediction Summary:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Final Price: ${final_price:.2f} ({price_change:.2f}%)")
    print(f"Overall Direction: {final_direction} with {avg_confidence*100:.1f}% average confidence")
    
    # Trading recommendation
    if avg_confidence >= 0.7:
        if final_direction == 'UP':
            print("\nTrading Recommendation: STRONG BUY")
            print(f"High confidence ({avg_confidence*100:.1f}%) in upward movement")
        else:
            print("\nTrading Recommendation: STRONG SELL")
            print(f"High confidence ({avg_confidence*100:.1f}%) in downward movement")
    elif avg_confidence >= 0.6:
        if final_direction == 'UP':
            print("\nTrading Recommendation: BUY")
            print(f"Moderate confidence ({avg_confidence*100:.1f}%) in upward movement")
        else:
            print("\nTrading Recommendation: SELL")
            print(f"Moderate confidence ({avg_confidence*100:.1f}%) in downward movement")
    else:
        print("\nTrading Recommendation: HOLD/NEUTRAL")
        print(f"Low confidence ({avg_confidence*100:.1f}%) in direction prediction")
    
    return ensemble_df

if __name__ == "__main__":
    ensemble_direction_prediction('AAPL') 