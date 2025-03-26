import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timezone
import yfinance as yf

# Function to load predictions from a file
def load_predictions(file_path):
    try:
        predictions = pd.read_csv(file_path)
        if 'Datetime' in predictions.columns:
            predictions['Datetime'] = pd.to_datetime(predictions['Datetime'])
            # Standardize timezones - convert all to UTC
            if predictions['Datetime'].dt.tz is not None:
                predictions['Datetime'] = predictions['Datetime'].dt.tz_convert('UTC')
            else:
                predictions['Datetime'] = predictions['Datetime'].dt.tz_localize('UTC')
            return predictions
        else:
            # Check for alternative date column names
            date_columns = [col for col in predictions.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                # Rename to Datetime and convert
                predictions.rename(columns={date_columns[0]: 'Datetime'}, inplace=True)
                predictions['Datetime'] = pd.to_datetime(predictions['Datetime'])
                predictions['Datetime'] = predictions['Datetime'].dt.tz_localize('UTC')
                return predictions
            
            # If no date column found, create one from Day column if exists
            if 'Day' in predictions.columns:
                predictions['Datetime'] = pd.to_datetime(predictions['Day'])
                predictions['Datetime'] = predictions['Datetime'].dt.tz_localize('UTC')
                return predictions
                
            print(f"No datetime column found in {file_path}")
            return None 
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to get current price
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        return current_price
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
        return None

# Function to filter and score models
def evaluate_model_quality(predictions, current_price):
    """
    Evaluate the quality of a model based on its predictions
    Returns a score between 0 and 1 (higher is better)
    """
    # Check if predictions are in a reasonable range
    first_pred = predictions['Predicted_Price'].iloc[0]
    last_pred = predictions['Predicted_Price'].iloc[-1]
    
    # Calculate percentage difference from current price
    start_pct_diff = abs((first_pred - current_price) / current_price)
    
    # Calculate volatility (standard deviation of percent changes)
    pct_changes = predictions['Predicted_Price'].pct_change().dropna()
    volatility = pct_changes.std()
    
    # Initialize base score
    score = 1.0
    
    # Penalize for starting far from current price
    if start_pct_diff > 0.1:  # More than 10% different
        score -= min(0.5, start_pct_diff)  # Penalty capped at 0.5
    
    # Penalize for excessive volatility
    if volatility > 0.05:  # More than 5% daily changes
        score -= min(0.3, volatility)  # Penalty capped at 0.3
    
    # Penalize extreme predictions (more than 20% change)
    overall_change_pct = abs((last_pred - first_pred) / first_pred)
    if overall_change_pct > 0.2:
        score -= min(0.2, overall_change_pct - 0.2)  # Penalty capped at 0.2
    
    # Ensure score is between 0 and 1
    return max(0.1, min(score, 1.0))

# Create directory for output
os.makedirs('outputs/ensemble', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Get all prediction files for a specific ticker
ticker = 'AAPL'
prediction_files = glob.glob(f'outputs/predictions/{ticker}_*.csv')

if not prediction_files:
    print(f"No prediction files found for {ticker}")
    exit()

print(f"Found {len(prediction_files)} prediction files for {ticker}")

# Load predictions
all_predictions = []
model_names = []

for file_path in prediction_files:
    model_name = os.path.basename(file_path).replace('.csv', '')
    predictions = load_predictions(file_path)
    
    if predictions is not None:
        # Check for predicted price column
        price_columns = [col for col in predictions.columns if 'price' in col.lower() or 'pred' in col.lower() or 'forecast' in col.lower()]
        if price_columns:
            if 'Predicted_Price' not in predictions.columns:
                predictions.rename(columns={price_columns[0]: 'Predicted_Price'}, inplace=True)
            
            all_predictions.append(predictions)
            model_names.append(model_name)
            print(f"Loaded {len(predictions)} predictions from {model_name}")
        else:
            print(f"No price column found in {file_path}")

if not all_predictions:
    print("No valid prediction files loaded")
    exit()

# Get current price
current_price = get_current_price(ticker)
print(f"Current price of {ticker}: ${current_price:.2f}")

# Get current time in UTC for comparison
now_utc = datetime.now(timezone.utc)
print(f"Current time (UTC): {now_utc}")

# Score and filter models
model_scores = {}
filtered_predictions = []
filtered_model_names = []

for predictions, model_name in zip(all_predictions, model_names):
    # Check if we have future dates
    future_predictions = predictions[predictions['Datetime'] > now_utc]
    
    if not future_predictions.empty:
        # Evaluate model quality
        score = evaluate_model_quality(future_predictions, current_price)
        model_scores[model_name] = score
        
        print(f"Model {model_name}: Quality score {score:.2f}")
        
        # Only use models with score above threshold
        if score > 0.3:  # Minimum quality threshold
            filtered_predictions.append(future_predictions)
            filtered_model_names.append(model_name)
            print(f"  - Included in ensemble")
        else:
            print(f"  - Excluded from ensemble (below quality threshold)")

if not filtered_predictions:
    print("No models passed quality filtering. Using all models instead.")
    filtered_predictions = [p[p['Datetime'] > now_utc] for p in all_predictions if not p[p['Datetime'] > now_utc].empty]
    filtered_model_names = [model_names[i] for i, p in enumerate(all_predictions) if not p[p['Datetime'] > now_utc].empty]

all_predictions = filtered_predictions
model_names = filtered_model_names

# Create ensemble prediction visualization
plt.figure(figsize=(14, 8))

# Plot each model's predictions
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', '*']

# Track future predictions
has_future_predictions = False

for i, (predictions, model_name) in enumerate(zip(all_predictions, model_names)):
    color_idx = i % len(colors)
    style_idx = i % len(line_styles)
    marker_idx = i % len(markers)
    
    # Check if we have future dates (handling timezone-aware comparison)
    future_predictions = predictions[predictions['Datetime'] > now_utc]
    
    if not future_predictions.empty:
        has_future_predictions = True
        # Get the change percentage
        first_price = future_predictions['Predicted_Price'].iloc[0]
        last_price = future_predictions['Predicted_Price'].iloc[-1]
        change = ((last_price / first_price) - 1) * 100
        direction = "UP" if change > 0 else "DOWN"
        
        # Create a short display name
        display_name = model_name.split('_')[-2:]
        display_name = '_'.join(display_name) if len(display_name) > 1 else model_name.split('_')[-1]
        
        label = f"{display_name}: {direction} ({change:.2f}%)"
        
        plt.plot(future_predictions['Datetime'], 
                 future_predictions['Predicted_Price'], 
                 color=colors[color_idx],
                 linestyle=line_styles[style_idx],
                 marker=markers[marker_idx],
                 linewidth=2,
                 markersize=5,
                 label=label)
        
        # Add final price annotation
        plt.annotate(f"${last_price:.2f}",
                    xy=(future_predictions['Datetime'].iloc[-1], last_price),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=colors[color_idx])
    else:
        # If no future predictions, plot all predictions but with different style
        plt.plot(predictions['Datetime'], 
                 predictions['Predicted_Price'], 
                 color=colors[color_idx],
                 linestyle=':',
                 marker=markers[marker_idx],
                 linewidth=1,
                 markersize=3,
                 alpha=0.5,
                 label=f"{model_name.split('_')[-1]}: Historical")

# If no future predictions were found, show a notice
if not has_future_predictions:
    print("Warning: No future predictions found in any model. Showing historical data only.")
    plt.axvline(x=now_utc, color='red', linestyle='--', alpha=0.7, label='Current Time')
    plt.title(f'{ticker} Historical Model Predictions', fontsize=16)
else:
    # Add current price line
    if current_price is not None:
        plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.3, label=f'Current: ${current_price:.2f}')
    
    # Format the chart
    plt.title(f'{ticker} Model Ensemble Prediction Comparison', fontsize=16)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# Save the chart
chart_path = f'outputs/ensemble/{ticker}_ensemble_prediction_{timestamp}.png'
plt.tight_layout()
plt.savefig(chart_path)
plt.close()

print(f"Ensemble visualization saved to {chart_path}")

if not has_future_predictions:
    print("No future predictions available for ensemble analysis. Exiting.")
    exit()

# Create ensemble prediction (weighted average)
# We'll create a combined prediction for dates that appear in multiple predictions
all_dates = set()
for predictions in all_predictions:
    all_dates.update(predictions['Datetime'].tolist())

all_dates = sorted(all_dates)

# Create ensemble dataframe
ensemble_df = pd.DataFrame({
    'Datetime': all_dates,
    'Ensemble_Price': np.nan,
    'Min_Price': np.nan,
    'Max_Price': np.nan,
    'Model_Count': 0
})

# Fill ensemble dataframe with weighted average
for i, predictions in enumerate(all_predictions):
    model_name = model_names[i]
    # Get model weight based on quality score
    weight = model_scores.get(model_name, 0.5)  # Default to 0.5 if not scored
    
    for idx, row in predictions.iterrows():
        date_match = ensemble_df['Datetime'] == row['Datetime']
        
        if any(date_match):
            # Update existing entry
            ensemble_idx = ensemble_df.index[date_match][0]
            current_count = ensemble_df.loc[ensemble_idx, 'Model_Count']
            current_price = ensemble_df.loc[ensemble_idx, 'Ensemble_Price']
            
            # Update ensemble price (weighted average)
            if np.isnan(current_price):
                ensemble_df.loc[ensemble_idx, 'Ensemble_Price'] = row['Predicted_Price']
            else:
                # Apply weighted average
                ensemble_df.loc[ensemble_idx, 'Ensemble_Price'] = (
                    (current_price * current_count + row['Predicted_Price'] * weight) / 
                    (current_count + weight)
                )
            
            # Update min/max prices
            if np.isnan(ensemble_df.loc[ensemble_idx, 'Min_Price']) or row['Predicted_Price'] < ensemble_df.loc[ensemble_idx, 'Min_Price']:
                ensemble_df.loc[ensemble_idx, 'Min_Price'] = row['Predicted_Price']
                
            if np.isnan(ensemble_df.loc[ensemble_idx, 'Max_Price']) or row['Predicted_Price'] > ensemble_df.loc[ensemble_idx, 'Max_Price']:
                ensemble_df.loc[ensemble_idx, 'Max_Price'] = row['Predicted_Price']
            
            # Increment model count
            ensemble_df.loc[ensemble_idx, 'Model_Count'] += weight

# Calculate consensus metrics
ensemble_df['Price_Range'] = ensemble_df['Max_Price'] - ensemble_df['Min_Price']
ensemble_df['Range_Pct'] = (ensemble_df['Price_Range'] / ensemble_df['Ensemble_Price']) * 100
ensemble_df['Consensus'] = np.where(ensemble_df['Range_Pct'] < 5, 'Strong', 
                           np.where(ensemble_df['Range_Pct'] < 10, 'Moderate', 'Weak'))

# Save ensemble predictions
ensemble_path = f'outputs/ensemble/{ticker}_ensemble_prediction_{timestamp}.csv'
ensemble_df.to_csv(ensemble_path, index=False)

# Plot ensemble prediction with confidence intervals
plt.figure(figsize=(14, 8))

# Plot ensemble prediction
plt.plot(ensemble_df['Datetime'], ensemble_df['Ensemble_Price'], 
         color='blue', linewidth=3, label='Ensemble Prediction')

# Plot confidence interval
plt.fill_between(ensemble_df['Datetime'], 
                 ensemble_df['Min_Price'], 
                 ensemble_df['Max_Price'], 
                 color='blue', alpha=0.2, label='Prediction Range')

# Add final price annotation
final_price = ensemble_df['Ensemble_Price'].iloc[-1]
if current_price is not None:
    change_pct = ((final_price / current_price) - 1) * 100
    plt.annotate(f"${final_price:.2f} ({change_pct:+.2f}%)", 
                xy=(ensemble_df['Datetime'].iloc[-1], final_price),
                xytext=(10, 0), textcoords='offset points',
                fontsize=12, fontweight='bold', color='blue')

# Add current price line
if current_price is not None:
    plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.3, label=f'Current: ${current_price:.2f}')

# Format the chart
plt.title(f'{ticker} Ensemble Prediction with Confidence Interval', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# Add direction and consensus
if current_price is not None:
    direction = "UP" if final_price > current_price else "DOWN"
    color = 'green' if direction == "UP" else 'red'
    consensus = ensemble_df['Consensus'].mode()[0]
    
    plt.annotate(f"Direction: {direction}\nConsensus: {consensus}", 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.2))

# Save the chart
ensemble_chart_path = f'outputs/ensemble/{ticker}_ensemble_confidence_{timestamp}.png'
plt.tight_layout()
plt.savefig(ensemble_chart_path)
plt.close()

print(f"Ensemble confidence visualization saved to {ensemble_chart_path}")
print(f"Ensemble prediction data saved to {ensemble_path}")

# Print summary
if current_price is not None:
    print("\nEnsemble Prediction Summary:")
    print(f"Initial ensemble price: ${ensemble_df['Ensemble_Price'].iloc[0]:.2f}")
    print(f"Final ensemble price: ${final_price:.2f}")
    print(f"Predicted change: {change_pct:.2f}%")
    print(f"Direction: {direction}")
    print(f"Consensus strength: {consensus}")
    print(f"Model agreement: {ensemble_df['Model_Count'].mean():.1f} models per prediction point")
    
    # Trading recommendation
    print("\nTrading Recommendation:")
    if change_pct > 1.5 and consensus in ['Strong', 'Moderate']:
        print("• STRONG BUY - Multiple models predict significant upside with good consensus")
    elif change_pct > 0.5:
        print("• BUY - Models predict positive movement")
    elif change_pct < -1.5 and consensus in ['Strong', 'Moderate']:
        print("• STRONG SELL - Multiple models predict significant downside with good consensus")
    elif change_pct < -0.5:
        print("• SELL - Models predict negative movement")
    else:
        print("• HOLD - No strong directional signal from the models")
        
    print(f"\nNOTE: These predictions are based on {len(all_predictions)} different models")
    print("      Always use proper risk management with any trading decision") 