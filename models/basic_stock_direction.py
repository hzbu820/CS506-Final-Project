import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import yfinance as yf
import time

def log_message(message):
    """Print a message with a timestamp and also write to a log file"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Write to log file
    with open('outputs/model_log.txt', 'a') as f:
        f.write(log_msg + '\n')

def get_stock_data(ticker="SPY", period="30d", interval="15m", use_cache=True):
    """Get real stock data using yfinance or from cache"""
    cache_file = f"{ticker}_{interval}_{period}.csv"
    
    if use_cache and os.path.exists(cache_file):
        log_message(f"Loading data from cache: {cache_file}")
        try:
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # Ensure numeric columns are properly converted
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop any rows with NaN values that might have resulted from conversion
            data = data.dropna()
            
            log_message(f"Loaded {len(data)} data points from cache")
            return data
        except Exception as e:
            log_message(f"Error loading from cache: {e}. Will download fresh data.")
            # Delete corrupted cache file
            try:
                os.remove(cache_file)
                log_message(f"Removed corrupted cache file: {cache_file}")
            except:
                pass
    
    try:
        log_message(f"Downloading {ticker} data for {period} with {interval} intervals...")
        data = yf.download(ticker, period=period, interval=interval, timeout=10)
        log_message(f"Downloaded {len(data)} data points")
        
        # Save to cache
        data.to_csv(cache_file)
        log_message(f"Saved data to cache: {cache_file}")
        
        return data
    except Exception as e:
        log_message(f"Error downloading data: {e}")
        
        # Try loading from cache as fallback
        if os.path.exists(cache_file):
            log_message(f"Loading from cache as fallback")
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure numeric columns
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna()
            return data
        else:
            raise

def add_indicators(df):
    """Add technical indicators to data"""
    log_message("Adding technical indicators...")
    
    data = df.copy()
    
    # Moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Moving average crossovers
    data['MA5_10_Cross'] = (data['MA5'] > data['MA10']).astype(int)
    data['MA10_20_Cross'] = (data['MA10'] > data['MA20']).astype(int)
    data['MA5_20_Cross'] = (data['MA5'] > data['MA20']).astype(int)
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Price momentum and rate of change
    data['Price_Change'] = data['Close'].diff()
    data['ROC_5'] = data['Close'].pct_change(periods=5) * 100
    data['ROC_10'] = data['Close'].pct_change(periods=10) * 100
    data['ROC_20'] = data['Close'].pct_change(periods=20) * 100
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    data['BB_Squeeze'] = (data['BB_Width'] < data['BB_Width'].rolling(window=50).mean()).astype(int)
    
    # Volatility measures
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['Volatility_Change'] = data['Volatility'].diff()
    data['ATR'] = data['High'] - data['Low'] # Simple Approximation of ATR
    
    # Support and resistance levels (basic approximation)
    data['Resistance'] = data['High'].rolling(window=20).max()
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Close_to_Resistance'] = (data['Resistance'] - data['Close']) / data['Close'] * 100
    data['Close_to_Support'] = (data['Close'] - data['Support']) / data['Close'] * 100
    
    # Target: Direction of next period's close price (1 for up, 0 for down)
    data['Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Fill NaN values
    data = data.ffill().fillna(0)
    
    log_message(f"Data shape after adding indicators: {data.shape}")
    return data

def generate_predictions(df):
    """Generate predictions using simple rules"""
    log_message("Generating predictions using simple rules...")
    
    data = df.copy()
    
    # Trend following signals
    data['Pred_MA_Cross'] = (data['MA5'] > data['MA10']).astype(int)
    data['Pred_MA_Trend'] = (data['MA5'].diff() > 0).astype(int)
    data['Pred_MACD'] = (data['MACD'] > data['MACD_Signal']).astype(int)
    data['Pred_MACD_Rising'] = (data['MACD'].diff() > 0).astype(int)
    data['Pred_MACD_Hist_Rising'] = (data['MACD_Hist'].diff() > 0).astype(int)
    
    # Mean reversion signals
    data['Pred_RSI_Oversold'] = (data['RSI'] < 30).astype(int)
    data['Pred_RSI_Overbought'] = (data['RSI'] > 70).astype(int)
    data['Pred_RSI_Rising'] = (data['RSI'].diff() > 0).astype(int)
    data['Pred_RSI_Falling'] = (data['RSI'].diff() < 0).astype(int)
    
    # Price action signals
    data['Pred_Price_Momentum'] = (data['Price_Change'] > 0).astype(int)
    data['Pred_ROC_5_Rising'] = (data['ROC_5'] > 0).astype(int)
    data['Pred_ROC_10_Rising'] = (data['ROC_10'] > 0).astype(int)
    
    # Volatility signals
    data['Pred_Low_Volatility'] = (data['Volatility'] < data['Volatility'].rolling(window=50).mean()).astype(int)
    data['Pred_Decreasing_Volatility'] = (data['Volatility_Change'] < 0).astype(int)
    data['Pred_BB_Squeeze'] = data['BB_Squeeze']  # Already binary
    
    # Support/Resistance signals
    data['Pred_Near_Support'] = (data['Close_to_Support'] < 1.0).astype(int)
    data['Pred_Near_Resistance'] = (data['Close_to_Resistance'] < 1.0).astype(int)
    
    # Candlestick patterns - handle potential DataFrame issues
    try:
        data['Body_Size'] = abs(data['Close'] - data['Open'])
        data['Range'] = data['High'] - data['Low']
        close_gt_open = data['Close'] > data['Open']
        body_gt_half_range = data['Body_Size'] > (0.5 * data['Range'])
        data['Pred_Strong_Close'] = (close_gt_open & body_gt_half_range).astype(int)
    except Exception as e:
        log_message(f"Warning: Could not compute Pred_Strong_Close: {e}")
        data['Pred_Strong_Close'] = 0  # Set a default value
    
    # Volume analysis
    try:
        if 'Volume' in data.columns:
            data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
            data['Pred_High_Volume'] = (data['Volume'] > 1.5 * data['Volume_MA5']).astype(int)
            data['Pred_Rising_Volume'] = (data['Volume'].diff() > 0).astype(int)
        else:
            data['Pred_High_Volume'] = 0
            data['Pred_Rising_Volume'] = 0
    except Exception as e:
        log_message(f"Warning: Could not compute volume indicators: {e}")
        data['Pred_High_Volume'] = 0
        data['Pred_Rising_Volume'] = 0
    
    # More sophisticated weighted combination for real market data
    data['Combined_Score'] = (
        # Trend following signals (stronger weight in real markets)
        2.0 * data['Pred_MA_Cross'] +
        1.0 * data['Pred_MA_Trend'] +
        2.0 * data['Pred_MACD'] +
        1.0 * data['Pred_MACD_Rising'] +
        1.5 * data['Pred_MACD_Hist_Rising'] +
        
        # Mean reversion signals (careful with these in trending markets)
        0.5 * data['Pred_RSI_Oversold'] -
        0.5 * data['Pred_RSI_Overbought'] +
        0.8 * data['Pred_RSI_Rising'] +
        
        # Price action signals (strong in real markets)
        1.5 * data['Pred_Price_Momentum'] +
        1.0 * data['Pred_ROC_5_Rising'] +
        0.8 * data['Pred_ROC_10_Rising'] +
        
        # Volatility signals
        0.5 * data['Pred_Low_Volatility'] +
        0.6 * data['Pred_Decreasing_Volatility'] +
        1.0 * data['Pred_BB_Squeeze'] +
        
        # Support/Resistance signals
        1.2 * data['Pred_Near_Support'] -
        0.8 * data['Pred_Near_Resistance'] +
        
        # Candlestick patterns
        1.2 * data['Pred_Strong_Close'] +
        
        # Volume signals
        0.8 * data['Pred_High_Volume'] +
        0.5 * data['Pred_Rising_Volume']
    )
    
    # Optimize threshold based on historical performance
    thresholds = np.arange(-1.0, 6.0, 0.1)  # Wider range of thresholds
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        # Create temporary prediction
        temp_prediction = (data['Combined_Score'] > threshold).astype(int)
        
        # Calculate F1 score for this threshold
        tp = np.sum((temp_prediction == 1) & (data['Direction'] == 1))
        fp = np.sum((temp_prediction == 1) & (data['Direction'] == 0))
        fn = np.sum((temp_prediction == 0) & (data['Direction'] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    log_message(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
    
    # Final prediction: predict up if combined score > threshold
    data['Prediction'] = (data['Combined_Score'] > best_threshold).astype(int)
    
    # Drop rows with NaN in the actual direction (last row)
    data = data.dropna(subset=['Direction'])
    
    log_message(f"Generated predictions for {len(data)} data points")
    return data

def evaluate_predictions(df):
    """Evaluate prediction performance"""
    log_message("Evaluating prediction performance...")
    
    # Make sure data has both actual and prediction columns
    if 'Direction' not in df.columns or 'Prediction' not in df.columns:
        log_message("ERROR: Missing required columns")
        return None
    
    # Calculate metrics
    correct = np.sum(df['Direction'] == df['Prediction'])
    total = len(df)
    accuracy = correct / total
    
    # Calculate confusion matrix metrics
    tp = np.sum((df['Prediction'] == 1) & (df['Direction'] == 1))
    fp = np.sum((df['Prediction'] == 1) & (df['Direction'] == 0))
    tn = np.sum((df['Prediction'] == 0) & (df['Direction'] == 0))
    fn = np.sum((df['Prediction'] == 0) & (df['Direction'] == 1))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Log performance metrics
    log_message(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    log_message(f"Precision: {precision:.4f}")
    log_message(f"Recall: {recall:.4f}")
    log_message(f"F1 Score: {f1:.4f}")
    
    # Calculate UP prediction accuracy
    up_actual = np.sum(df['Direction'] == 1)
    up_correct = tp
    up_accuracy = up_correct / up_actual if up_actual > 0 else 0
    log_message(f"UP prediction accuracy: {up_accuracy:.4f} ({up_correct}/{up_actual})")
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'up_accuracy': up_accuracy,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }
    
    return metrics

def plot_results(df, metrics):
    """Plot prediction results"""
    log_message("Plotting results...")
    
    try:
        os.makedirs('outputs', exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Stock price with predictions
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df['Close'], color='blue', label='Close Price')
        
        # Add buy/sell signals
        buy_signals = df[df['Prediction'] == 1]['Close']
        sell_signals = df[df['Prediction'] == 0]['Close']
        
        plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', alpha=0.7, label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', alpha=0.7, label='Sell Signal')
        
        plt.title('Stock Price with Buy/Sell Signals')
        plt.legend()
        
        # Plot 2: Actual vs Predicted Direction
        plt.subplot(2, 2, 2)
        plt.plot(df.index, df['Direction'], label='Actual', color='blue')
        plt.plot(df.index, df['Prediction'], label='Predicted', color='red', linestyle='--')
        plt.title('Actual vs Predicted Direction')
        plt.legend()
        
        # Plot 3: Confusion Matrix
        plt.subplot(2, 2, 3)
        cm = np.array(metrics['confusion_matrix'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Down', 'Up'])
        plt.yticks([0, 1], ['Down', 'Up'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        # Plot 4: Metrics
        plt.subplot(2, 2, 4)
        metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'UP Accuracy']
        metric_values = [metrics['accuracy'], metrics['f1_score'], 
                        metrics['precision'], metrics['recall'], 
                        metrics['up_accuracy']]
        plt.bar(metric_names, metric_values)
        plt.ylim(0, 1)
        plt.title('Performance Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/real_data_prediction_results.png')
        plt.close()
        
        log_message("Results saved to outputs/real_data_prediction_results.png")
    except Exception as e:
        log_message(f"Error during plotting: {e}")

def main():
    """Main function"""
    try:
        start_time = datetime.datetime.now()
        log_message("Starting stock direction prediction using real data...")
        
        # Stock ticker and data interval settings
        ticker = "SPY"  # S&P 500 ETF
        period = "60d"  # 60 days of data
        interval = "15m"  # 15-minute intervals
        
        # Make sure output directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # Get real stock data
        data = get_stock_data(ticker=ticker, period=period, interval=interval, use_cache=True)
        log_message(f"Using real {interval} data for {ticker}")
        
        # Add technical indicators
        data = add_indicators(data)
        
        # Split data into training and testing sets (80% training, 20% testing)
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        log_message(f"Training data size: {len(train_data)}, Testing data size: {len(test_data)}")
        
        # Generate predictions for training data to optimize threshold
        train_with_predictions = generate_predictions(train_data)
        best_threshold = train_with_predictions.get('best_threshold', 0.0) 
        
        # Find best threshold from training data if not already set
        if best_threshold == 0:
            for col in train_with_predictions.columns:
                if 'Combined_Score' in col:
                    thresholds = np.arange(-1.0, 6.0, 0.1)
                    best_f1 = 0
                    
                    for threshold in thresholds:
                        temp_pred = (train_with_predictions['Combined_Score'] > threshold).astype(int)
                        tp = np.sum((temp_pred == 1) & (train_with_predictions['Direction'] == 1))
                        fp = np.sum((temp_pred == 1) & (train_with_predictions['Direction'] == 0))
                        fn = np.sum((temp_pred == 0) & (train_with_predictions['Direction'] == 1))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                    
                    log_message(f"Best threshold from training: {best_threshold:.2f}, F1: {best_f1:.4f}")
                    break
        
        # Generate predictions for test data
        test_with_predictions = generate_predictions(test_data)
        
        # Use the best threshold from training
        test_with_predictions['Prediction'] = (test_with_predictions['Combined_Score'] > best_threshold).astype(int)
        
        # Evaluate predictions on test data
        test_metrics = evaluate_predictions(test_with_predictions)
        
        # Plot results for this ticker and save outputs
        if test_metrics:
            output_file = f'outputs/{ticker}_{interval}_prediction_results.png'
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Stock price with predictions
            plt.subplot(2, 2, 1)
            plt.plot(test_with_predictions.index, test_with_predictions['Close'], color='blue', label='Close Price')
            
            # Add buy/sell signals
            buy_signals = test_with_predictions[test_with_predictions['Prediction'] == 1]['Close']
            sell_signals = test_with_predictions[test_with_predictions['Prediction'] == 0]['Close']
            
            plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', alpha=0.7, label='Buy Signal')
            plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', alpha=0.7, label='Sell Signal')
            
            plt.title(f'{ticker} Price with Buy/Sell Signals')
            plt.legend()
            
            # Plot 2: Actual vs Predicted Direction
            plt.subplot(2, 2, 2)
            plt.plot(test_with_predictions.index, test_with_predictions['Direction'], label='Actual', color='blue')
            plt.plot(test_with_predictions.index, test_with_predictions['Prediction'], label='Predicted', color='red', linestyle='--')
            plt.title('Actual vs Predicted Direction')
            plt.legend()
            
            # Plot 3: Confusion Matrix
            plt.subplot(2, 2, 3)
            cm = np.array(test_metrics['confusion_matrix'])
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xticks([0, 1], ['Down', 'Up'])
            plt.yticks([0, 1], ['Down', 'Up'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Add text annotations to confusion matrix
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            
            # Plot 4: Metrics
            plt.subplot(2, 2, 4)
            metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'UP Accuracy']
            metric_values = [test_metrics['accuracy'], test_metrics['f1_score'],
                            test_metrics['precision'], test_metrics['recall'],
                            test_metrics['up_accuracy']]
            plt.bar(metric_names, metric_values)
            plt.ylim(0, 1)
            plt.title('Performance Metrics')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            log_message(f"Results saved to {output_file}")
            
            # Compare with baseline
            baseline_f1 = 0.8
            if test_metrics['f1_score'] >= baseline_f1:
                log_message(f"SUCCESS! {ticker} F1 score ({test_metrics['f1_score']:.4f}) meets or exceeds baseline ({baseline_f1:.4f})")
            else:
                improvement_needed = ((baseline_f1 - test_metrics['f1_score']) / test_metrics['f1_score']) * 100
                log_message(f"{ticker} F1 score ({test_metrics['f1_score']:.4f}) is below baseline ({baseline_f1:.4f})")
                log_message(f"Improvement of {improvement_needed:.2f}% needed")
            
            # Print detailed metrics
            log_message("\nDetailed Performance Metrics:")
            log_message("-" * 50)
            log_message(f"Accuracy: {test_metrics['accuracy']:.4f}")
            log_message(f"Precision: {test_metrics['precision']:.4f}")
            log_message(f"Recall: {test_metrics['recall']:.4f}")
            log_message(f"F1 Score: {test_metrics['f1_score']:.4f}")
            log_message(f"UP Prediction Accuracy: {test_metrics['up_accuracy']:.4f}")
            
            # Print confusion matrix in readable format
            log_message("\nConfusion Matrix:")
            log_message("-" * 50)
            log_message(f"             | Predicted Down | Predicted Up")
            log_message(f"Actual Down  | {test_metrics['confusion_matrix'][0][0]:14d} | {test_metrics['confusion_matrix'][0][1]:11d}")
            log_message(f"Actual Up    | {test_metrics['confusion_matrix'][1][0]:14d} | {test_metrics['confusion_matrix'][1][1]:11d}")
        
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        log_message(f"Completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        log_message(f"Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main() 