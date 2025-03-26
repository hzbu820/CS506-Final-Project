"""
Simple Analysis of Existing Results - Minimal dependencies
"""

import os
import sys
import glob
import csv
from datetime import datetime

def main():
    """Analyze existing output files"""
    print("=" * 60)
    print("Simple Analysis of Existing Prediction Results")
    print("=" * 60)
    
    # Check for prediction files
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    
    print(f"Analyzing existing prediction files for {ticker}...")
    
    # Find prediction files
    prediction_dir = os.path.join("outputs", "predictions")
    ensemble_dir = os.path.join("outputs", "ensemble")
    normalized_dir = os.path.join("outputs", "normalized_ensemble")
    
    all_dirs = [
        (prediction_dir, "Standard predictions"),
        (ensemble_dir, "Ensemble predictions"),
        (normalized_dir, "Normalized ensemble predictions")
    ]
    
    all_predictions = []
    
    # Process each directory
    for directory, desc in all_dirs:
        if os.path.exists(directory):
            pattern = os.path.join(directory, f"{ticker}_*.csv")
            matching_files = glob.glob(pattern)
            
            print(f"\nFound {len(matching_files)} {desc} files in {directory}")
            
            for file_path in matching_files:
                try:
                    # Get file modification time
                    mod_time = os.path.getmtime(file_path)
                    mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Get file size
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    
                    # Read the first and last rows to get prediction info
                    first_price = None
                    last_price = None
                    date_col = None
                    price_col = None
                    
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        
                        # Find column indices
                        for i, header in enumerate(headers):
                            if header.lower() in ['date', 'datetime']:
                                date_col = i
                            elif header.lower() in ['predicted_price', 'price_predicted', 'mean_price', 'close']:
                                price_col = i
                        
                        if date_col is not None and price_col is not None:
                            # Get first row data
                            first_row = next(reader)
                            first_price = float(first_row[price_col])
                            
                            # Read all rows to get the last one
                            rows = list(reader)
                            if rows:
                                last_row = rows[-1]
                                last_price = float(last_row[price_col])
                    
                    if first_price is not None and last_price is not None:
                        file_name = os.path.basename(file_path)
                        pct_change = ((last_price - first_price) / first_price) * 100
                        direction = "UP" if pct_change > 0 else "DOWN"
                        
                        prediction_info = {
                            'file': file_name,
                            'directory': os.path.basename(directory),
                            'mod_date': mod_date,
                            'first_price': first_price,
                            'last_price': last_price,
                            'pct_change': pct_change,
                            'direction': direction,
                            'file_size': file_size
                        }
                        
                        all_predictions.append(prediction_info)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Sort by modification date (newest first)
    all_predictions.sort(key=lambda x: x['mod_date'], reverse=True)
    
    # Print summary
    print("\nPrediction Summary (newest first):")
    print("-" * 100)
    print(f"{'File':<30} {'Date':<20} {'First $':<10} {'Last $':<10} {'Change':<10} {'Direction':<8}")
    print("-" * 100)
    
    for pred in all_predictions[:10]:  # Show top 10
        print(f"{pred['file']:<30} {pred['mod_date']:<20} ${pred['first_price']:<9.2f} ${pred['last_price']:<9.2f} {pred['pct_change']:<9.2f}% {pred['direction']:<8}")
    
    if len(all_predictions) > 10:
        print(f"... and {len(all_predictions) - 10} more predictions")
    
    # Calculate consensus
    if all_predictions:
        up_votes = sum(1 for p in all_predictions if p['direction'] == 'UP')
        down_votes = sum(1 for p in all_predictions if p['direction'] == 'DOWN')
        
        print("\nConsensus Direction:")
        total = len(all_predictions)
        print(f"UP: {up_votes} predictions ({up_votes/total*100:.1f}%)")
        print(f"DOWN: {down_votes} predictions ({down_votes/total*100:.1f}%)")
        
        consensus = "UP" if up_votes > down_votes else "DOWN"
        confidence = max(up_votes, down_votes) / total
        
        print(f"\nOverall consensus: {consensus} with {confidence*100:.1f}% confidence")
        
        # Trading signal
        if confidence >= 0.7:
            if consensus == 'UP':
                signal = "BUY - Strong upward trend consensus"
            else:
                signal = "SELL - Strong downward trend consensus"
        elif confidence >= 0.6:
            if consensus == 'UP':
                signal = "BUY - Moderate upward trend consensus"
            else:
                signal = "SELL - Moderate downward trend consensus"
        else:
            signal = "HOLD/NEUTRAL - Mixed signals or weak consensus"
            
        print(f"Trading signal: {signal}")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 