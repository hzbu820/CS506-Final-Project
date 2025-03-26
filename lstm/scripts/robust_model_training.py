"""
Robust Model Training Script - Handles dependency issues and ensures reliable execution
"""

import os
import sys
import time
import traceback
import argparse
from datetime import datetime
import importlib

# Add short delay to avoid import rush
time.sleep(0.5)

# Handle imports with care
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')  # Use non-interactive backend
except ImportError as e:
    print(f"Error importing core libraries: {e}")
    print("Please run scripts/fix_dependencies.py to resolve dependency issues")
    sys.exit(1)

# Add src directory to the Python path
sys.path.append("src")

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Robust LSTM Stock Price Prediction')
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL, MSFT)')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--sequence_length', type=int, default=14,
                        help='Number of days to use for sequence prediction (default: 14 for stability)')
    
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Number of hidden units in LSTM layer (default: 128)')
    
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimization (default: 0.001)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    
    parser.add_argument('--future_days', type=int, default=14,
                        help='Number of days to predict into the future (default: 14)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed error messages')
    
    return parser.parse_args()

def safely_import_stock_predictor():
    """Safely import StockPredictor with additional error handling"""
    try:
        # Add a small delay before importing
        time.sleep(0.5)
        from models.stock_predictor import StockPredictor
        return StockPredictor
    except ImportError as e:
        print(f"Error importing StockPredictor: {e}")
        print("Trying alternative import path...")
        
        try:
            time.sleep(0.5)
            from src.models.stock_predictor import StockPredictor
            return StockPredictor
        except ImportError as e2:
            print(f"Alternative import also failed: {e2}")
            print("Please check your project structure and dependencies")
            return None

def main():
    """
    Main function for robust stock price prediction
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("=" * 50)
    print(f"Robust Stock Price Prediction for {args.ticker}")
    print("=" * 50)
    print(f"Model parameters: sequence_length={args.sequence_length}, hidden_size={args.hidden_size}, layers={args.num_layers}")
    
    try:
        # Create output directories if they don't exist
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/figures', exist_ok=True)
        os.makedirs('outputs/predictions', exist_ok=True)
        
        # Safely import StockPredictor
        StockPredictor = safely_import_stock_predictor()
        if StockPredictor is None:
            print("Could not import StockPredictor. Exiting.")
            return 1
        
        # Initialize stock predictor
        print(f"Initializing StockPredictor for {args.ticker}...")
        predictor = StockPredictor(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            sequence_length=args.sequence_length
        )
        
        # Train model
        print(f"\nTraining LSTM model for {args.ticker}...")
        print(f"Training parameters: {args.epochs} epochs, learning rate {args.learning_rate}")
        
        history = predictor.train_model(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        
        # Predict future prices
        print(f"\nPredicting future prices for {args.future_days} days...")
        future_df = predictor.predict_future(days=args.future_days)
        
        # Save predictions to CSV with timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prediction_file = f"outputs/predictions/{args.ticker}_robust_{timestamp}.csv"
        future_df.to_csv(prediction_file, index=False)
        print(f"Future predictions saved to {prediction_file}")
        
        # Print summary safely with proper error handling
        print("\nPrediction Summary:")
        try:
            # Get the latest close price and handle pandas Series correctly
            latest_data = predictor.data_loader.processed_data.iloc[-1]
            if isinstance(latest_data, pd.Series):
                latest_close = float(latest_data['Close'])
            else:
                latest_close = float(latest_data.loc['Close'])
            
            # Get the final predicted price
            future_close = float(future_df['Predicted_Price'].iloc[-1])
            
            # Calculate percentage change
            change_pct = ((future_close - latest_close) / latest_close) * 100
            
            # Print results
            print(f"Latest Close Price: ${latest_close:.2f}")
            print(f"Predicted Price ({args.future_days} days): ${future_close:.2f}")
            print(f"Predicted Change: {change_pct:.2f}%")
            print(f"Prediction Direction: {'UP' if change_pct > 0 else 'DOWN'}")
            
            # Print confidence based on validation loss
            val_loss = min(history['val_loss']) if history and 'val_loss' in history else None
            if val_loss is not None:
                if val_loss < 0.01:
                    confidence = "High"
                elif val_loss < 0.05:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                print(f"Model confidence: {confidence} (validation loss: {val_loss:.4f})")
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            if args.debug:
                traceback.print_exc()
        
        print("\nTraining and prediction completed successfully!")
        print(f"Results saved to {prediction_file}")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            print("\nDetailed traceback:")
            traceback.print_exc()
        print("\nTry again with --debug flag for more detailed error information.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 