import argparse
import os
import sys
import traceback
from datetime import datetime

# Add src directory to the Python path
sys.path.append("src")

from models.stock_predictor import StockPredictor

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Enhanced LSTM Stock Price Prediction with Real Data')
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL, MSFT)')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Number of days to use for sequence prediction (default: 30 for more context)')
    
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Number of hidden units in LSTM layer (default: 256 for more capacity)')
    
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of LSTM layers (default: 3 for deeper network)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimization (default: 0.001)')
    
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs (default: 150 for more thorough training)')
    
    parser.add_argument('--future_days', type=int, default=30,
                        help='Number of days to predict into the future (default: 30)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed error messages')
    
    return parser.parse_args()

def main():
    """
    Main function for enhanced stock price prediction
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("=" * 50)
    print(f"Enhanced Stock Price Prediction for {args.ticker}")
    print("=" * 50)
    print("Training a new model with enhanced parameters for more accurate predictions")
    
    try:
        # Create output directories if they don't exist
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/figures', exist_ok=True)
        os.makedirs('outputs/predictions', exist_ok=True)
        
        # Initialize stock predictor with improved sequence length
        print(f"Initializing StockPredictor for {args.ticker} with sequence_length={args.sequence_length}...")
        predictor = StockPredictor(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            sequence_length=args.sequence_length
        )
        
        # Train model with enhanced parameters
        print(f"\nTraining enhanced LSTM model for {args.ticker}...")
        print(f"Hidden Size: {args.hidden_size}, Layers: {args.num_layers}, Epochs: {args.epochs}")
        
        history = predictor.train_model(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        
        # Predict future prices
        print(f"\nPredicting future prices for {args.future_days} days...")
        future_df = predictor.predict_future(days=args.future_days)
        
        # Save predictions to CSV
        predictions_file = f"outputs/predictions/{args.ticker}_enhanced_predictions.csv"
        future_df.to_csv(predictions_file, index=False)
        print(f"Future predictions saved to {predictions_file}")
        
        # Print summary
        print("\nEnhanced Prediction Summary:")
        try:
            # Handle Series correctly by using .item() for scalar values
            latest_close = float(predictor.data_loader.processed_data['Close'].iloc[-1].item())
            future_close = float(future_df['Predicted_Price'].iloc[-1])
            change_pct = (future_close - latest_close) / latest_close * 100
            
            print(f"Latest Close Price: ${latest_close:.2f}")
            print(f"Predicted Price ({args.future_days} days): ${future_close:.2f}")
            print(f"Predicted Change: {change_pct:.2f}%")
            print(f"Prediction Direction: {'UP' if change_pct > 0 else 'DOWN'}")
        except Exception as e:
            print(f"Error calculating summary statistics: {str(e)}")
            print(f"Latest data shape: {predictor.data_loader.processed_data.shape}")
            print(f"Future data shape: {future_df.shape}")
            
            # Alternative calculation if the above fails
            if not future_df.empty and not predictor.data_loader.processed_data.empty:
                try:
                    latest_row = predictor.data_loader.processed_data.iloc[-1]
                    latest_close = float(latest_row['Close'])
                    future_close = float(future_df['Predicted_Price'].iloc[-1])
                    change_pct = (future_close - latest_close) / latest_close * 100
                    
                    print(f"Latest Close Price (alt method): ${latest_close:.2f}")
                    print(f"Predicted Price ({args.future_days} days): ${future_close:.2f}")
                    print(f"Predicted Change: {change_pct:.2f}%")
                    print(f"Prediction Direction: {'UP' if change_pct > 0 else 'DOWN'}")
                except Exception as inner_e:
                    print(f"Alternative calculation also failed: {str(inner_e)}")
        
        print("\nAll charts and predictions have been saved to the outputs directory.")
        
        # Print next steps
        print("\nNext Steps:")
        print("1. To continue training this model with more recent data:")
        print(f"   python scripts/update_existing_model.py --ticker {args.ticker}")
        print("2. To predict with different parameters:")
        print(f"   python scripts/update_existing_model.py --ticker {args.ticker} --future_days 60")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            print("\nDetailed traceback:")
            traceback.print_exc()
        print("\nTry again with --debug flag for more detailed error information.")

if __name__ == "__main__":
    main() 