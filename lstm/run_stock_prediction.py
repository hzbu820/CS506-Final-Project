import argparse
import os
import sys

# Add src directory to the Python path
sys.path.append("src")

from models.stock_predictor import StockPredictor

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Stock Price Prediction with LSTM')
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL, MSFT)')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD)')
    
    parser.add_argument('--sequence_length', type=int, default=20,
                        help='Number of days to use for sequence prediction')
    
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Number of hidden units in LSTM layer')
    
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimization')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    parser.add_argument('--future_days', type=int, default=30,
                        help='Number of days to predict into the future')
    
    return parser.parse_args()

def main():
    """
    Main function for stock price prediction
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("=" * 50)
    print(f"Stock Price Prediction for {args.ticker}")
    print("=" * 50)
    
    # Create output directories if they don't exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/predictions', exist_ok=True)
    
    # Initialize stock predictor
    predictor = StockPredictor(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        sequence_length=args.sequence_length
    )
    
    # Train model
    print(f"\nTraining LSTM model for {args.ticker}...")
    predictor.train_model(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    # Predict future prices
    print(f"\nPredicting future prices for {args.future_days} days...")
    future_df = predictor.predict_future(days=args.future_days)
    
    # Save predictions to CSV
    predictions_file = f"outputs/predictions/{args.ticker}_future_predictions.csv"
    future_df.to_csv(predictions_file, index=False)
    print(f"Future predictions saved to {predictions_file}")
    
    # Print summary
    print("\nPrediction Summary:")
    latest_close = float(predictor.data_loader.processed_data['Close'].iloc[-1].item())
    future_close = float(future_df['Predicted_Price'].iloc[-1])
    change_pct = (future_close - latest_close) / latest_close * 100
    
    print(f"Latest Close Price: ${latest_close:.2f}")
    print(f"Predicted Price ({args.future_days} days): ${future_close:.2f}")
    print(f"Predicted Change: {change_pct:.2f}%")
    print(f"Prediction Direction: {'UP' if change_pct > 0 else 'DOWN'}")
    
    print("\nAll charts and predictions have been saved to the outputs directory.")
    print("=" * 50)

if __name__ == "__main__":
    main() 