import argparse
import os
import sys
import glob
import traceback
from datetime import datetime, timedelta

# Add src directory to the Python path
sys.path.append("src")

from models.stock_predictor import StockPredictor

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Continue Training LSTM Stock Prediction Model')
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL, MSFT)')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a previously trained model file. If not provided, will use the most recent model.')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for new training data (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for new training data (YYYY-MM-DD)')
    
    parser.add_argument('--sequence_length', type=int, default=None,
                        help='Number of days to use for sequence prediction (default: same as original model)')
    
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for fine-tuning (default: 0.0005)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    
    parser.add_argument('--future_days', type=int, default=30,
                        help='Number of days to predict into the future')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed error messages')
    
    return parser.parse_args()

def find_latest_model_file(ticker):
    """
    Find the most recent model file for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        model_path: Path to the most recent model file
    """
    models_dir = 'outputs/models'
    
    # Find all model files for this ticker
    model_files = glob.glob(f"{models_dir}/{ticker}_model_*.pth")
    
    if not model_files:
        print(f"No existing model found for {ticker}. Please train a model first.")
        return None
    
    # Sort by modification time (most recent first)
    latest_model = max(model_files, key=os.path.getmtime)
    
    return latest_model

def extract_model_parameters(model_path):
    """
    Extract model parameters from filename
    
    Args:
        model_path: Path to the model file
        
    Returns:
        sequence_length: Sequence length used in the model
    """
    # Default values if we can't extract from filename
    sequence_length = 30  # Enhanced default
    
    # For now, we'll just use defaults
    # In the future, we could store model parameters in a config file
    
    return sequence_length

def main():
    """
    Main function for continued training
    """
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Print header
        print("=" * 50)
        print(f"Continued Training for {args.ticker} Stock Prediction Model")
        print("=" * 50)
        
        # Find the latest model if not specified
        if not args.model_path:
            args.model_path = find_latest_model_file(args.ticker)
            
        if not args.model_path:
            print("Could not find model files. Please train a model first or specify the path.")
            return
        
        # Print info about the model being used
        print(f"Using model: {args.model_path}")
        
        # Extract model parameters if not specified
        if not args.sequence_length:
            args.sequence_length = extract_model_parameters(args.model_path)
            print(f"Using sequence length: {args.sequence_length}")
        
        # Setup default dates if not provided
        if not args.start_date:
            # If no start date, use 6 months of data up to today
            args.start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
        if not args.end_date:
            # If no end date, use today
            args.end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Training with new data from {args.start_date} to {args.end_date}")
        
        # Create output directories if they don't exist
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/figures', exist_ok=True)
        os.makedirs('outputs/predictions', exist_ok=True)
        
        # Initialize predictor with existing model
        predictor = StockPredictor(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            sequence_length=args.sequence_length
        )
        
        # Prepare data first to initialize the data_loader
        print("Preparing data...")
        train_loader, test_loader, test_X, test_y = predictor.prepare_data()
        
        # Now load the model (after data_loader is initialized)
        print(f"Loading model from {args.model_path}...")
        predictor.model = predictor.load_model(args.model_path)
        
        # Continue training the model (standard train_model but with different learning rate)
        print(f"\nContinuing training for {args.ticker} with {args.epochs} epochs...")
        print(f"Using learning rate: {args.learning_rate} (reduced for fine-tuning)")
        
        # Use trainer with lower learning rate for fine-tuning
        from utils.trainer import LSTMTrainer
        trainer = LSTMTrainer(predictor.model, learning_rate=args.learning_rate)
        
        # Train model with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_path = f'outputs/models/{args.ticker}_model_continued_{timestamp}.pth'
        
        history = trainer.train(
            train_loader, 
            test_loader, 
            epochs=args.epochs, 
            model_save_path=model_save_path
        )
        
        # Store the latest model path
        predictor.latest_model_path = model_save_path
        
        # Plot training history
        predictor.plot_training_history(history)
        
        # Evaluate model
        predictor.evaluate_model(test_X, test_y)
        
        # Predict future prices
        print(f"\nPredicting future prices for {args.future_days} days...")
        future_df = predictor.predict_future(days=args.future_days)
        
        # Save predictions to CSV
        predictions_file = f"outputs/predictions/{args.ticker}_continued_predictions.csv"
        future_df.to_csv(predictions_file, index=False)
        print(f"Future predictions saved to {predictions_file}")
        
        # Print summary
        print("\nPrediction Summary After Continued Training:")
        latest_close = float(predictor.data_loader.processed_data['Close'].iloc[-1].item())
        future_close = float(future_df['Predicted_Price'].iloc[-1])
        
        # Download current real price for comparison
        try:
            import yfinance as yf
            current_data = yf.download(args.ticker, period="5d")
            real_close = float(current_data['Close'].iloc[-1])
            print(f"Latest Real Market Close Price: ${real_close:.2f}")
            print(f"Model's Internal Close Price: ${latest_close:.2f}")
            
            # If there's a large discrepancy, use the real price for percentage calculation
            if abs((real_close - latest_close) / real_close) > 0.5:  # If more than 50% different
                print("NOTE: Using real market price for percentage calculation due to scaling difference")
                latest_close = real_close
        except:
            print("Could not fetch real-time price data for comparison")
        
        # Calculate percentage change
        change_pct = (future_close - latest_close) / latest_close * 100
        
        print(f"Predicted Price ({args.future_days} days): ${future_close:.2f}")
        print(f"Predicted Change: {change_pct:.2f}%")
        print(f"Prediction Direction: {'UP' if change_pct > 0 else 'DOWN'}")
        
        print("\nAll charts and predictions have been saved to the outputs directory.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            print("\nDetailed traceback:")
            traceback.print_exc()
        print("\nTry again with --debug flag for more detailed error information.")

if __name__ == "__main__":
    main() 