#!/usr/bin/env python
"""
Run Models Script
Main entry point for running AAPL stock prediction models
"""

import os
import sys
import argparse
from datetime import datetime

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def log_message(message):
    """Print a message with a timestamp and also write to a log file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Ensure the outputs directory exists
    os.makedirs(os.path.join(project_root, 'outputs'), exist_ok=True)
    
    # Write to log file
    with open(os.path.join(project_root, 'outputs', 'run_log.txt'), 'a') as f:
        f.write(log_msg + '\n')

def run_price_prediction():
    """Run the AAPL price prediction model"""
    log_message("Starting AAPL price prediction model...")
    try:
        from models.aapl_lstm_predictor import train_price_prediction_model
        
        # Run the price prediction model
        model, feature_columns, scaler_x, scaler_y = train_price_prediction_model(
            ticker="AAPL",
            sequence_length=60,
            hidden_size=128,
            num_layers=2,
            learning_rate=0.0005,
            batch_size=32,
            epochs=100,
            train_split=0.8
        )
        
        log_message("Price prediction model completed successfully!")
        return True
    except Exception as e:
        log_message(f"Error running price prediction model: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False

def run_direction_prediction():
    """Run the AAPL direction prediction model (classifier)"""
    log_message("Starting AAPL direction prediction model...")
    try:
        from models.aapl_lstm_classifier import train_direction_model
        
        # Run the direction classification model
        model, feature_columns, scaler, threshold = train_direction_model(
            ticker="AAPL",
            sequence_length=60,
            hidden_size=64,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,
            train_split=0.8
        )
        
        log_message("Direction prediction model completed successfully!")
        return True
    except Exception as e:
        log_message(f"Error running direction prediction model: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False

def run_ensemble_model():
    """Run the ensemble model for direction prediction"""
    log_message("Starting ensemble model training...")
    try:
        from models.aapl_lstm_classifier import train_ensemble_model
        
        # Run the ensemble model training
        ensemble_meta = train_ensemble_model(
            ticker="AAPL",
            sequence_length=60,
            train_split=0.8,
            n_models=3
        )
        
        log_message("Ensemble model training completed successfully!")
        return True
    except Exception as e:
        log_message(f"Error running ensemble model: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False

def run_visualization():
    """Run the visualization scripts"""
    log_message("Generating visualizations...")
    try:
        from visualizations.plot_predictions import plot_predictions
        
        # Generate prediction visualization
        plot_predictions()
        
        log_message("Visualization completed successfully!")
        return True
    except Exception as e:
        log_message(f"Error generating visualization: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False

def main():
    """Main function to parse arguments and run models"""
    parser = argparse.ArgumentParser(description='Run AAPL stock prediction models')
    parser.add_argument('--price', action='store_true', help='Run price prediction model')
    parser.add_argument('--direction', action='store_true', help='Run direction prediction model')
    parser.add_argument('--ensemble', action='store_true', help='Run ensemble model')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--all', action='store_true', help='Run all models and visualizations')
    
    args = parser.parse_args()
    
    # If no args or --all, run everything
    if len(sys.argv) == 1 or args.all:
        log_message("Running all models and visualizations...")
        run_price_prediction()
        run_direction_prediction()
        run_ensemble_model()
        run_visualization()
    else:
        # Run specified models
        if args.price:
            run_price_prediction()
        if args.direction:
            run_direction_prediction()
        if args.ensemble:
            run_ensemble_model()
        if args.visualize:
            run_visualization()
    
    log_message("All processes completed!")

if __name__ == "__main__":
    main() 