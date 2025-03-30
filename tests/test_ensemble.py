import os
import sys
import numpy as np
import pandas as pd
import torch
from aapl_lstm_classifier import predict_with_ensemble, train_ensemble_model, log_message

def main():
    """Test the ensemble model for AAPL stock direction prediction"""
    print("====== Testing AAPL Stock Direction Ensemble Prediction ======")
    
    # Check if ensemble models exist
    ensemble_dir = 'outputs/models/ensemble'
    if not os.path.exists(ensemble_dir):
        print(f"Ensemble directory {ensemble_dir} not found. Training new ensemble...")
        # Train ensemble with 3 models
        train_ensemble_model(n_models=3)
    else:
        # Check if there are any model files
        import glob
        model_files = glob.glob(f"{ensemble_dir}/model_*.pth")
        if not model_files:
            print(f"No ensemble models found in {ensemble_dir}. Training new ensemble...")
            # Train ensemble with 3 models
            train_ensemble_model(n_models=3)
    
    # Make prediction with ensemble
    print("\nMaking predictions with ensemble model...")
    results = predict_with_ensemble()
    
    if results is not None:
        # Display the most recent predictions
        last_n = min(10, len(results))
        print(f"\nLast {last_n} predictions:")
        recent_results = results.tail(last_n)
        
        # Format the output for better display
        for i, row in recent_results.iterrows():
            direction = "UP" if row['prediction'] == 1 else "DOWN"
            print(f"{row['timestamp']}: {direction} ({row['probability']:.4f})")
        
        # Print distribution
        up_count = results['prediction'].sum()
        total = len(results)
        up_pct = up_count / total * 100 if total > 0 else 0
        print(f"\nPrediction distribution: UP={up_count} ({up_pct:.1f}%), DOWN={total - up_count}")
    else:
        print("Prediction failed or no results returned")

if __name__ == "__main__":
    main() 