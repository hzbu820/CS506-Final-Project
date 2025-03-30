#!/usr/bin/env python
"""
Plot AAPL Classifier Predictions
Visualizes the directional predictions from the AAPL LSTM Classifier
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from aapl_lstm_classifier import predict_with_ensemble, get_stock_data, log_message

def plot_predictions():
    """Plot the directional predictions from the AAPL classifier"""
    log_message("Generating prediction visualization for AAPL classifier")
    
    # Get prediction results
    results = predict_with_ensemble()
    
    if results is None or len(results) == 0:
        log_message("No prediction results available to plot")
        return
    
    # Get the original price data to overlay
    df = get_stock_data(ticker="AAPL", timeframe='15m')
    
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Price chart with prediction markers
    ax1 = plt.subplot(2, 1, 1)
    
    # Get the date range that matches our predictions
    start_idx = max(0, len(df) - len(results))
    prices = df['Close'].iloc[start_idx:].values
    dates = df.index[start_idx:]
    
    # Plot the price line
    ax1.plot(dates, prices, color='gray', alpha=0.5, label='AAPL Price')
    
    # Plot prediction points - green for UP, red for DOWN
    up_indices = results['prediction'] == 1
    down_indices = results['prediction'] == 0
    
    # For visual clarity, scale the probabilities to marker size
    marker_sizes = results['probability'] * 100
    
    # Plot UP predictions
    if any(up_indices):
        ax1.scatter(
            results.loc[up_indices, 'timestamp'], 
            prices[up_indices], 
            color='green', 
            label='UP Prediction',
            s=marker_sizes[up_indices],
            zorder=10,
            alpha=0.7
        )
    
    # Plot DOWN predictions
    if any(down_indices):
        ax1.scatter(
            results.loc[down_indices, 'timestamp'], 
            prices[down_indices], 
            color='red', 
            label='DOWN Prediction',
            s=marker_sizes[down_indices],
            zorder=10,
            alpha=0.7
        )
    
    # Format the price chart
    ax1.set_title('AAPL Price with Direction Predictions')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    
    # Plot 2: Prediction probabilities
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Plot probability bars - color coded by prediction
    bar_colors = ['red' if pred == 0 else 'green' for pred in results['prediction']]
    ax2.bar(results['timestamp'], results['probability'], color=bar_colors, alpha=0.7)
    
    # Add threshold line
    try:
        import json
        with open('outputs/models/ensemble/ensemble_meta.json', 'r') as f:
            meta = json.load(f)
        threshold = meta.get('ensemble_threshold', 0.35)
    except:
        threshold = 0.35  # Default if file not found
        
    ax2.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
    
    # Format the probability chart
    ax2.set_title('Prediction Probabilities')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('UP Probability')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add distribution information
    up_count = sum(results['prediction'] == 1)
    down_count = sum(results['prediction'] == 0)
    total = len(results)
    up_pct = up_count / total * 100 if total > 0 else 0
    
    # Add text with distribution info
    plt.figtext(
        0.5, 0.01, 
        f"Distribution: UP={up_count}/{total} ({up_pct:.1f}%), DOWN={down_count}/{total} ({100-up_pct:.1f}%)",
        ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5}
    )
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.gcf().autofmt_xdate()
    plt.savefig('outputs/aapl_classifier_predictions.png')
    log_message("Prediction visualization saved to outputs/aapl_classifier_predictions.png")
    
    # Show latest prediction
    if len(results) > 0:
        latest = results.iloc[-1]
        direction = "UP" if latest['prediction'] == 1 else "DOWN"
        log_message(f"Latest prediction ({latest['timestamp']}): {direction} with probability {latest['probability']:.4f}")
    
    plt.close()

if __name__ == "__main__":
    plot_predictions() 