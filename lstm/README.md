# LSTM Stock Price Prediction System

A comprehensive implementation of an LSTM (Long Short-Term Memory) neural network specifically optimized for stock price prediction. This system downloads real stock data, generates technical indicators, trains a deep learning model, and forecasts future stock prices.

│
├── src/                         # Source code
│   ├── __init__.py              # Main package initialization
│   ├── models/                  # Model definitions
│   │   ├── __init__.py          # Models module initialization
│   │   ├── lstm_model.py        # LSTM model implementation
│   │   └── stock_predictor.py   # Stock prediction wrapper
│   ├── data/                    # Data handling
│   │   ├── __init__.py          # Data module initialization
│   │   ├── stock_data_loader.py # Stock data loading/processing
│   │   └── load_custom_data.py  # Custom data import utilities
│   ├── utils/                   # Utilities
│   │   ├── __init__.py          # Utilities module initialization
│   │   └── trainer.py           # LSTM training utilities
│   └── visualization/           # Visualization tools
│       └── __init__.py          # Visualization module initialization
│
├── examples/                    # Example scripts
│   ├── __init__.py              # Examples initialization
│   └── custom_data_example.py   # Custom data usage example
│
├── outputs/                     # Output files
│   ├── models/                  # Saved models
│   │   └── best_model.pth       # Best trained model
│   ├── figures/                 # Generated figures
│   │   ├── *_stock_history.png  # Stock history visualizations
│   │   ├── *_training_history.png # Training history 
│   │   ├── *_predictions.png    # Prediction visualizations
│   │   └── *_future_predictions.png # Future predictions
│   └── predictions/             # Prediction results
│       └── *_future_predictions.csv # Future prediction data
│
├── run_stock_prediction.py      # Main execution script
└── requirements.txt             # Project dependencies

## Installation

1. Clone this repository to your local machine
2. Install the required dependencies:

```bash
python -m pip install torch numpy pandas scikit-learn matplotlib yfinance tqdm seaborn
```

## Quick Start

To predict stock prices for a specific company, use the command-line interface:

```bash
python run_stock_prediction.py --ticker AAPL
```

This will:
1. Download 5 years of historical data for Apple Inc.
2. Calculate technical indicators and prepare sequences
3. Train an LSTM model on the data
4. Make predictions for the next 30 days
5. Generate visualizations and save results in the `results` directory

## Command-Line Options

The system supports various command-line parameters for customization:

```
--ticker SYMBOL       Stock ticker symbol (e.g., AAPL, MSFT) [required]
--start_date DATE     Start date for historical data (YYYY-MM-DD) [default: 5 years ago]
--end_date DATE       End date for historical data (YYYY-MM-DD) [default: today]
--sequence_length N   Number of days to use for sequence prediction [default: 20]
--hidden_size N       Number of hidden units in LSTM layer [default: 128]
--num_layers N        Number of LSTM layers [default: 2]
--learning_rate R     Learning rate for optimization [default: 0.001]
--epochs N            Number of training epochs [default: 100]
--future_days N       Number of days to predict into the future [default: 30]
```

## Usage Examples

### Basic Usage

```bash
# Predict Apple stock prices
python run_stock_prediction.py --ticker AAPL

# Predict Microsoft stock with 50 days of future predictions
python run_stock_prediction.py --ticker MSFT --future_days 50

# Use more historical data for Tesla
python run_stock_prediction.py --ticker TSLA --start_date 2015-01-01
```

### Usage Scenarios

**For short-term prediction (1 week):**
```bash
python run_stock_prediction.py --ticker AAPL --future_days 7 --sequence_length 10
```

**For long-term prediction (3 months):**
```bash
python run_stock_prediction.py --ticker AAPL --future_days 90 --sequence_length 60
```

**For more accurate predictions (with longer training):**
```bash
python run_stock_prediction.py --ticker AAPL --epochs 200 --hidden_size 256
```

**For volatile stocks (with more complex patterns):**
```bash
python run_stock_prediction.py --ticker TSLA --hidden_size 256 --num_layers 3
```

### Advanced Model Configuration

```bash
# Use a deeper LSTM model for more complex patterns
python run_stock_prediction.py --ticker NVDA --num_layers 3 --hidden_size 256

# More thorough training for cryptocurrencies
python run_stock_prediction.py --ticker BTC-USD --epochs 200 --learning_rate 0.0005
```

## Output Files

All results are saved in the project directory and the `results` subfolder:

- `TICKER_stock_history.png`: Historical stock price chart with moving averages and volume
- `TICKER_training_history.png`: Training and validation loss curves
- `TICKER_predictions.png`: Test set predictions vs actual prices
- `TICKER_future_predictions.png`: Future price predictions chart
- `results/TICKER_future_predictions.csv`: CSV file with predicted prices and dates
- `best_model.pth`: Saved model weights for future use

## Understanding the Prediction Quality

The system outputs several metrics to evaluate prediction quality:

- **MSE (Mean Squared Error)**: Measures the average squared difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same units as the stock price
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **Direction Accuracy**: Percentage of correctly predicted price movements (up/down)

A good model should have:
- Low MSE, RMSE, and MAE values
- Direction accuracy consistently above 50% (better than random guessing)

## Using the API in Custom Scripts

For more advanced usage, you can import the `StockPredictor` class directly in your scripts:

```python
from stock_predictor import StockPredictor

# Initialize predictor for Amazon
predictor = StockPredictor(ticker='AMZN')

# Train the model
predictor.train_model(
    hidden_size=128,
    num_layers=2,
    learning_rate=0.001,
    epochs=100
)

# Predict future prices for 30 days
future_prices = predictor.predict_future(days=30)

# Print predictions
print(future_prices)
```

## Technical Implementation Details

### Data Processing

The system uses the `StockDataLoader` class to:
1. Download stock data from Yahoo Finance
2. Calculate technical indicators including:
   - Moving averages (5, 10, 20, and 50-day)
   - Price momentum (returns)
   - Volatility measures
   - Trading ranges
   - Volume indicators
   - Price relative to moving averages
   - MACD (Moving Average Convergence/Divergence)
3. Normalize data using Min-Max scaling
4. Create sequence-based training examples

### Model Architecture

The stock prediction model uses a stacked LSTM architecture:

1. **Input Layer**: Takes sequences of stock data with 14 features
2. **LSTM Layers**: Multiple LSTM layers with dropout for regularization
3. **Fully Connected Layer**: Projects LSTM output to the predicted stock price

The model architecture is defined in `lstm_model.py` and can be customized through command-line parameters.

### Training Process

The training process (managed by `trainer.py`):
1. Uses Mean Squared Error (MSE) loss
2. Employs the Adam optimizer
3. Implements early stopping to prevent overfitting
4. Saves the best model based on validation performance
5. Provides visualization of the training progress

## Best Practices and Recommendations

1. **Data Quality**: Use at least 3-5 years of historical data for better results
2. **Sequence Length**: 
   - 10-20 days for short-term patterns
   - 30-60 days for longer-term cycles
3. **Model Size**: 
   - 1-2 layers and 64-128 hidden units for stable stocks
   - 2-3 layers and 128-256 hidden units for volatile stocks
4. **Training Duration**:
   - 50-100 epochs for initial testing
   - 100-200 epochs for final models with early stopping
5. **Prediction Horizon**:
   - 1-10 days: Higher accuracy
   - 10-30 days: Moderate accuracy
   - 30+ days: Lower accuracy, use with caution

## Limitations and Considerations

- Stock markets are influenced by factors that cannot be predicted from historical data alone (news, economic events, etc.)
- The model cannot predict unexpected events like earnings surprises or macroeconomic shocks
- Past performance is not indicative of future results
- Consider using predictions as one of several inputs for investment decisions, not the sole basis
- Models perform best during stable market conditions and may struggle during extreme volatility

## Troubleshooting

If you encounter issues:

1. **Data Download Problems**: Check internet connection and verify ticker symbol exists on Yahoo Finance
2. **Memory Errors**: Reduce batch size or sequence length
3. **Training Instability**: Lower the learning rate (try 0.0005 or 0.0001)
4. **Poor Predictions**: Try increasing historical data, adjusting sequence length, or adding more layers

## Extending the System

The modular design allows for easy extensions:
- Add new technical indicators in `stock_data_loader.py`
- Implement different model architectures in `lstm_model.py`
- Create new evaluation metrics in `stock_predictor.py`
- Add additional visualization tools as needed 