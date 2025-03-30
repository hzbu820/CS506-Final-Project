# AAPL Stock Prediction Models

This project implements various LSTM-based deep learning models for AAPL stock price and direction prediction using 15-minute interval data.

## Project Organization

```
├── data/                  # Data files for model training and testing
├── docs/                  # Documentation files
├── models/                # Model implementation files
│   ├── aapl_lstm_predictor.py   # Price prediction model
│   ├── aapl_lstm_classifier.py  # Direction prediction model
│   └── basic_stock_direction.py # Basic direction model
├── notebooks/             # Jupyter notebooks for analysis
├── outputs/               # Model outputs, saved models, and visualizations
│   └── models/            # Saved trained models
├── scripts/               # Scripts for running models and utilities
│   └── run_models.py      # Main script for running all models
├── src/                   # Source code for utils and data processing
│   ├── data/              # Data loading utilities
│   ├── models/            # Model components and architectures
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization utilities
├── tests/                 # Test files
├── visualizations/        # Visualization scripts
│   └── plot_predictions.py # Prediction visualization
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Models

The project includes the following models:

1. **Price Prediction Model**: An LSTM model that predicts the actual price of AAPL stock for the next 15-minute interval.
   - Uses bidirectional LSTM layers with attention mechanism
   - Includes technical indicators like RSI, MACD, Bollinger Bands

2. **Direction Classification Model**: An LSTM model that predicts the direction (up/down) of the AAPL stock.
   - Uses focal loss to handle class imbalance
   - Includes relative strength indicators and market regime detection

3. **Ensemble Model**: A combination of different model architectures (LSTM, GRU, CNN-LSTM) for improved direction prediction.
   - Uses model averaging for final predictions
   - Each model is trained with different hyperparameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aapl-stock-prediction.git
cd aapl-stock-prediction

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Models

You can run all models using the main script:

```bash
python scripts/run_models.py --all
```

Or run specific models:

```bash
# Run price prediction model
python scripts/run_models.py --price

# Run direction prediction model
python scripts/run_models.py --direction

# Run ensemble model
python scripts/run_models.py --ensemble

# Generate visualizations
python scripts/run_models.py --visualize
```

### Model Performance

- **Price Prediction**: RMSE of 1.88, MAE of 1.53, R² of 0.82, MAPE of 0.63%
- **Direction Classification**: F1-score of 0.67 for the "Down" class, 0.18 for the "Up" class
- **Ensemble Model**: Improved F1 scores and better balanced predictions between classes

## Visualizations

The project generates several visualizations:

1. **Price Prediction**: Actual vs. predicted price chart
2. **Direction Prediction**: Price chart with up/down prediction markers
3. **Ensemble Predictions**: Distribution of predictions and probability visualization

## Future Improvements

- Expand training data to capture more market conditions
- Implement attention mechanisms for better feature importance
- Optimize hyperparameters using automated search
- Add sentiment analysis features from financial news
- Implement market regime detection

## License

This project is licensed under the MIT License - see the LICENSE file for details. 