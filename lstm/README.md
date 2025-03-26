# LSTM Stock Price Prediction System

This repository contains an advanced LSTM-based stock price prediction system that combines multiple prediction models, ensemble forecasting, and trading signals generation.


## Output Files

This repository includes output files from our experiments:

- **Figures**: Visualizations of predictions, model performance, and technical indicators
- **Predictions**: CSV files containing prediction data
- **Ensemble Analysis**: Results from our ensemble prediction approaches
- **Normalized Ensemble**: Results from normalized ensemble methods that address scaling issues

These output files demonstrate the model's performance and provide examples of the system's capabilities.
## Features

- LSTM-based stock price prediction models with different architectures
- Enhanced model training with optimized hyperparameters
- Continuous training capability to update models with new data
- Intraday trading advisor with technical indicators (RSI, MACD)
- Ensemble prediction from multiple models with normalization
- Direction-based voting system for generating trading signals
- Comprehensive visualization of predictions and model performance

## Repository Organization

```
├── src/                  # Source code modules
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # LSTM model implementations
│   ├── visualization/    # Visualization utilities
│   └── utils/            # Helper functions
├── scripts/              # Script files for training and prediction
├── outputs/              # Output directories (created during execution)
│   ├── models/           # Trained models
│   ├── predictions/      # Prediction CSV files
│   ├── figures/          # Visualization outputs
│   └── ensemble_analysis/# Ensemble analysis results
└── sample_models/        # Sample pre-trained model
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/lstm-stock-prediction.git
   cd lstm-stock-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create required directories (if not already present):
   ```
   python prepare_for_github.py
   ```

## Usage Examples

### Training a new model:
```
python scripts/train_new_model.py --ticker AAPL --epochs 100 --sequence_length 30
```

### Intraday prediction:
```
python scripts/train_intraday_model.py --ticker AAPL --interval 5m --period 5d
```

### Generate normalized ensemble prediction:
```
python scripts/normalized_ensemble.py --ticker AAPL
```

### Get trading signals:
```
python scripts/generate_trading_signals.py --ticker AAPL
```

## Model Files

Due to size constraints, trained model files are not included in this repository. When you run the training scripts, models will be saved in the `outputs/models/` directory.

A sample pre-trained model is included in the `sample_models/` directory for reference.

## Notes for Contributors

- Before committing, run `prepare_for_github.py` to organize the repository structure
- Don't commit large model files or prediction outputs
- Add new scripts to the `scripts/` directory
- Implement core functionality in the `src/` modules

## License

This project is licensed under the MIT License - see the LICENSE file for details. 