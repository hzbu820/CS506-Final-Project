# Enhanced LSTM Stock Price Prediction System

This is an upgraded version of the LSTM Stock Price Prediction System with improved accuracy and support for continued training with new data.

## Enhanced Features

- **Increased Model Capacity**: Uses larger hidden layers (256 units) and deeper networks (3 layers)
- **Improved Sequence Context**: Uses 30 days of context instead of 20 for better pattern recognition
- **Better Regularization**: Uses dropout of 0.4 to prevent overfitting
- **Continuous Training**: Support for continuing training with newer data
- **Model Versioning**: All models are saved with timestamps for better tracking

## Usage Instructions

### Training a New Enhanced Model

```
python scripts/train_new_model.py --ticker AAPL
```

This will:
- Download and process historical data for AAPL
- Create a larger LSTM model (256 hidden units, 3 layers)
- Train for 150 epochs with a learning rate of 0.001
- Predict prices for the next 30 days
- Save the model with a timestamp
- Generate visualizations

You can customize the training process with these parameters:
```
python scripts/train_new_model.py --ticker MSFT --epochs 200 --start_date 2018-01-01 --future_days 60
```

### Continuing Training with New Data

```
python scripts/update_existing_model.py --ticker AAPL
```

This will:
- Load the latest trained model for AAPL
- Download the most recent market data
- Continue training for 50 epochs with a lower learning rate (0.0005)
- Update predictions with new market information
- Save an updated model version

Custom parameters:
```
python scripts/update_existing_model.py --ticker AAPL --model_path outputs/models/AAPL_model_20230101_120000.pth --epochs 30
```

## Output Files

All outputs are saved in the `outputs` directory:

- **Models**: `outputs/models/TICKER_model_TIMESTAMP.pth`
- **Continued Models**: `outputs/models/TICKER_model_continued_TIMESTAMP.pth` 
- **Figures**: `outputs/figures/TICKER_*.png`
- **Predictions**: `outputs/predictions/TICKER_*.csv`

## Recommended Workflow

For best results:

1. Train an initial enhanced model:
   ```bash
   python scripts/train_new_model.py --ticker AAPL --epochs 150
   ```

2. Update the model weekly or monthly with new data:
   ```bash
   python scripts/update_existing_model.py --ticker AAPL --epochs 50
   ```

3. For greater accuracy with volatile stocks, use more hidden layers:
   ```bash
   python scripts/train_new_model.py --ticker TSLA --hidden_size 512 --num_layers 4
   ``` 