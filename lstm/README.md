# LSTM Time Series Forecasting for Stock Prices

This directory contains an implementation of Long Short-Term Memory (LSTM) networks for predicting stock prices. The implementation uses PyTorch and demonstrates how to preprocess data, build and train an LSTM model, and visualize the results.

## Contents

- `LSTM.csv`: Sample stock price data with technical indicators
- `lstm_model.py`: Python script containing the LSTM implementation
- `lstm.ipynb`: (Future) Jupyter notebook for interactive exploration of the LSTM model

## Features

- Data loading and preprocessing
- Sequence creation for LSTM input
- LSTM model implementation with configurable parameters
- Training with early stopping
- Model evaluation and visualization
- Future price forecasting

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

### Running the Model

You can run the LSTM model script with:

```bash
cd CS506-Final-Project-main/lstm
python lstm_model.py
```

This will:
1. Load the sample data (or real data if available)
2. Preprocess the data and create sequences
3. Train the LSTM model
4. Generate predictions and visualize the results
5. Save plots to the current directory

## Model Architecture

The LSTM model in this implementation consists of:

- LSTM layers with configurable hidden size and number of layers
- Dropout for regularization
- A fully connected layer for final prediction

## Training and Evaluation

The model is trained using:
- Mean Squared Error (MSE) loss function
- Adam optimizer
- Early stopping based on validation loss
- 80/20 train/test split

Performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

## Comparison with ARIMA

Unlike the ARIMA model in the adjacent directory, LSTM networks can:
- Capture non-linear relationships in the data
- Consider multiple features simultaneously
- Remember long-term dependencies in the time series

This makes LSTM particularly well-suited for stock price prediction where complex patterns exist.

## Future Work

- Hyperparameter tuning
- Addition of attention mechanisms
- Incorporation of market sentiment data
- Implementation of bidirectional LSTMs 