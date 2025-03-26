# LSTM Stock Prediction Project: Midterm Presentation Script

## Introduction (30 seconds)
Good afternoon everyone. Today I'll be presenting our LSTM Stock Prediction project, which aims to forecast stock price movements using deep learning. We've focused on Apple stock as our primary case study, with the goal of creating reliable trading signals through ensemble prediction methods.

## Data Visualization and Exploration (1 minute)
Our project utilizes historical stock data from Yahoo Finance, capturing daily price movements and trading volumes. 

We've implemented several visualization techniques to understand the data:
- Time series plots of historical stock prices, revealing long-term trends and volatility patterns
- Candlestick charts showing daily price movements with trading volumes
- Correlation heatmaps between different technical indicators
- Distribution analysis of daily returns to understand volatility characteristics

One key observation is the non-stationary nature of stock prices, which necessitated our normalization approaches to make the data suitable for LSTM modeling.

## Data Processing Pipeline (1 minute)
Our data processing pipeline consists of several critical steps:

1. **Data Acquisition**: We fetch historical data directly from Yahoo Finance using the yfinance API, giving us access to OHLCV (Open, High, Low, Close, Volume) data for any publicly traded stock.

2. **Preprocessing**: We've implemented robust preprocessing techniques including:
   - Handling missing values through forward-filling methods
   - Feature normalization using MinMaxScaler to constrain values between 0 and 1
   - Sequence generation for LSTM inputs, converting daily prices into overlapping sequences
   - Train/validation splitting with temporal considerations (earlier data for training, recent data for validation)

3. **Feature Engineering**: We've created several derived features including:
   - Moving averages at different time windows (5-day, 10-day, 20-day)
   - Price momentum indicators
   - Volatility measures
   - Trading volume normalization

## Modeling Methodology (1.5 minutes)
Our modeling approach centers on Long Short-Term Memory (LSTM) networks, which are particularly well-suited for time series prediction due to their ability to capture long-term dependencies.

We've implemented several model architectures:
1. **Basic LSTM**: Single-layer LSTM with sequence-to-one prediction
2. **Enhanced LSTM**: Multi-layer LSTM with dropout for regularization
3. **Bi-directional LSTM**: Capturing both forward and backward temporal dependencies
4. **Ensemble Methods**: Combining predictions from multiple models for improved reliability

Key hyperparameters we've optimized include:
- Sequence length (14 days providing optimal balance)
- Hidden layer size (128 units showing best performance)
- Number of LSTM layers (2 layers being optimal)
- Learning rate scheduling
- Dropout rates for regularization

We've also implemented early stopping based on validation loss to prevent overfitting.

## Preliminary Results (1 minute)
Our preliminary results are promising:

1. **Individual Model Performance**:
   - Validation loss (MSE) ranges from 0.0023 to 0.0087
   - Different models show varying degrees of sensitivity to market volatility

2. **Ensemble Results**:
   - Direction ensemble achieves 80.4% consensus on upward price movement for AAPL
   - Normalized ensemble effectively addresses scaling issues between models
   - Trading signals demonstrate reasonable alignment with actual market movements

3. **Current Trading Signal**:
   - For AAPL: BUY recommendation with 80.4% confidence
   - Predicted price movement of +1.94% over the next 14 days
   - Most short-term models predict slight increases (0.04% to 0.19%)
   - Enhanced models predict larger increases (up to +6.93%)

## Challenges and Next Steps (30 seconds)
We've encountered several challenges, including dependency issues with scientific libraries and timezone handling in ensemble creation. We've addressed these through robust error handling and lightweight analysis scripts.

Our next steps include:
1. Implementing backtesting to quantify prediction accuracy
2. Expanding to additional stocks for cross-sector analysis
3. Optimizing ensemble weights based on historical performance
4. Adding visualization tools for more intuitive interpretation

Thank you for your attention. I'm happy to answer any questions. 