# Stock Price Prediction Using Advanced Machine Learning Techniques


## Final Presentation Video
[Link to Stock Price Prediction with LSTM - Final Presentation](https://www.youtube.com/your-final-video-link-here)

## Table of Contents
- [Project Overview](#project-overview)
- [Technical Approach](#technical-approach)
- [Environment Setup](#environment-setup)
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
  - [ARIMA Baseline](#arima-baseline)
  - [LSTM Model](#lstm-model)
- [Training Process](#training-process)
- [Results and Visualization](#results-and-visualization)
- [Performance Analysis](#performance-analysis)
- [Testing](#testing)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)
- [How to Contribute](#how-to-contribute)
- [Supported Environments](#supported-environments)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project implements advanced machine learning techniques to predict stock price movements, aiming to outperform traditional "buy-and-hold" strategies. We focus on binary classification of stock trends (hold/sell signals) using LSTM (Long Short-Term Memory) neural networks, with ARIMA (AutoRegressive Integrated Moving Average) as our baseline model.

We specifically analyze the "Magnificent 7" tech stocks (Apple, Microsoft, Google, Amazon, Meta, Tesla, and Nvidia) using historical price data, trading volumes, and technical indicators. Our models predict whether to hold or sell a stock based on trend analysis, evaluated using the F1 score metric.

### Problem Statement

Predicting stock market movements is notoriously difficult due to:
- High volatility and non-linear patterns
- Influence of external factors (news, market sentiment, macroeconomic conditions)
- Temporal dependencies across different timeframes
- Signal-to-noise ratio challenges

### Project Goals
- Successfully predict stock trend movements for decision support
- Compare LSTM performance against traditional ARIMA baseline
- Create visualizations of predictions vs. actual stock movements
- Provide a reproducible pipeline for stock trend analysis

## Technical Approach

Our approach follows a structured machine learning pipeline:

1. **Data Acquisition**: Collect historical stock data from reliable sources (Yahoo Finance)
2. **Preprocessing**: Clean, normalize, and structure data for time-series analysis
3. **Feature Engineering**: Create technical indicators and derived features
4. **Model Development**:
   - Baseline: Traditional ARIMA model for time-series forecasting
   - Advanced: LSTM neural network for capturing complex patterns
5. **Evaluation**: Use F1 score for binary classification performance
6. **Visualization**: Generate plots comparing predicted vs. actual trends

## Environment Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (for faster LSTM training)
- Git
- 8GB+ RAM

### Detailed Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CS506-Final-Project.git
cd CS506-Final-Project
```

2. Set up a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify CUDA installation (optional, for GPU acceleration):
```bash
# In Python interpreter
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### Troubleshooting

Common installation issues:
- **PyTorch installation fails**: Try installing with specific CUDA version (see [PyTorch website](https://pytorch.org/get-started/locally/))
- **Memory errors during LSTM training**: Reduce batch size in config.yaml
- **CUDA not available**: Check NVIDIA drivers and CUDA toolkit compatibility

## Getting Started

To reproduce our results, follow these steps:

1. Collect stock data:
```bash
# For a single stock
python DataCollection/yfinance_data_processing.py --ticker MSFT --start 2015-01-01 --end 2023-12-31

# For all Magnificent 7 stocks
python DataCollection/yfinance_data_processing.py --all-mag7
```

2. Run the ARIMA baseline model:
```bash
cd Arima\(Baseline\)
jupyter notebook arima.ipynb
```

3. Train and evaluate the LSTM model:
```bash
cd LSTM
python main.py --config config.yaml
```

4. Generate results and visualizations:
```bash
# Results will be saved in LSTM/Result/ directory
python main.py --config config.yaml --visualize-only
```

## Data Collection

Our data collection process uses the `yfinance` library to fetch historical stock prices for the Magnificent 7 tech companies:

- **Data Sources**: Yahoo Finance API (reliable, free, and comprehensive)
- **Stocks**: AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA
- **Timeframes**: 
  - 1-day interval (primary dataset, dating back to stock inception)
  - 15-minute interval (higher frequency data for more recent periods)
- **Features**: Open, High, Low, Close prices and Volume
- **Date Range**: Historical data from stock inception to December 2023, with training set typically ending in 2020, and testing on post-2020 data

The data collection module organizes data in ticker-specific directories with consistent CSV formatting:

```python
# Example usage of data collection (from yfinance_data_processing.py)
python DataCollection/yfinance_data_processing.py --ticker MSFT --start 2010-01-01 --end 2023-12-31 --interval 1d
```

Each stock's data is saved as a CSV file with the following structure:
```
datetime,open,high,low,close,volume
2010-01-04,30.49,30.64,30.34,30.95,38414900
2010-01-05,30.95,31.10,30.76,30.96,49758200
...
```

## Data Preprocessing

After collecting raw data, we clean and engineer features for both ARIMA and LSTM models:

1. **Handling Missing Values**: 
   - Forward filling for continuity in time series
   - Interpolation for gaps in the middle of sequences
   - Drop days with no trading activity

2. **Normalization**: 
   - Using `MinMaxScaler` from scikit-learn to standardize values between 0-1
   - Separate scaling for each feature to preserve relative relationships
   - Store scalers for later inverse transformation during evaluation

3. **Sequence Generation**: 
   - Creating sliding windows of length 24 (default) for LSTM training
   - Each sequence becomes an input sample, with subsequent value(s) as target
   - Configurable via `seq_length` and `forecast_horizon` parameters

4. **Technical Indicator Calculation**: 
   - Add Moving Averages, RSI, MACD, and other indicators
   - These derived features help the model identify patterns

The preprocessing steps are implemented in the `LSTM/utils/data_utils.py` module:

```python
# Example usage
X, y, scaler = prepare_sequences(df, 
                                ["open", "high", "low", "close", "volume"], 
                                "close", 
                                seq_length=24, 
                                forecast_horizon=1)
```

## Feature Engineering

We extract and engineer the following features:

1. **Price-based Features**: 
   - Open, High, Low, Close, Volume (raw data)
   - Price differences and returns (first-order derivatives)
   - Volatility measures (standard deviation over rolling windows)

2. **Technical Indicators**:
   - **MACD (Moving Average Convergence Divergence)**:
     - Calculated as difference between 12-day and 26-day EMAs
     - Signal line: 9-day EMA of MACD
     - MACD histogram: MACD - Signal line
   - **RSI (Relative Strength Index)**:
     - Momentum oscillator measuring speed and change of price movements
     - Formula: 100 - (100 / (1 + RS)), where RS = Avg. Gain / Avg. Loss
   - **Bollinger Bands**:
     - Middle band: 20-day SMA
     - Upper/lower bands: Middle band ± (20-day standard deviation × 2)
   - **Moving Averages**:
     - Simple Moving Averages (SMA): 5-day, 10-day, 20-day
     - Exponential Moving Averages (EMA): 12-day, 26-day

3. **Binary Classification Target**:
   - MACD signal for buy/sell/hold recommendation
   - Converted to binary (1 = hold, 0 = sell) based on MACD crossing signal line
   - Custom binary target formula: 1 if MACD > Signal line, 0 otherwise

## Model Architecture

### ARIMA Baseline

Our baseline forecasting model uses ARIMA with the following configuration:

- **Model Parameters**: ARIMA(p, d, q)
  - p: Number of past data points (autoregressive terms)
  - d: Degree of differencing
  - q: Number of lagged forecast errors (moving average terms)

- **Implementation Details**:
  - Library: `statsmodels` in Python
  - Parameter selection: Grid search for optimal (p,d,q) combination
  - Stationarity testing: Augmented Dickey-Fuller test
  - Location: `Arima(Baseline)/arima.ipynb`

- **Workflow**:
  1. Test stationarity and apply differencing as needed
  2. Determine optimal p, d, q parameters using AIC/BIC
  3. Fit ARIMA model on training data
  4. Generate forecasts and convert to binary signals
  5. Evaluate using F1 score

- **Results**:
  - Amazon (AMZN) F1 Score: 0.6831
  - Nvidia (NVDA) F1 Score: 0.7101

### LSTM Model

Our advanced solution uses a Long Short-Term Memory neural network:

- **Architecture**:
  - **Input Layer**: Takes sequences of length 24 with 5 features
  - **LSTM Layers**: 2 stacked LSTM layers with hidden size 64
  - **Dropout**: 0.1 between layers for regularization
  - **Output Layer**: Fully connected layer for prediction

- **Memory Cells**: Each LSTM cell contains
  - Forget gate: Controls what information to discard
  - Input gate: Controls what new information to store
  - Output gate: Controls what information to output
  - Cell state: Long-term memory component
  - Hidden state: Short-term memory component

- **Implementation Details**:
  - Framework: PyTorch
  - Optimizer: Adam with learning rate 0.001
  - Loss Function: MSE (for regression)
  - Training: 75 epochs with early stopping (patience 10)
  - Batch Size: 32

- **Model Variants**:
  - Standard: Single fully connected layer after LSTM
  - Advanced: Additional dropout and hidden layer with ReLU activation

The LSTM model definition can be found in `LSTM/model/lstm.py`:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
```

## Training Process

Our training process for the LSTM model is implemented in `LSTM/utils/train.py` and includes:

1. **Initialization**:
   - Load data and config parameters
   - Initialize model, optimizer, and loss function
   - Set up device (GPU if available)

2. **Training Loop**:
   - Batch processing of input sequences
   - Forward pass through model
   - Loss calculation (MSE)
   - Backward pass for gradient computation
   - Parameter updates via optimizer

3. **Validation**:
   - Regular evaluation on validation set
   - Track validation loss for early stopping

4. **Early Stopping**:
   - Monitor validation loss
   - Stop training if no improvement for `patience` epochs
   - Save best model based on validation loss

5. **Hyperparameter Configuration**:
   - Learning rate: 0.001
   - Batch size: 32
   - Epochs: 75 (maximum)
   - Patience: 10 (for early stopping)
   - Hidden size: 64
   - Number of layers: 2
   - Dropout: 0.1

Here's the workflow for hyperparameter tuning:
1. Start with baseline parameters
2. Adjust one parameter at a time (e.g., hidden size, layers)
3. Evaluate F1 score on validation set
4. Select best configuration for final testing

## Results and Visualization

Our models were evaluated on test data (most recent 20% of the dataset):

### ARIMA Baseline Results
- AMZN F1 Score: 0.6831
- NVDA F1 Score: 0.7101

### LSTM Model Results
- AMZN F1 Score: 0.7410
- NVDA F1 Score: 0.7372
- MSFT F1 Score: 0.5800
- GOOG F1 Score: 0.6200

### Performance Comparison
LSTM outperformed ARIMA for Amazon and showed comparable performance for Nvidia, demonstrating its ability to capture more complex patterns in time series data.

### Visualizations
Prediction visualizations for each stock can be found in the `LSTM/Result/` directory. The plots show:
- Actual close price (blue line)
- Predicted close price (red dashed line)
- The degree of alignment demonstrates model accuracy

## Performance Analysis

### Comparative Metrics
For each stock, we calculated:
- **F1 Score**: Primary metric for binary classification performance
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Key Insights
- LSTM consistently outperformed ARIMA on F1 score for most stocks
- Performance varied significantly between stocks (best for AMZN, challenging for MSFT)
- Model performance was better for stocks with clearer trends
- The 80/20 training/testing split (with chronological ordering) proved effective

### Limitations
- LSTM struggled more with abrupt market changes (e.g., COVID-19 crash)
- Higher frequency data (15-min) showed more noise and challenging patterns
- Both models showed similar limitations during high volatility periods

## Testing

We implemented a comprehensive test suite to ensure the reliability of our pipeline. Run the tests with:

```bash
pytest tests/
```

### Test Suite Structure

Our tests cover multiple aspects of the pipeline:

1. **Model Tests** (`test_lstm_model.py`):
   - Tests the LSTM model structure and forward pass
   - Validates output shapes and dimensions

2. **Data Processing Tests** (`test_data_processing.py`):
   - Tests sequence preparation and scaling
   - Verifies data formatting and transformation

3. **Evaluation Tests** (`test_evaluation.py`):
   - Tests metric computation functions
   - Validates plot generation

4. **Training Tests** (`test_trainer.py`):
   - Tests the training and validation loops
   - Verifies early stopping functionality

5. **Integration Tests** (`test_lstm_integration.py`):
   - End-to-end testing of the full pipeline
   - Validates the entire workflow from data loading to evaluation

6. **ARIMA Tests** (`test_arima.py`):
   - Tests ARIMA model fitting and prediction
   - Validates binary classification with ARIMA

7. **Data Collection Tests** (`test_data_collection.py`):
   - Tests data retrieval and processing
   - Validates technical indicator calculation

### GitHub Workflow

Our project uses GitHub Actions for continuous integration:

1. **Linting and Style Checks**:
   - Checks code formatting using flake8
   - Ensures PEP8 style compliance

2. **Unit Tests**:
   - Automated testing on each push or pull request
   - Validates core functionality remains intact

The GitHub workflow configuration is defined in `.github/workflows/test.yml`.

## Challenges and Solutions

During development, we faced several challenges:

1. **Data Quality Issues**
   - **Challenge**: Missing values, inconsistent formats
   - **Solution**: Robust preprocessing with forward-filling and interpolation

2. **Model Overfitting**
   - **Challenge**: LSTM memorizing patterns rather than generalizing
   - **Solution**: Added dropout layers and early stopping

3. **Hyperparameter Tuning**
   - **Challenge**: Finding optimal settings for different stocks
   - **Solution**: Systematic grid search and cross-validation

4. **Binary Classification Performance**
   - **Challenge**: Converting regression outputs to binary signals
   - **Solution**: MACD-based signal generation with optimized thresholds

5. **GPU Resource Limitations**
   - **Challenge**: Training larger models with limited resources
   - **Solution**: Batch size optimization and efficient sequence handling

## Future Work

Several directions for future improvement include:

1. **Advanced Model Architectures**:
   - Transformer-based models for improved sequence modeling
   - Hybrid CNN-LSTM models for feature extraction

2. **Enhanced Features**:
   - News sentiment analysis integration
   - Macroeconomic indicators incorporation
   - Cross-asset correlations (e.g., sector indices, commodities)

3. **Trading Strategy Implementation**:
   - Back-testing framework with transaction costs
   - Portfolio optimization using model predictions
   - Risk management integration

4. **Real-time Prediction System**:
   - API integration for live data feeds
   - Deployment as a web service or application
   - Automated alert system for trading signals

## How to Contribute

We welcome contributions to this project. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Contribution Guidelines

- Maintain test coverage for new code
- Follow PEP 8 style guidelines
- Document new functions and classes
- Update README.md with any necessary changes

## Supported Environments

This project has been tested on:
- Windows 10/11
- Ubuntu 20.04/22.04
- Python 3.9, 3.10
- CUDA 11.7, 12.0 (for GPU acceleration)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project builds upon research in time series forecasting and deep learning
- Thanks to Yahoo Finance for providing historical stock data
- Special thanks to CS506 course staff for guidance and feedback

---

This README serves as the documentation for our final CS506 project. For questions or further information, please open an issue on the repository.


