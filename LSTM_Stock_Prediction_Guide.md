# LSTM Stock Prediction Project Guide

This guide provides detailed instructions for setting up and using the LSTM Stock Prediction project. The project uses Long Short-Term Memory (LSTM) neural networks to predict future stock prices based on historical data.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Project Structure](#project-structure)
3. [Running Predictions](#running-predictions)
4. [Understanding Results](#understanding-results)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)
7. [Verifying Installation](#verifying-installation)
8. [Workflow Overview](#workflow-overview)

## Project Setup

### Prerequisites
- Python 3.7+ installed
- Git (optional, for version control)
- Pip (Python package manager)

### Step 1: Clone or Download the Project
If you received this project as a ZIP file, extract it to a directory of your choice. If using Git:

```bash
git clone https://github.com/yourusername/LSTM-Stock-Prediction.git
cd LSTM-Stock-Prediction
```

### Step 2: Set Up Virtual Environment
Creating a virtual environment is highly recommended to avoid dependency conflicts:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
Install the required packages:

```bash
pip install numpy pandas matplotlib torch scikit-learn scipy yfinance statsmodels
```

If you encounter any issues with dependencies, run the dependency checking script:

```bash
python scripts/fix_dependencies.py
```

### Step 4: Create Required Directories
Ensure the necessary output directories exist:

```bash
mkdir -p outputs/models outputs/figures outputs/predictions outputs/ensemble outputs/normalized_ensemble
```

## Project Structure

```
LSTM/
├── src/
│   ├── models/
│   │   └── stock_predictor.py      # Main LSTM model implementation
│   └── data/
│       └── data_loader.py          # Data loading and preprocessing
├── scripts/
│   ├── train_new_model.py          # Train a new LSTM model
│   ├── direction_ensemble.py       # Create direction-based ensemble predictions
│   ├── quick_prediction.py         # Use existing models for quick predictions
│   ├── analyze_existing_results.py # Analyze existing prediction files
│   ├── robust_model_training.py    # More robust model training script
│   └── fix_dependencies.py         # Check and fix dependency issues
├── outputs/
│   ├── models/                     # Saved model files
│   ├── figures/                    # Generated charts
│   ├── predictions/                # Individual model predictions
│   ├── ensemble/                   # Ensemble predictions
│   └── normalized_ensemble/        # Normalized ensemble predictions
└── README.md                       # Project overview
```

## Workflow Overview

The LSTM Stock Prediction project follows a structured workflow from data acquisition to trading signals:

```
+-------------------+     +-------------------+     +-------------------+
| Data Acquisition  |     | Data              |     | LSTM Model        |
| (Yahoo Finance)   |---->| Preprocessing     |---->| Training          |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
| Trading Signal    |     | Ensemble          |     | Individual Model  |
| Generation        |<----| Prediction        |<----| Predictions       |
+-------------------+     +-------------------+     +-------------------+
```

### Workflow Steps

1. **Data Acquisition**: 
   - Historical stock data is fetched from Yahoo Finance using the yfinance library
   - Data includes Open, High, Low, Close prices and Volume
   - Adjustable date ranges allow for different training periods

2. **Data Preprocessing**:
   - Missing values handling
   - Feature normalization (usually with MinMaxScaler)
   - Creating sequence data for LSTM input
   - Train/validation split

3. **LSTM Model Training**:
   - Multi-layer LSTM architecture
   - Configurable parameters (sequence length, hidden size, layers)
   - Training with early stopping based on validation loss
   - Model checkpointing to save best weights

4. **Individual Model Predictions**:
   - Generate future price predictions from each model
   - Save predictions to CSV files
   - Create visualization charts

5. **Ensemble Prediction**:
   - Combine predictions from multiple models
   - Two main approaches:
     - Direction Ensemble: Focus on price movement direction
     - Normalized Ensemble: Scale predictions to current price

6. **Trading Signal Generation**:
   - Calculate confidence scores
   - Generate BUY/SELL/HOLD recommendations
   - Provide detailed analysis summary

### Optimizing the Workflow

For best results:

1. **Start with data exploration**: Run the analysis script first to understand existing predictions
2. **Use robust training**: For new models, use the robust_model_training.py script
3. **Ensemble for reliability**: Ensemble predictions tend to be more reliable than individual models
4. **Consider confidence levels**: Higher confidence predictions should carry more weight in decisions

## Running Predictions

### Option 1: Quick Analysis (Recommended for Beginners)
To quickly analyze existing prediction files without running any model training:

```bash
python scripts/analyze_existing_results.py AAPL
```

Replace `AAPL` with any other ticker symbol you want to analyze. This script provides:
- Summary of existing prediction files
- Consensus direction (UP/DOWN)
- Trading signal recommendation
- No heavy dependencies required

### Option 2: Train a New Model
To train a new model from scratch:

```bash
python scripts/robust_model_training.py --ticker AAPL --epochs 50 --sequence_length 14 --hidden_size 128 --num_layers 2
```

Parameters explained:
- `--ticker`: Stock symbol (e.g., AAPL, MSFT, GOOGL)
- `--epochs`: Number of training epochs (50 is a good default)
- `--sequence_length`: Historical days used for each prediction (14 recommended for stability)
- `--hidden_size`: LSTM hidden layer size (128 recommended)
- `--num_layers`: Number of LSTM layers (2 recommended)
- `--future_days`: Days to predict into the future (default: 14)
- `--debug`: Enable detailed error messages

### Option 3: Direction Ensemble Prediction
Create a direction-focused ensemble from existing predictions:

```bash
python scripts/direction_ensemble.py AAPL
```

This script:
- Loads existing prediction files
- Normalizes predictions to current price
- Creates an ensemble focusing on price direction
- Generates trading recommendations

### Option 4: Quick Prediction
Use existing models to make predictions without heavy training:

```bash
python scripts/quick_prediction.py --ticker AAPL --debug
```

## Understanding Results

### Prediction Files
The system generates several types of prediction files in the `outputs` directory:

1. **Individual Model Predictions** (`outputs/predictions/`):
   - Predictions from single models with different parameters
   - Example file: `AAPL_enhanced_predictions.csv`

2. **Ensemble Predictions** (`outputs/ensemble/`):
   - Combined predictions from multiple models
   - Example file: `AAPL_direction_ensemble_20250326_183013.csv`

3. **Normalized Ensemble Predictions** (`outputs/normalized_ensemble/`):
   - Ensemble predictions normalized to current stock price
   - Example file: `AAPL_normalized_ensemble_20250326_183013.csv`

### Expected Output File Structure

A typical prediction CSV file contains:

```
Datetime,Predicted_Price,Model_Count,Direction_Confidence,Direction
2023-06-01,221.53,5,0.78,UP
2023-06-02,222.46,5,0.65,UP
...
```

Key columns:
- `Datetime`: Date of the prediction
- `Predicted_Price`: Forecasted stock price
- `Direction`: Predicted price movement (UP/DOWN)
- `Direction_Confidence`: Confidence level in the direction prediction

For ensemble files, additional columns may include:
- `Up_Vote`: Number of models predicting upward movement
- `Down_Vote`: Number of models predicting downward movement
- `Total_Votes`: Total number of model votes
- `Pct_Change`: Percentage change from current price

### Interpreting Results
The prediction summary provides:

1. **Price Forecasts**: Future prices predicted by the models
2. **Percentage Change**: Expected price change over the prediction period
3. **Direction**: UP or DOWN trend prediction
4. **Confidence Level**: How confident the model is in its prediction
5. **Trading Signal**: BUY, SELL, or HOLD recommendation based on predictions

Example output:
```
Ensemble Direction Prediction Summary:
Current Price: $221.53
Final Price: $225.83 (1.94%)
Overall Direction: UP with 80.4% average confidence

Trading Recommendation: BUY
Moderate confidence (80.4%) in upward movement
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
If you encounter import errors, run:
```bash
python scripts/fix_dependencies.py
```

This script will:
- Check all required dependencies
- Offer to install missing packages
- Suggest modifications for better import stability

#### 2. Memory/Performance Issues
For large models or performance issues:
- Reduce `--sequence_length` to 10-14 days
- Reduce `--hidden_size` to 64-128
- Reduce `--num_layers` to 1-2
- Use fewer `--epochs` (30-50)

#### 3. "Cannot compare tz-naive and tz-aware timestamps" Error
This timezone error can occur in some ensemble scripts. To avoid it:
- Use the `analyze_existing_results.py` script which is more robust
- Open CSV files directly to check predictions

#### 4. Missing Module Errors
If you encounter "No module named 'src.models'" errors:
- Ensure you are running scripts from the project root directory
- Check that the src directory exists with the expected structure
- If needed, add `import sys; sys.path.append('.')` at the top of scripts

## Advanced Usage

### Custom Date Ranges
Train a model with specific date ranges:

```bash
python scripts/robust_model_training.py --ticker MSFT --start_date 2020-01-01 --end_date 2023-01-01
```

### Multi-Stock Analysis
Analyze multiple stocks at once:

```bash
# Create a batch file or shell script
for ticker in AAPL MSFT GOOGL AMZN META NVDA
do
    python scripts/analyze_existing_results.py $ticker
done
```

### Model Ensemble Techniques
The project implements several ensemble techniques:

1. **Direction Ensemble**: Focuses on price movement direction
2. **Normalized Ensemble**: Aligns predictions with current price
3. **Weighted Ensemble**: Gives more weight to better-performing models

For custom ensembles, modify the ensemble parameters in the respective scripts.

## Verifying Installation

After setting up the project, you can verify that everything is working correctly by:

### 1. Running the Dependency Check

```bash
python scripts/fix_dependencies.py
```

This should show that all required dependencies are available or help you install any missing ones.

### 2. Analyzing Existing Results

```bash
python scripts/analyze_existing_results.py AAPL
```

If successful, you should see a summary of prediction files and a consensus direction.

### 3. Checking Directory Structure

Verify that the expected directory structure exists:

```bash
# Windows
dir outputs

# macOS/Linux
ls -la outputs
```

You should see subdirectories for models, figures, predictions, ensemble, and normalized_ensemble.

### 4. Examining Project Files

Check that key project files are in place:

```bash
# Windows
dir scripts
dir src\models

# macOS/Linux
ls -la scripts
ls -la src/models
```

Key scripts to look for include:
- `scripts/train_new_model.py`
- `scripts/analyze_existing_results.py`
- `src/models/stock_predictor.py`

If all of the above checks pass, your LSTM Stock Prediction project is set up correctly and ready to use!

---

By following this guide, you should be able to set up and use the LSTM Stock Prediction project effectively. For more detailed information on specific components, refer to the code documentation within each file. 