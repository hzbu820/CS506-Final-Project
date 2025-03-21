# LSTM Model Implementation


## Project Structure

- `main.py`: Main script demonstrating the usage with sample data
- `data_processor.py`: Data preprocessing utilities 
- `lstm_model.py`: LSTM model definition
- `trainer.py`: Training pipeline
- `load_custom_data.py`: Script for using the model with your own data

## Installation

Install the required dependencies: inside requirements.txt

```bash
pip install -r requirements.txt
```

## Using the Model with our Data

### Option 1: Using the `load_custom_data.py` script

This script provides comprehensive utilities for loading different types of data and using it with the LSTM model.

```python
from load_custom_data import load_csv_data, train_with_custom_data, make_predictions

# 1. Load data 
data = load_csv_data(
    file_path='your_data.csv', 
    target_column='target',  # Column you want to predict
    feature_columns=['feature1', 'feature2']  # Optional - columns to use as features
)

# 2. Train model
model, history, data_processor, test_X, test_y = train_with_custom_data(
    data, 
    sequence_length=15,  # Number of time steps to use for prediction
    train_split=0.8,     # Ratio of training data
    batch_size=32,       # Batch size for training
    hidden_size=128,     # Hidden units in LSTM layer
    num_layers=2,        # Number of LSTM layers
    learning_rate=0.001, # Learning rate for optimization
    epochs=100           # Number of training epochs
)

# 3. Make predictions
target_idx = 0  # Index of target column (if multi-dimensional)
predictions = make_predictions(model, test_X, test_y, data_processor, target_idx)
```

Directly with own script

If you prefer to integrate the model into your own script

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_processor import DataProcessor
from lstm_model import LSTMModel
from trainer import LSTMTrainer

# 1. Prepare your data - should be numpy array with shape (n_samples, n_features)
data = your_data_loading_function()  # Replace with your data loading

# 2. Initialize data processor
data_processor = DataProcessor(sequence_length=15)  # Adjust sequence length

# 3. Process data
train_X, train_y, test_X, test_y = data_processor.prepare_data(data, train_split=0.8)

# 4. Convert to PyTorch tensors
train_X = torch.FloatTensor(train_X)
train_y = torch.FloatTensor(train_y)
test_X = torch.FloatTensor(test_X)
test_y = torch.FloatTensor(test_y)

# 5. Create data loaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. Initialize model
input_size = data.shape[1]   # Number of features in your data
hidden_size = 128            # Adjust based on your needs
num_layers = 2               # Number of LSTM layers
output_size = data.shape[1]  # Usually same as input_size for forecasting

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 7. Train model
trainer = LSTMTrainer(model, learning_rate=0.001)
history = trainer.train(train_loader, test_loader, epochs=100)

# 8. Make predictions
model.load_state_dict(torch.load('best_model.pth'))
predictions = model.predict(test_X)

# 9. Convert predictions back to original scale
predictions = data_processor.inverse_transform(predictions)
```

## Data Format Requirements

- Your data should be in a format that can be converted to a NumPy array
- For time series data, the shape should be (n_samples, n_features)
- For CSV files, each row represents a time step and each column a feature
- The model can handle both univariate (1 feature) and multivariate (multiple features) data

## Customizing the Model

You can customize the LSTM model by modifying the parameters:

- `sequence_length`: Number of previous time steps used to predict the next step
- `hidden_size`: Number of units in LSTM hidden layers (increase for complex patterns)
- `num_layers`: Number of stacked LSTM layers (increase for more complex dependencies)
- `learning_rate`: Controls how quickly the model learns
- `batch_size`: Number of samples processed before model update



## Evaluation

The model automatically calculates:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

These metrics are displayed after prediction with the `make_predictions` function. 