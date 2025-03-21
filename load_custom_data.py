import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from lstm_model import LSTMModel
from trainer import LSTMTrainer

def load_csv_data(file_path, target_column, feature_columns=None):
    """
    Load data from CSV file
    
    Args:
        file_path: path to CSV file
        target_column: name of the target column
        feature_columns: list of feature column names, if None, will use all columns except target
    
    Returns:
        data: numpy array with shape (n_samples, n_features)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # If no feature columns provided, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Extract features and target
    features = df[feature_columns].values
    target = df[target_column].values.reshape(-1, 1)
    
    # Combine features and target (for time series forecasting)
    data = np.concatenate([features, target], axis=1)
    
    return data

def load_numpy_data(file_path):
    """
    Load data from numpy file
    
    Args:
        file_path: path to numpy file (.npy)
    
    Returns:
        data: numpy array
    """
    return np.load(file_path)

def train_with_custom_data(data, sequence_length=10, train_split=0.8, batch_size=32, 
                          hidden_size=64, num_layers=2, learning_rate=0.001, epochs=50):
    """
    Train LSTM model with custom data
    
    Args:
        data: numpy array with shape (n_samples, n_features)
        sequence_length: length of input sequences
        train_split: ratio of training data
        batch_size: batch size for training
        hidden_size: number of hidden units in LSTM
        num_layers: number of LSTM layers
        learning_rate: learning rate for optimization
        epochs: number of training epochs
        
    Returns:
        model: trained LSTM model
        history: training history
        data_processor: data processor instance for inverse transform
    """
    # Initialize data processor
    data_processor = DataProcessor(sequence_length=sequence_length)
    
    # Prepare data
    train_X, train_y, test_X, test_y = data_processor.prepare_data(data, train_split=train_split)
    
    # Print shapes for verification
    print(f"Training data shape: {train_X.shape}")
    print(f"Training targets shape: {train_y.shape}")
    print(f"Test data shape: {test_X.shape}")
    print(f"Test targets shape: {test_y.shape}")
    
    # Convert to PyTorch tensors
    train_X = torch.FloatTensor(train_X)
    train_y = torch.FloatTensor(train_y)
    test_X = torch.FloatTensor(test_X)
    test_y = torch.FloatTensor(test_y)
    
    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = data.shape[1]  # number of features
    output_size = data.shape[1]  # predict all features
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    # Initialize trainer
    trainer = LSTMTrainer(model, learning_rate=learning_rate)
    
    # Train model
    history = trainer.train(train_loader, test_loader, epochs=epochs)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('custom_training_history.png')
    plt.close()
    
    return model, history, data_processor, test_X, test_y

def make_predictions(model, test_X, test_y, data_processor, target_idx=None):
    """
    Make predictions with trained model
    
    Args:
        model: trained LSTM model
        test_X: test data
        test_y: test targets
        data_processor: data processor instance
        target_idx: index of target column, if None, predict all features
        
    Returns:
        predictions: numpy array of predictions
    """
    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Make predictions
    predictions = model.predict(test_X)
    
    # Inverse transform predictions and actual values
    predictions = data_processor.inverse_transform(predictions)
    test_y_numpy = data_processor.inverse_transform(test_y.numpy())
    
    # If target_idx is provided, only plot that column
    if target_idx is not None:
        predictions = predictions[:, target_idx].reshape(-1, 1)
        test_y_numpy = test_y_numpy[:, target_idx].reshape(-1, 1)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_y_numpy, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('LSTM Predictions vs Actual Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('custom_predictions.png')
    plt.show()
    
    # Calculate evaluation metrics
    mse = np.mean((predictions - test_y_numpy) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - test_y_numpy))
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    return predictions

def example_usage():
    """
    Example of how to use the functions to train and test the model with custom data
    """
    # 1. Load your data (example with CSV)
    # data = load_csv_data('your_data.csv', target_column='target', feature_columns=['feature1', 'feature2'])
    
    # 2. Or load numpy data
    # data = load_numpy_data('your_data.npy')
    
    # 3. For demonstration, create sample data
    # Example: multivariate time series (3 features)
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    feature1 = np.sin(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    feature2 = np.cos(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    feature3 = 0.5 * np.sin(0.1 * t) + 0.5 * np.cos(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    
    data = np.column_stack([feature1, feature2, feature3])
    
    # 4. Train model
    model, history, data_processor, test_X, test_y = train_with_custom_data(
        data, 
        sequence_length=15,  # Adjust based on your data characteristics
        train_split=0.8,
        batch_size=32,
        hidden_size=128,  # Larger for more complex patterns
        num_layers=2,
        learning_rate=0.001,
        epochs=100
    )
    
    # 5. Make predictions (predict third feature only)
    target_idx = 2  # Index of the target column to predict
    predictions = make_predictions(model, test_X, test_y, data_processor, target_idx)
    
    return model, predictions

if __name__ == "__main__":
    example_usage() 