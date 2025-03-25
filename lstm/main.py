# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from lstm_model import LSTMModel
from trainer import LSTMTrainer

def generate_sample_data(n_samples=1000, n_features=1):
    """
    Generate sample data for demonstration
    """
    t = np.linspace(0, 100, n_samples)
    data = np.sin(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    return data.reshape(-1, n_features)

def plot_results(history):
    """
    Plot training results
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Generate sample data
    data = generate_sample_data()
    
    # Initialize data processor
    data_processor = DataProcessor(sequence_length=10)
    
    # Prepare data
    train_X, train_y, test_X, test_y = data_processor.prepare_data(data)
    
    # Convert to PyTorch tensors
    train_X = torch.FloatTensor(train_X)
    train_y = torch.FloatTensor(train_y)
    test_X = torch.FloatTensor(test_X)
    test_y = torch.FloatTensor(test_y)
    
    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    input_size = 1  # number of features
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    # Initialize trainer
    trainer = LSTMTrainer(model)
    
    # Train model
    history = trainer.train(train_loader, test_loader, epochs=50)
    
    # Plot results
    plot_results(history)
    
    # Make predictions
    model.load_state_dict(torch.load('best_model.pth'))
    predictions = model.predict(test_X)
    
    # Inverse transform predictions and actual values
    predictions = data_processor.inverse_transform(predictions)
    test_y = data_processor.inverse_transform(test_y.numpy())
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(test_y, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('LSTM Predictions vs Actual Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()

if __name__ == '__main__':
    main()
