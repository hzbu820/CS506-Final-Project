import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data):
        """
        Create sequences for LSTM model
        Args:
            data: numpy array of shape (n_samples, n_features)
        Returns:
            X: sequences of shape (n_sequences, sequence_length, n_features)
            y: target values of shape (n_sequences, n_features)
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, data, train_split=0.8):
        """
        Prepare data for training and testing
        Args:
            data: numpy array of shape (n_samples, n_features)
            train_split: ratio of training data
        Returns:
            train_X, train_y, test_X, test_y: numpy arrays
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        train_X, train_y = X[:train_size], y[:train_size]
        test_X, test_y = X[train_size:], y[train_size:]
        
        return train_X, train_y, test_X, test_y
    
    def inverse_transform(self, data):
        """
        Inverse transform scaled data back to original scale
        """
        return self.scaler.inverse_transform(data) 