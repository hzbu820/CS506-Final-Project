"""
LSTM Model for stock price prediction
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        """
        Initialize the LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Output prediction
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step output
        out = self.fc(lstm_out[:, -1, :])
        
        return out
    
    def predict(self, x):
        """
        Make predictions using the model
        
        Args:
            x: Input data
            
        Returns:
            torch.Tensor: Predictions
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions 