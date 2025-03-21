import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize LSTM model
        Args:
            input_size: number of input features
            hidden_size: number of hidden units in LSTM
            num_layers: number of LSTM layers
            output_size: number of output features
            dropout: dropout rate
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
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            output: output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output
        out = self.fc(out[:, -1, :])
        return out
    
    def predict(self, x):
        """
        Make predictions
        Args:
            x: input tensor
        Returns:
            predictions: numpy array of predictions
        """
        self.eval()
        with torch.no_grad():
            predictions = self(x)
            return predictions.numpy() 