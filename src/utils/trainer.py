import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

class LSTMTrainer:
    """
    Trainer class for LSTM models
    """
    def __init__(self, model, learning_rate=0.001):
        """
        Initialize the trainer
        
        Args:
            model: LSTM model to train
            learning_rate: Learning rate for the optimizer
        """
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = next(model.parameters()).device
        
    def train(self, train_loader, val_loader, epochs=100, model_save_path='outputs/models/best_model.pth'):
        """
        Train the LSTM model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            model_save_path: Path to save the best model
            
        Returns:
            history: Dictionary containing training history
        """
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Initialize best validation loss
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output, target)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    # Move data to device
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    
                    # Compute loss
                    loss = self.criterion(output, target)
                    
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                
            # Save the model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                
                # Save the model
                torch.save(self.model.state_dict(), model_save_path)
                
        print(f'Finished training. Best validation loss: {best_val_loss:.4f}')
        
        return history 