"""
Trainer for LSTM model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class LSTMTrainer:
    """Trainer class for LSTM models"""
    
    def __init__(self, model, learning_rate=0.001):
        """
        Initialize the trainer
        
        Args:
            model: LSTM model to train
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(self, train_loader, valid_loader, epochs, patience=5, verbose=True, save_path=None):
        """
        Train the LSTM model
        
        Args:
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            epochs: Number of epochs to train
            patience: Early stopping patience
            verbose: Whether to print progress
            save_path: Path to save the best model
            
        Returns:
            dict: Training history
        """
        self.model.train()
        
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        counter = 0  # Counter for early stopping
        
        # Create directory for model saving if it doesn't exist
        if save_path and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
            
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            self.model.train()
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                
                # Calculate loss
                loss = self.criterion(y_pred, y_batch)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            valid_loss = self._validate(valid_loader)
            valid_losses.append(valid_loss)
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')
            
            # Check for improvement
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                counter = 0
                
                # Save the best model
                if save_path:
                    self._save_model(save_path)
            else:
                counter += 1
                
            # Early stopping
            if counter >= patience:
                if verbose:
                    print(f'Early stopping after {epoch+1} epochs')
                break
        
        # Load the best model if saved
        if save_path and os.path.exists(save_path):
            self._load_model(save_path)
            
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'best_valid_loss': best_valid_loss
        }
    
    def _validate(self, valid_loader):
        """Validate the model on validation data"""
        self.model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                valid_loss += loss.item()
        
        return valid_loss / len(valid_loader)
    
    def _save_model(self, path):
        """Save model to disk"""
        torch.save(self.model.state_dict(), path)
    
    def _load_model(self, path):
        """Load model from disk"""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        
    def plot_losses(self, train_losses, valid_losses, save_path=None):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show() 