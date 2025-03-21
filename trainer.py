import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class LSTMTrainer:
    def __init__(self, model, learning_rate=0.001):
        """
        Initialize trainer
        Args:
            model: LSTM model instance
            learning_rate: learning rate for optimization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        Args:
            train_loader: DataLoader for training data
        Returns:
            loss: average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate the model
        Args:
            val_loader: DataLoader for validation data
        Returns:
            loss: average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=5):
        """
        Train the model
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: number of epochs to train
            early_stopping_patience: number of epochs to wait before early stopping
        Returns:
            history: dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc='Training'):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
                
        return history 