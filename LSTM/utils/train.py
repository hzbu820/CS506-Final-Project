import torch

class LSTMTrainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, train_loader, val_loader, epochs=30, patience=5):
        best_val_loss = float('inf')
        early_stop_count = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train = total_train_loss / len(train_loader)
            train_losses.append(avg_train)

            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}: train {avg_train:.4f} | val {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print("Early stopping triggered.")
                    break

        return train_losses, val_losses

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)
