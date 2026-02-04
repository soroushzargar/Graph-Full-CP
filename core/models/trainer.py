"""Trainer class for GNN models."""
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Trainer:
    """Trainer class for training and evaluating GNN models."""
    
    def __init__(self, model, data, device='cpu', lr=0.01, weight_decay=5e-4):
        """Initialize the trainer.
        
        Args:
            model: GNN model
            data: PyTorch Geometric data object
            device (str): Device to train on ('cpu' or 'cuda')
            lr (float): Learning rate
            weight_decay (float): Weight decay for L2 regularization
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        
    def train_epoch(self):
        """Train for one epoch.
        
        Returns:
            float: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss only on training nodes
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate the model on a given mask.
        
        Args:
            mask: Boolean mask for evaluation
            
        Returns:
            float: Accuracy
        """
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)
        
        correct = pred[mask] == self.data.y[mask]
        acc = correct.sum().item() / mask.sum().item()
        
        return acc
    
    def train(self, epochs=200, verbose=True):
        """Train the model for multiple epochs.
        
        Args:
            epochs (int): Number of epochs
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            loss = self.train_epoch()
            self.scheduler.step()
            
            # Evaluate
            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask) if hasattr(self.data, 'val_mask') and self.data.val_mask is not None else 0
            test_acc = self.evaluate(self.data.test_mask)
            
            history['train_loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f'\nLoaded best model with validation accuracy: {best_val_acc:.4f}')
        
        return history
    
    def save_model(self, path='model.pth'):
        """Save the model weights.
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path='model.pth'):
        """Load the model weights.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {path}')
    
    @torch.no_grad()
    def predict(self, mask=None):
        """Get predictions for nodes.
        
        Args:
            mask: Optional mask to get predictions for specific nodes
            
        Returns:
            tuple: (predictions, logits)
        """
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)
        
        if mask is not None:
            return pred[mask], out[mask]
        else:
            return pred, out
