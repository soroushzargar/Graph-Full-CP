"""Conformal Prediction for graph node classification."""
import torch
import numpy as np
from typing import List, Tuple


class ConformalPredictor:
    """Conformal Prediction for uncertainty quantification in node classification.
    
    Implements split conformal prediction for graph neural networks.
    """
    
    def __init__(self, model, data, device='cpu'):
        """Initialize the conformal predictor.
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric data object
            device (str): Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.model.eval()
        self.quantile = None
        
    def compute_nonconformity_scores(self, logits, labels):
        """Compute nonconformity scores using softmax probabilities.
        
        Args:
            logits: Model output logits [num_nodes, num_classes]
            labels: True labels [num_nodes]
            
        Returns:
            scores: Nonconformity scores [num_nodes]
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Nonconformity score = 1 - probability of true class
        scores = 1 - probs[torch.arange(len(labels)), labels]
        
        return scores
    
    def calibrate(self, calibration_mask, alpha=0.1):
        """Calibrate the conformal predictor on calibration set.
        
        Args:
            calibration_mask: Boolean mask for calibration nodes
            alpha (float): Significance level (1 - coverage)
            
        Returns:
            float: Calibrated quantile threshold
        """
        with torch.no_grad():
            # Get model predictions on calibration set
            logits = self.model(self.data.x, self.data.edge_index)
            calib_logits = logits[calibration_mask]
            calib_labels = self.data.y[calibration_mask]
            
            # Compute nonconformity scores
            scores = self.compute_nonconformity_scores(calib_logits, calib_labels)
            
            # Compute quantile
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            self.quantile = torch.quantile(scores, q_level).item()
            
            return self.quantile
    
    def predict(self, test_mask=None, alpha=0.1):
        """Make prediction sets with conformal prediction.
        
        Args:
            test_mask: Boolean mask for test nodes (if None, use all nodes)
            alpha (float): Significance level (1 - coverage)
            
        Returns:
            tuple: (prediction_sets, set_sizes, coverages)
        """
        if self.quantile is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        with torch.no_grad():
            # Get model predictions
            logits = self.model(self.data.x, self.data.edge_index)
            
            if test_mask is not None:
                logits = logits[test_mask]
                labels = self.data.y[test_mask]
            else:
                labels = self.data.y
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Create prediction sets
            # Include classes where 1 - prob <= quantile
            prediction_sets = []
            for i in range(len(probs)):
                nonconformity = 1 - probs[i]
                pred_set = (nonconformity <= self.quantile).nonzero(as_tuple=True)[0].cpu().tolist()
                prediction_sets.append(pred_set)
            
            # Compute set sizes
            set_sizes = [len(ps) for ps in prediction_sets]
            
            # Compute coverage (proportion of true labels in prediction sets)
            coverage = sum(
                labels[i].item() in prediction_sets[i] 
                for i in range(len(prediction_sets))
            ) / len(prediction_sets)
            
            return prediction_sets, set_sizes, coverage
    
    def evaluate_coverage(self, test_mask, alpha=0.1):
        """Evaluate coverage and efficiency of conformal prediction.
        
        Args:
            test_mask: Boolean mask for test nodes
            alpha (float): Significance level
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        prediction_sets, set_sizes, coverage = self.predict(test_mask, alpha)
        
        metrics = {
            'coverage': coverage,
            'target_coverage': 1 - alpha,
            'average_set_size': np.mean(set_sizes),
            'median_set_size': np.median(set_sizes),
            'min_set_size': np.min(set_sizes),
            'max_set_size': np.max(set_sizes),
            'singleton_rate': sum(s == 1 for s in set_sizes) / len(set_sizes),
            'empty_rate': sum(s == 0 for s in set_sizes) / len(set_sizes)
        }
        
        return metrics
    
    def get_prediction_sets_for_nodes(self, node_indices, alpha=0.1):
        """Get prediction sets for specific nodes.
        
        Args:
            node_indices: List or tensor of node indices
            alpha (float): Significance level
            
        Returns:
            list: Prediction sets for specified nodes
        """
        if self.quantile is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            logits = logits[node_indices]
            probs = torch.softmax(logits, dim=1)
            
            prediction_sets = []
            for i in range(len(probs)):
                nonconformity = 1 - probs[i]
                pred_set = (nonconformity <= self.quantile).nonzero(as_tuple=True)[0].cpu().tolist()
                prediction_sets.append(pred_set)
            
            return prediction_sets
