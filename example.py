"""Example script demonstrating conformal prediction on graph node classification."""
import torch
import numpy as np
from core.data import DatasetLoader
from core.models import GCN, Trainer
from core.conformal import ConformalPredictor


def main():
    """Main function to run the example."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    dataset_name = 'Cora'  # Options: 'Cora', 'CiteSeer', 'PubMed', 'CoraML'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_dim = 64
    dropout = 0.5
    lr = 0.01
    weight_decay = 5e-4
    epochs = 200
    alpha = 0.1  # Significance level for conformal prediction (90% coverage)
    
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    loader = DatasetLoader(dataset_name=dataset_name, root='./data')
    data = loader.load()
    
    # Print dataset information
    info = loader.get_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Create model
    print("Creating GCN model...")
    model = GCN(
        num_features=loader.num_features,
        num_classes=loader.num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # Train model
    print(f"Training model for {epochs} epochs...")
    trainer = Trainer(
        model=model,
        data=data,
        device=device,
        lr=lr,
        weight_decay=weight_decay
    )
    
    history = trainer.train(epochs=epochs, verbose=True)
    print()
    
    # Save model
    model_path = './saved_models/gcn_model.pth'
    trainer.save_model(model_path)
    print()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_acc = trainer.evaluate(data.test_mask)
    print(f"Test Accuracy: {test_acc:.4f}")
    print()
    
    # Conformal Prediction
    print("=" * 60)
    print("Conformal Prediction")
    print("=" * 60)
    
    # Split validation set for calibration
    # Use half of validation set for calibration
    val_indices = data.val_mask.nonzero(as_tuple=True)[0]
    n_calib = len(val_indices) // 2
    calib_indices = val_indices[:n_calib]
    
    calib_mask = torch.zeros_like(data.val_mask)
    calib_mask[calib_indices] = True
    
    print(f"Calibration set size: {calib_mask.sum().item()}")
    print(f"Test set size: {data.test_mask.sum().item()}")
    print(f"Significance level (alpha): {alpha}")
    print(f"Target coverage: {1 - alpha:.2%}")
    print()
    
    # Initialize conformal predictor
    conformal_predictor = ConformalPredictor(
        model=model,
        data=data,
        device=device
    )
    
    # Calibrate
    print("Calibrating conformal predictor...")
    quantile = conformal_predictor.calibrate(calib_mask, alpha=alpha)
    print(f"Calibrated quantile: {quantile:.4f}")
    print()
    
    # Evaluate on test set
    print("Evaluating conformal prediction on test set...")
    metrics = conformal_predictor.evaluate_coverage(data.test_mask, alpha=alpha)
    
    print("Conformal Prediction Metrics:")
    print(f"  Coverage: {metrics['coverage']:.4f} (target: {metrics['target_coverage']:.4f})")
    print(f"  Average set size: {metrics['average_set_size']:.2f}")
    print(f"  Median set size: {metrics['median_set_size']:.2f}")
    print(f"  Min set size: {metrics['min_set_size']}")
    print(f"  Max set size: {metrics['max_set_size']}")
    print(f"  Singleton rate: {metrics['singleton_rate']:.4f}")
    print(f"  Empty rate: {metrics['empty_rate']:.4f}")
    print()
    
    # Show some example prediction sets
    print("Example prediction sets for first 5 test nodes:")
    test_indices = data.test_mask.nonzero(as_tuple=True)[0][:5]
    prediction_sets = conformal_predictor.get_prediction_sets_for_nodes(test_indices, alpha=alpha)
    
    for i, (node_idx, pred_set) in enumerate(zip(test_indices.cpu().tolist(), prediction_sets)):
        true_label = data.y[node_idx].item()
        in_set = "✓" if true_label in pred_set else "✗"
        print(f"  Node {node_idx}: True label: {true_label}, Prediction set: {pred_set}, Coverage: {in_set}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
