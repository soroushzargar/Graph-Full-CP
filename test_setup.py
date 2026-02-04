"""Quick test script to verify the setup works."""
import torch
import numpy as np
from core.data import DatasetLoader
from core.models import GCN, Trainer
from core.conformal import ConformalPredictor


def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("=" * 60)
    print("Testing Graph-Full-CP Framework")
    print("=" * 60)
    print()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    dataset_name = 'Cora'
    device = 'cpu'
    epochs = 10  # Just a few epochs for testing
    
    # Test 1: Load dataset
    print("Test 1: Loading dataset...")
    try:
        loader = DatasetLoader(dataset_name=dataset_name, root='./data')
        data = loader.load()
        info = loader.get_info()
        print(f"✓ Dataset loaded: {info['name']}")
        print(f"  - Nodes: {info['num_nodes']}, Edges: {info['num_edges']}")
        print(f"  - Features: {info['num_features']}, Classes: {info['num_classes']}")
        print()
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Test 2: Create model
    print("Test 2: Creating GCN model...")
    try:
        model = GCN(
            num_features=loader.num_features,
            num_classes=loader.num_classes,
            hidden_dim=16,
            dropout=0.5
        )
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print()
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Test 3: Train model
    print(f"Test 3: Training model for {epochs} epochs...")
    try:
        trainer = Trainer(
            model=model,
            data=data,
            device=device,
            lr=0.01,
            weight_decay=5e-4
        )
        history = trainer.train(epochs=epochs, verbose=False)
        test_acc = trainer.evaluate(data.test_mask)
        print(f"✓ Model trained successfully")
        print(f"  - Final test accuracy: {test_acc:.4f}")
        print()
    except Exception as e:
        print(f"✗ Failed to train model: {e}")
        return False
    
    # Test 4: Conformal prediction
    print("Test 4: Applying conformal prediction...")
    try:
        # Use part of validation set for calibration
        val_indices = data.val_mask.nonzero(as_tuple=True)[0]
        n_calib = len(val_indices) // 2
        calib_indices = val_indices[:n_calib]
        calib_mask = torch.zeros_like(data.val_mask)
        calib_mask[calib_indices] = True
        
        conformal_predictor = ConformalPredictor(
            model=model,
            data=data,
            device=device
        )
        
        alpha = 0.1
        quantile = conformal_predictor.calibrate(calib_mask, alpha=alpha)
        metrics = conformal_predictor.evaluate_coverage(data.test_mask, alpha=alpha)
        
        print(f"✓ Conformal prediction applied successfully")
        print(f"  - Target coverage: {1-alpha:.2%}")
        print(f"  - Actual coverage: {metrics['coverage']:.2%}")
        print(f"  - Average set size: {metrics['average_set_size']:.2f}")
        print()
    except Exception as e:
        print(f"✗ Failed conformal prediction: {e}")
        return False
    
    # Test 5: Model save/load
    print("Test 5: Saving and loading model...")
    try:
        import os
        os.makedirs('./test_models', exist_ok=True)
        model_path = './test_models/test_model.pth'
        trainer.save_model(model_path)
        
        # Create new trainer and load
        new_model = GCN(
            num_features=loader.num_features,
            num_classes=loader.num_classes,
            hidden_dim=16,
            dropout=0.5
        )
        new_trainer = Trainer(new_model, data, device=device)
        new_trainer.load_model(model_path)
        
        # Verify loaded model has same accuracy
        loaded_acc = new_trainer.evaluate(data.test_mask)
        assert abs(loaded_acc - test_acc) < 1e-6, "Loaded model accuracy mismatch"
        
        print(f"✓ Model saved and loaded successfully")
        print()
        
        # Cleanup
        os.remove(model_path)
        os.rmdir('./test_models')
    except Exception as e:
        print(f"✗ Failed to save/load model: {e}")
        return False
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
