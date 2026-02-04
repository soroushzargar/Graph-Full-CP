# Graph-Full-CP Usage Guide

This guide provides step-by-step instructions for using the Graph-Full-CP framework.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/soroushzargar/Graph-Full-CP.git
cd Graph-Full-CP

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Basic Example

```bash
python example.py
```

This will:
- Load the Cora dataset
- Train a GCN model for 200 epochs
- Apply conformal prediction
- Display results and metrics

### 3. Test the Setup

```bash
python test_setup.py
```

This runs a comprehensive test suite to verify all components.

### 4. Compare Datasets

```bash
python compare_datasets.py
```

This compares conformal prediction performance across Cora, CiteSeer, and PubMed.

## Detailed Usage

### Loading a Dataset

```python
from core.data import DatasetLoader

# Load Cora dataset
loader = DatasetLoader(dataset_name='Cora', root='./data')
data = loader.load()

# Get dataset information
info = loader.get_info()
print(f"Nodes: {info['num_nodes']}, Classes: {info['num_classes']}")

# Get features and labels
features, labels = loader.get_features_and_labels()

# Get train/val/test splits
train_mask, val_mask, test_mask = loader.get_splits()
```

### Training a Model

```python
from core.models import GCN, Trainer
import torch

# Create model
model = GCN(
    num_features=loader.num_features,
    num_classes=loader.num_classes,
    hidden_dim=64,
    dropout=0.5
)

# Create trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(
    model=model,
    data=data,
    device=device,
    lr=0.01,
    weight_decay=5e-4
)

# Train
history = trainer.train(epochs=200, verbose=True)

# Evaluate
test_acc = trainer.evaluate(data.test_mask)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
trainer.save_model('./saved_models/my_model.pth')
```

### Applying Conformal Prediction

```python
from core.conformal import ConformalPredictor

# Create conformal predictor
cp = ConformalPredictor(model=model, data=data, device=device)

# Prepare calibration set (use half of validation set)
val_indices = data.val_mask.nonzero(as_tuple=True)[0]
n_calib = len(val_indices) // 2
calib_mask = torch.zeros_like(data.val_mask)
calib_mask[val_indices[:n_calib]] = True

# Calibrate
alpha = 0.1  # For 90% coverage
quantile = cp.calibrate(calib_mask, alpha=alpha)

# Evaluate on test set
metrics = cp.evaluate_coverage(data.test_mask, alpha=alpha)
print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Average set size: {metrics['average_set_size']:.2f}")

# Get prediction sets for specific nodes
node_indices = [0, 1, 2, 3, 4]
pred_sets = cp.get_prediction_sets_for_nodes(node_indices, alpha=alpha)
```

## Understanding Conformal Prediction

### Key Concepts

1. **Significance Level (α)**: Controls the coverage guarantee
   - α = 0.1 → 90% coverage
   - α = 0.05 → 95% coverage
   - Lower α = higher coverage but larger prediction sets

2. **Prediction Sets**: Instead of single predictions, conformal prediction returns sets of labels
   - Empty set: Very uncertain
   - Single label: High confidence
   - Multiple labels: Uncertain between these options

3. **Coverage Guarantee**: With probability ≥ 1-α, the true label is in the prediction set

4. **Set Size**: Indicates uncertainty
   - Smaller sets = more certain predictions
   - Larger sets = less certain predictions

### Metrics

- **Coverage**: Proportion of test samples where true label is in prediction set (should be ≥ 1-α)
- **Average Set Size**: Mean number of labels in prediction sets (lower is better)
- **Singleton Rate**: Proportion of sets with exactly one label (higher is better)
- **Empty Rate**: Proportion of empty sets (should be close to 0)

## Advanced Usage

### Custom Model Architecture

```python
# Modify GCN architecture
model = GCN(
    num_features=data.num_features,
    num_classes=data.num_classes,
    hidden_dim=128,  # Increase hidden dimension
    dropout=0.6      # Adjust dropout
)
```

### Different Training Configurations

```python
# Longer training
trainer = Trainer(model, data, device=device, lr=0.005, weight_decay=1e-3)
history = trainer.train(epochs=500, verbose=True)
```

### Different Coverage Levels

```python
# Test different coverage levels
for alpha in [0.05, 0.1, 0.15, 0.2]:
    cp.calibrate(calib_mask, alpha=alpha)
    metrics = cp.evaluate_coverage(test_mask, alpha=alpha)
    print(f"α={alpha}: Coverage={metrics['coverage']:.2%}, "
          f"Avg Size={metrics['average_set_size']:.2f}")
```

### Loading Saved Models

```python
# Load previously saved model
model = GCN(num_features, num_classes, hidden_dim=64)
trainer = Trainer(model, data, device=device)
trainer.load_model('./saved_models/my_model.pth')

# Use for conformal prediction
cp = ConformalPredictor(model, data, device)
```

## Troubleshooting

### Installation Issues

If you encounter issues with PyTorch Geometric:
```bash
# Install with specific CUDA version (example for CUDA 11.8)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Memory Issues

For large datasets like PubMed:
```python
# Use CPU if GPU memory is insufficient
device = 'cpu'

# Or reduce batch size/model size
model = GCN(num_features, num_classes, hidden_dim=32)
```

### Coverage Too Low/High

Adjust calibration set size:
```python
# Use more data for calibration
n_calib = len(val_indices) * 3 // 4  # Use 75% of validation set
```

## Examples

See the following files for complete examples:
- `example.py` - Basic usage demonstration
- `test_setup.py` - Component testing
- `compare_datasets.py` - Multi-dataset comparison

## References

1. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
2. Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction.
3. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
