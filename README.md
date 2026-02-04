# Graph-Full-CP

A comprehensive framework for testing conformal prediction on graph node classification tasks. This project provides tools for loading graph datasets (CoraML, CiteSeer, PubMed), training Graph Neural Networks (GNNs), and applying conformal prediction for uncertainty quantification.

## Features

- **Dataset Loading**: Easy loading of popular graph datasets (Cora, CoraML, CiteSeer, PubMed)
- **GNN Models**: Implementation of Graph Convolutional Network (GCN)
- **Training Framework**: Trainer class for training models with automatic best model selection
- **Conformal Prediction**: Split conformal prediction for node classification with coverage guarantees
- **Model Persistence**: Save and load trained models

## Project Structure

```
Graph-Full-CP/
├── core/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_loader.py      # Dataset loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py                 # GCN model implementation
│   │   └── trainer.py             # Training and evaluation
│   └── conformal/
│       ├── __init__.py
│       └── conformal_predictor.py # Conformal prediction
├── example.py                      # Example usage script
└── requirements.txt                # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/soroushzargar/Graph-Full-CP.git
cd Graph-Full-CP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyTorch Geometric may require additional installation steps depending on your CUDA version. See [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Quick Start

Run the example script to train a GCN model and apply conformal prediction:

```bash
python example.py
```

This will:
1. Load the Cora dataset
2. Train a GCN model for 200 epochs
3. Save the trained model
4. Apply conformal prediction with 90% coverage guarantee
5. Display coverage metrics and example prediction sets

## Usage

### Loading a Dataset

```python
from core.data import DatasetLoader

# Load dataset
loader = DatasetLoader(dataset_name='Cora', root='./data')
data = loader.load()

# Get dataset information
info = loader.get_info()
print(info)
```

Supported datasets: `'Cora'`, `'CoraML'`, `'CiteSeer'`, `'PubMed'`

### Training a GNN Model

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

# Train model
history = trainer.train(epochs=200, verbose=True)

# Save model
trainer.save_model('./saved_models/my_model.pth')

# Load model later
trainer.load_model('./saved_models/my_model.pth')
```

### Applying Conformal Prediction

```python
from core.conformal import ConformalPredictor

# Create conformal predictor
conformal_predictor = ConformalPredictor(
    model=model,
    data=data,
    device=device
)

# Calibrate on calibration set
alpha = 0.1  # For 90% coverage
quantile = conformal_predictor.calibrate(calib_mask, alpha=alpha)

# Evaluate on test set
metrics = conformal_predictor.evaluate_coverage(data.test_mask, alpha=alpha)
print(f"Coverage: {metrics['coverage']:.4f}")
print(f"Average set size: {metrics['average_set_size']:.2f}")

# Get prediction sets for specific nodes
prediction_sets = conformal_predictor.get_prediction_sets_for_nodes(
    node_indices=[0, 1, 2, 3, 4],
    alpha=alpha
)
```

## Conformal Prediction

Conformal prediction provides finite-sample coverage guarantees for predictions. Given a significance level α (e.g., 0.1 for 90% coverage), the method produces prediction sets that contain the true label with probability at least 1-α.

Key features:
- **Coverage Guarantee**: Mathematically guaranteed coverage on test data
- **Distribution-Free**: Works without assumptions about data distribution
- **Set-Valued Predictions**: Returns sets of possible labels instead of single predictions
- **Uncertainty Quantification**: Set size indicates prediction uncertainty

## Metrics

The framework provides several metrics for evaluating conformal prediction:

- **Coverage**: Proportion of test samples where true label is in prediction set
- **Average Set Size**: Mean size of prediction sets (lower is better for efficiency)
- **Singleton Rate**: Proportion of prediction sets with exactly one class
- **Empty Rate**: Proportion of empty prediction sets (should be close to 0)

## Examples

### Example 1: Basic Usage

```python
import torch
from core.data import DatasetLoader
from core.models import GCN, Trainer
from core.conformal import ConformalPredictor

# Load data
loader = DatasetLoader('Cora')
data = loader.load()

# Train model
model = GCN(loader.num_features, loader.num_classes)
trainer = Trainer(model, data)
trainer.train(epochs=200)

# Apply conformal prediction
cp = ConformalPredictor(model, data)
cp.calibrate(data.val_mask, alpha=0.1)
metrics = cp.evaluate_coverage(data.test_mask)
print(metrics)
```

### Example 2: Different Datasets

```python
# Try different datasets
for dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'CoraML']:
    loader = DatasetLoader(dataset_name)
    data = loader.load()
    print(f"{dataset_name}: {loader.get_info()}")
```

### Example 3: Different Coverage Levels

```python
# Test different coverage levels
for alpha in [0.05, 0.1, 0.2]:
    cp.calibrate(calib_mask, alpha=alpha)
    metrics = cp.evaluate_coverage(test_mask, alpha=alpha)
    print(f"Alpha={alpha}: Coverage={metrics['coverage']:.4f}, "
          f"Avg Size={metrics['average_set_size']:.2f}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{graph_full_cp,
  title = {Graph-Full-CP: Conformal Prediction for Graph Node Classification},
  author = {Zargar, Soroush},
  year = {2026},
  url = {https://github.com/soroushzargar/Graph-Full-CP}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.