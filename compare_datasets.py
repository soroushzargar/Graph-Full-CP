"""Example script to compare conformal prediction across different datasets."""
import torch
import numpy as np
from core.data import DatasetLoader
from core.models import GCN, Trainer
from core.conformal import ConformalPredictor


def run_experiment(dataset_name, epochs=50, alpha=0.1, device='cpu'):
    """Run experiment on a single dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        epochs (int): Number of training epochs
        alpha (float): Significance level for conformal prediction
        device (str): Device to run on
        
    Returns:
        dict: Results dictionary
    """
    # Load dataset
    loader = DatasetLoader(dataset_name=dataset_name, root='./data')
    data = loader.load()
    
    # Create and train model
    model = GCN(
        num_features=loader.num_features,
        num_classes=loader.num_classes,
        hidden_dim=64,
        dropout=0.5
    )
    
    trainer = Trainer(model, data, device=device)
    history = trainer.train(epochs=epochs, verbose=False)
    
    # Get test accuracy
    test_acc = trainer.evaluate(data.test_mask)
    
    # Conformal prediction
    val_indices = data.val_mask.nonzero(as_tuple=True)[0]
    n_calib = len(val_indices) // 2
    calib_mask = torch.zeros_like(data.val_mask)
    calib_mask[val_indices[:n_calib]] = True
    
    cp = ConformalPredictor(model, data, device=device)
    quantile = cp.calibrate(calib_mask, alpha=alpha)
    metrics = cp.evaluate_coverage(data.test_mask, alpha=alpha)
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'num_nodes': loader.get_info()['num_nodes'],
        'num_classes': loader.get_info()['num_classes'],
        'test_accuracy': test_acc,
        'coverage': metrics['coverage'],
        'avg_set_size': metrics['average_set_size'],
        'singleton_rate': metrics['singleton_rate'],
        'empty_rate': metrics['empty_rate']
    }
    
    return results


def main():
    """Main function to run experiments on multiple datasets."""
    # Configuration
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    epochs = 100
    alpha = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("Conformal Prediction Comparison Across Datasets")
    print("=" * 80)
    print(f"Configuration: {epochs} epochs, alpha={alpha}, device={device}")
    print()
    
    results_list = []
    
    for dataset_name in datasets:
        print(f"Running experiment on {dataset_name}...")
        results = run_experiment(dataset_name, epochs=epochs, alpha=alpha, device=device)
        results_list.append(results)
        print(f"  ✓ Completed")
    
    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    
    # Print header
    header = f"{'Dataset':<12} {'Nodes':<8} {'Classes':<8} {'Test Acc':<10} {'Coverage':<10} {'Avg Size':<10} {'Singleton':<10}"
    print(header)
    print("-" * 80)
    
    # Print results
    for res in results_list:
        row = (f"{res['dataset']:<12} "
               f"{res['num_nodes']:<8} "
               f"{res['num_classes']:<8} "
               f"{res['test_accuracy']:<10.4f} "
               f"{res['coverage']:<10.4f} "
               f"{res['avg_set_size']:<10.2f} "
               f"{res['singleton_rate']:<10.4f}")
        print(row)
    
    print()
    print(f"Target coverage: {1-alpha:.2%}")
    print()
    
    # Analysis
    print("Analysis:")
    avg_coverage = np.mean([r['coverage'] for r in results_list])
    avg_set_size = np.mean([r['avg_set_size'] for r in results_list])
    
    print(f"  - Average coverage across datasets: {avg_coverage:.2%}")
    print(f"  - Average set size across datasets: {avg_set_size:.2f}")
    print(f"  - Coverage guarantee satisfied: {all(r['coverage'] >= 1-alpha-0.05 for r in results_list)}")
    print()
    
    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
