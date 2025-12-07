import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from transformer_model import TransformerForecaster
from trainer import TimeSeriesDataset, train_model, get_attention_weights_direct
from visualization import (
    plot_attention_heatmap, plot_attention_analysis_summary, 
    plot_entropy_analysis, analyze_attention_patterns
)
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fixed hyperparameters
CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'input_len': 100,
    'output_len': 20,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'train_split': 0.7,
    'val_split': 0.15,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def load_real_series(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df["value"].values.astype(np.float32)

def get_real_world_generators(data_dir='real_data'):
    files = sorted(os.listdir(data_dir))
    generators = {}
    for fname in files:
        if fname.endswith('.csv'):
            name = fname.replace(".csv", "")
            path = os.path.join(data_dir, fname)
            generators[name] = lambda path=path: load_real_series(path)
    return generators

def prepare_data(series, config):
    series_mean = np.mean(series)
    series_std = np.std(series)
    series = (series - series_mean) / (series_std + 1e-8)

    n = len(series)
    train_end = int(n * config['train_split'])
    val_end = train_end + int(n * config['val_split'])

    train_series = series[:train_end]
    val_series = series[train_end:val_end]
    test_series = series[val_end:]

    train_dataset = TimeSeriesDataset(train_series, config['input_len'], config['output_len'])
    val_dataset = TimeSeriesDataset(val_series, config['input_len'], config['output_len'])
    test_dataset = TimeSeriesDataset(test_series, config['input_len'], config['output_len'])

    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, {'mean': series_mean, 'std': series_std}

def run_experiment(generator_name, generator, config, results_dir='results_real'):
    print(f"\n{'='*60}\nExperiment: {generator_name}\n{'='*60}")
    os.makedirs(results_dir, exist_ok=True)
    generator_dir = os.path.join(results_dir, generator_name)
    os.makedirs(generator_dir, exist_ok=True)

    print("Loading data...")
    series = generator()
    train_loader, val_loader, test_loader, norm_params = prepare_data(series, config)

    print("Initializing model...")
    model = TransformerForecaster(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        input_len=config['input_len'],
        output_len=config['output_len']
    )

    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        device=config['device']
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Curves: {generator_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(generator_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print("Extracting attention weights...")
    test_batch = next(iter(test_loader))[0][:5].to(config['device'])
    attention_weights = get_attention_weights_direct(model, test_batch, device=config['device'])

    print("Creating visualizations...")
    plot_attention_analysis_summary(attention_weights, generator_name, save_dir=generator_dir)
    plot_entropy_analysis(attention_weights, generator_name, save_path=os.path.join(generator_dir, 'entropy_analysis.png'))

    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[1]
    n_samples = min(3, attention_weights[0].shape[0])

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            for sample_idx in range(n_samples):
                plot_attention_heatmap(
                    attention_weights,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    sample_idx=sample_idx,
                    title=f'{generator_name} - Layer {layer_idx}, Head {head_idx}, Sample {sample_idx}',
                    save_path=os.path.join(generator_dir, f'attention_L{layer_idx}_H{head_idx}_S{sample_idx}.png')
                )
                plt.close()

    print("Computing attention metrics...")
    metrics = {}
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            layer_head_metrics = [analyze_attention_patterns(attention_weights, layer_idx, head_idx, s) for s in range(attention_weights[0].shape[0])]
            avg_metrics = {k: float(np.mean([m[k] for m in layer_head_metrics])) for k in layer_head_metrics[0]}
            metrics[f'layer_{layer_idx}_head_{head_idx}'] = avg_metrics

    with open(os.path.join(generator_dir, 'attention_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    summary = {
        'generator_name': generator_name,
        'test_loss': test_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'metrics': metrics,
        'config': config
    }

    with open(os.path.join(generator_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Test Loss: {test_loss:.6f}\nResults saved to {generator_dir}")
    return summary

def main():
    print("="*60)
    print("Attention Map Analysis: Real-World Data")
    print("="*60)
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])

    generators = get_real_world_generators(data_dir='real_data')
    all_results = {}

    for generator_name, generator in tqdm(generators.items(), desc="Running experiments"):
        try:
            result = run_experiment(generator_name, generator, CONFIG)
            all_results[generator_name] = result
        except Exception as e:
            print(f"Error in {generator_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    with open('results/comparison_summary.json', 'w') as f:
        json.dump([{ 'generator': k, 'test_loss': v['test_loss'], 'final_val_loss': v['final_val_loss']} for k, v in all_results.items()], f, indent=2)

    print("\nAll experiments completed! Results in 'results/'")

if __name__ == '__main__':
    main()