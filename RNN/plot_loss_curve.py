"""
Training vs Validation Loss Curve for Model E (Attention, K=50, N=30)

Parses the training log to extract loss values and plots the convergence curve.

Usage:
    python3 plot_loss_curve.py [--log LOG_FILE]

Output:
    Results/LSTM/loss_curve_model_E.png
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str,
                        default='results_attention_based30predictions50WindowSize.txt',
                        help='Path to training log file')
    return parser.parse_args()


def main():
    args = parse_args()

    epochs = []
    train_losses = []
    val_losses = []

    pattern = re.compile(
        r'Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)'
    )

    with open(args.log, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))

    print(f"Parsed {len(epochs)} data points (epoch {epochs[0]} to {epochs[-1]})")
    print(f"Train loss: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
    print(f"Val   loss: {val_losses[0]:.6f} -> {val_losses[-1]:.6f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, label='Training Loss', linewidth=1.8, color='#d62728')
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=1.8, color='#2ca02c')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE Loss', fontsize=14)
    ax.set_title('Model E (Attention, K=50, N=30): Training vs Validation Loss',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    os.makedirs('Results/LSTM', exist_ok=True)
    output = 'Results/LSTM/loss_curve_model_E.png'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.show()


if __name__ == '__main__':
    main()
