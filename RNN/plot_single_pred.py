"""
Single Sample LSTM Prediction Visualizer

Plots the LSTM multi-step ahead prediction vs ground truth for a single test sample.
Reads the predictions CSV exported by lstmEval.out (--save-all mode), filters to the
test set, and creates a 3-row subplot (Roll, Pitch, Yaw) showing predicted vs actual
angles in degrees at each step ahead. Each subplot includes an RMSE annotation.

Usage:
    python3 plot_single_pred.py <sample_index> [csv_path] [lookback_window]

Arguments:
    sample_index    - Index into the test set (0 = first test sample)
    csv_path        - Path to predictions CSV (default: Results/lstm_predictions.csv)
    lookback_window - K value used during training, displayed in the plot title (default: 10)

Output:
    Saves PNG to Results/LSTM/single_sample_prediction_N{steps}_K{lookback}_test_{index}.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LSTM multi-step prediction vs ground truth for a single test sample")
    parser.add_argument("sample_index", type=int, help="Index into the test set (0 = first test sample)")
    parser.add_argument("--csv", type=str, default="Results/lstm_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--lookback", "-K", type=int, default=10, help="Lookback window K for plot title")
    return parser.parse_args()


def load_test_data(csv_path):
    """Load predictions CSV and filter to test set only."""
    df = pd.read_csv(csv_path)
    df = df[df['set'] == 'test'].reset_index(drop=True)

    num_steps = df['step_ahead'].max()
    num_samples = len(df) // num_steps
    print(f"Test set: {len(df)} rows ({num_samples} samples, {num_steps} steps each)")

    # Convert radians to degrees
    for col in ['roll_pred', 'roll_gt', 'pitch_pred', 'pitch_gt', 'yaw_pred', 'yaw_gt']:
        df[col + '_deg'] = np.rad2deg(df[col])

    return df, num_steps, num_samples


def extract_sample(df, sample_index, num_steps):
    """Extract a single sample's predictions and ground truth."""
    start_pos = sample_index * num_steps

    if start_pos + num_steps > len(df):
        raise ValueError(f"Sample index too large! Max: {len(df) // num_steps - 1}")

    return df.iloc[start_pos:start_pos + num_steps].copy()


def compute_rmse(df_sample):
    """Compute per-angle RMSE for a single sample."""
    rmse = {}
    for angle in ['roll', 'pitch', 'yaw']:
        pred = df_sample[f'{angle}_pred_deg']
        gt = df_sample[f'{angle}_gt_deg']
        rmse[angle] = np.sqrt(np.mean((pred - gt) ** 2))
    return rmse


def plot_sample(df_sample, sample_index, num_steps, lookback_window, rmse):
    """Create 3-row subplot with Roll, Pitch, Yaw predictions vs ground truth."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x_axis = df_sample['step_ahead'].values

    angles = [
        ('Roll',  'roll_pred_deg',  'roll_gt_deg'),
        ('Pitch', 'pitch_pred_deg', 'pitch_gt_deg'),
        ('Yaw',   'yaw_pred_deg',   'yaw_gt_deg'),
    ]

    for i, (name, pred_col, gt_col) in enumerate(angles):
        ax = axes[i]
        ax.tick_params(axis='both', labelsize=14)

        ax.plot(x_axis, df_sample[pred_col].values,
                label='LSTM Prediction', linestyle='-', linewidth=2,
                marker='x', markersize=8, color='#ff7f0e')
        ax.plot(x_axis, df_sample[gt_col].values,
                label='Ground Truth', linestyle='--', linewidth=2,
                marker='o', markersize=6, color='#1f77b4')

        ax.grid(True, alpha=0.3)

    # Y-axis labels and RMSE annotation on right side of each subplot
    angles_names = ['Roll', 'Pitch', 'Yaw']
    for i, name in enumerate(angles_names):
        axes[i].set_ylabel(f'{name} Angle (deg)', fontsize=13)
        # Place RMSE on the right y-axis area
        ax2 = axes[i].twinx()
        ax2.set_ylabel(f'RMSE: {rmse[name.lower()]:.3f}°', fontsize=12,
                        color='#d62728', fontweight='bold', rotation=270, labelpad=18)
        ax2.set_yticks([])

    # Title and shared legend
    fig.suptitle(
        f'LSTM {num_steps}-Step Ahead Prediction (K={lookback_window}) - Test Sample {sample_index}',
        fontsize=14, fontweight='bold', y=1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=13,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.99))

    axes[2].set_xlabel('Step Ahead', fontsize=16)
    axes[2].set_xticks(range(1, num_steps + 1))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    args = parse_args()

    df, num_steps, num_samples = load_test_data(args.csv)
    df_sample = extract_sample(df, args.sample_index, num_steps)
    rmse = compute_rmse(df_sample)

    print(f"Sample {args.sample_index} | Roll RMSE: {rmse['roll']:.3f} | "
          f"Pitch RMSE: {rmse['pitch']:.3f} | Yaw RMSE: {rmse['yaw']:.3f}")

    fig = plot_sample(df_sample, args.sample_index, num_steps, args.lookback, rmse)

    filename = f'Results/LSTM/single_sample_prediction_N{num_steps}_K{args.lookback}_test_{args.sample_index}.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {filename}")

    plt.show()


if __name__ == '__main__':
    main()
