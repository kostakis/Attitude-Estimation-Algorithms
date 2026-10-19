"""
Multi-Model LSTM Prediction Comparison

Creates two separate comparison plots grouped by prediction horizon:
  1. N=10 horizon: All models with 10-step predictions
  2. N=30 horizon: All models with 30-step predictions

Each plot overlays predictions from all relevant experiments on the same test sample
to visually compare attention vs no-attention and different lookback windows.

Usage:
    python3 plot_comparison.py <sample_index>

Arguments:
    sample_index - Index into the test set (0 = first test sample)

Output:
    Saves PNGs to Results/LSTM/comparison_N10_test_{index}.png
                   Results/LSTM/comparison_N30_test_{index}.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_N10 = {
    'LSTM (K=10)':              ('Results/lstm_predictions_noattn_N10.csv',  '#4daf4a'),
    'LSTM + Attention (K=10)':  ('Results/lstm_predictions_attn_N10.csv',    '#e41a1c')
}

EXPERIMENTS_N30 = {
    'LSTM (K=10)':              ('Results/lstm_predictions_noattn_N30.csv',        '#4daf4a'),
    'LSTM + Attention (K=10)':  ('Results/lstm_predictions_attn_N30.csv',          '#ff7f00')
}
# 'LSTM + Attention (K=50)':  ('Results/lstm_predictions_attn_N30lookback50.csv','#377eb8'),

ANGLE_NAMES = ['Roll', 'Pitch', 'Yaw']
PRED_COLS = ['roll_pred', 'pitch_pred', 'yaw_pred']
GT_COLS = ['roll_gt', 'pitch_gt', 'yaw_gt']


def parse_args():
    parser = argparse.ArgumentParser(description="Compare LSTM experiments grouped by prediction horizon")
    parser.add_argument("sample_index", type=int, help="Index into the test set (0 = first test sample)")
    return parser.parse_args()


def load_experiment(csv_path, sample_index):
    """Load a prediction CSV and extract a single test sample."""
    df = pd.read_csv(csv_path)
    df = df[df['set'] == 'test'].reset_index(drop=True)

    num_steps = df['step_ahead'].max()
    start_pos = sample_index * num_steps

    if start_pos + num_steps > len(df):
        return None, num_steps

    sample = df.iloc[start_pos:start_pos + num_steps]
    return sample, num_steps


def plot_horizon(experiments, sample_index, horizon):
    """Create comparison plot for a single prediction horizon."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot ground truth from the first valid experiment
    gt_plotted = False

    for exp_name, (csv_path, color) in experiments.items():
        sample, num_steps = load_experiment(csv_path, sample_index)

        if sample is None:
            print(f"Sample {sample_index} out of range for {exp_name}, skipping")
            continue

        x = sample['step_ahead'].values

        for i, (angle, pred_col, gt_col) in enumerate(zip(ANGLE_NAMES, PRED_COLS, GT_COLS)):
            pred_deg = np.rad2deg(sample[pred_col].values)

            if not gt_plotted:
                gt_deg = np.rad2deg(sample[gt_col].values)
                axes[i].plot(x, gt_deg, label='Ground Truth', linestyle='--',
                             linewidth=2.5, color='black', marker='o', markersize=4, zorder=10)

            axes[i].plot(x, pred_deg, label=exp_name, linestyle='-',
                         linewidth=1.8, color=color, marker='x', markersize=5, alpha=0.85)

        gt_plotted = True

    for i, angle in enumerate(ANGLE_NAMES):
        axes[i].set_ylabel(f'{angle} Angle (deg)', fontsize=14)
        axes[i].legend(loc='best', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', labelsize=12)

    axes[0].set_title(
        f'{horizon}-Step Ahead Prediction Comparison - Test Sample {sample_index}',
        fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Step Ahead', fontsize=14)
    axes[2].set_xticks(range(1, horizon + 1))

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    # Plot N=10 comparison
    fig_n10 = plot_horizon(EXPERIMENTS_N10, args.sample_index, horizon=10)
    filename_n10 = f'Results/LSTM/comparison_N10_test_{args.sample_index}.png'
    fig_n10.savefig(filename_n10, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename_n10}")

    # Plot N=30 comparison
    fig_n30 = plot_horizon(EXPERIMENTS_N30, args.sample_index, horizon=30)
    filename_n30 = f'Results/LSTM/comparison_N30_test_{args.sample_index}.png'
    fig_n30.savefig(filename_n30, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename_n30}")

    plt.show()


if __name__ == '__main__':
    main()
