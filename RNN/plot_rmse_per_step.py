"""
Per-Step RMSE Curve: How prediction error grows from step 1 to step N.

Computes RMSE at each step ahead, averaged across ALL test samples.
Shows how error accumulates over the prediction horizon for each angle.

Usage:
    python3 plot_rmse_per_step.py [--csv CSV_PATH]

Output:
    Results/LSTM/rmse_per_step_model_E.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str,
                        default='../Results/lstm_predictions_attn_N30lookback50.csv',
                        help='Path to predictions CSV')
    parser.add_argument('--title', type=str, default='Model E (Attention, K=50, N=30)',
                        help='Plot title prefix')
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    test_df = df[df['set'] == 'test'].copy()
    num_steps = test_df['step_ahead'].max()
    num_samples = len(test_df) // num_steps
    print(f"Test set: {num_samples} samples, {num_steps} steps")

    # Compute RMSE per step per angle
    steps = np.arange(1, num_steps + 1)
    rmse_roll = np.zeros(num_steps)
    rmse_pitch = np.zeros(num_steps)
    rmse_yaw = np.zeros(num_steps)

    for s in range(num_steps):
        step_df = test_df[test_df['step_ahead'] == s + 1]
        rmse_roll[s] = np.sqrt(np.mean((np.rad2deg(step_df['roll_pred']) - np.rad2deg(step_df['roll_gt'])) ** 2))
        rmse_pitch[s] = np.sqrt(np.mean((np.rad2deg(step_df['pitch_pred']) - np.rad2deg(step_df['pitch_gt'])) ** 2))
        rmse_yaw[s] = np.sqrt(np.mean((np.rad2deg(step_df['yaw_pred']) - np.rad2deg(step_df['yaw_gt'])) ** 2))

    # Print table
    print(f"\n{'Step':>4} | {'Roll (deg)':>10} | {'Pitch (deg)':>11} | {'Yaw (deg)':>10}")
    print("-" * 45)
    for s in range(num_steps):
        print(f"{s+1:4d} | {rmse_roll[s]:10.3f} | {rmse_pitch[s]:11.3f} | {rmse_yaw[s]:10.3f}")

    # Time axis (each step = 0.015s at 66.6 Hz)
    TS = 0.015
    time_ms = steps * TS * 1000  # in milliseconds

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, rmse_roll, label='Roll', linewidth=2, color='#d62728', marker='o', markersize=4)
    ax.plot(steps, rmse_pitch, label='Pitch', linewidth=2, color='#2ca02c', marker='s', markersize=4)
    ax.plot(steps, rmse_yaw, label='Yaw', linewidth=2, color='#1f77b4', marker='^', markersize=4)

    ax.set_xlabel('Step Ahead', fontsize=14)
    ax.set_ylabel('RMSE (degrees)', fontsize=14)
    ax.set_title(f'{args.title}: Per-Step RMSE Across Test Set ({num_samples} samples)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticks(steps[::2])  # every 2 steps

    # Secondary x-axis for time in ms
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = steps[::5] - 1  # every 5 steps
    ax2.set_xticks(steps[::5])
    ax2.set_xticklabels([f'{s * TS * 1000:.0f}' for s in steps[::5]], fontsize=10)
    ax2.set_xlabel('Prediction Horizon (ms)', fontsize=12)

    plt.tight_layout()

    os.makedirs('Results/LSTM', exist_ok=True)
    output = 'Results/LSTM/rmse_per_step_model_E.png'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output}")
    plt.show()


if __name__ == '__main__':
    main()
