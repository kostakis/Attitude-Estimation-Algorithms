"""
LSTM Prediction vs Ground Truth Timeline (Test Set, 1-step ahead)

Plots 1-step ahead predictions against ground truth for consecutive test samples.
X-axis can be sample index or time (seconds) when --time flag is used.

Usage:
    python3 plot_predictions.py [start] [end]
    python3 plot_predictions.py [start] [end] --csv <path>
    python3 plot_predictions.py 0 333 --time    # 333 samples ≈ 5 seconds at 66.6 Hz

Arguments:
    start  - First test sample index (default: 0)
    end    - Last test sample index (default: 200)
    --csv  - Path to predictions CSV (default: Results/lstm_predictions_attn_N30lookback50.csv)
    --time - Use time axis in seconds instead of sample index (Ts=0.015s)

Output:
    Results/LSTM/lstm_predictions_test_{start}_{end}.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

TS = 0.015  # 66.6 Hz


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LSTM 1-step predictions vs ground truth")
    parser.add_argument("start", type=int, nargs='?', default=0, help="Start sample index")
    parser.add_argument("end", type=int, nargs='?', default=200, help="End sample index")
    parser.add_argument("--csv", type=str,
                        default="Results/lstm_predictions_attn_N30lookback50.csv",
                        help="Path to predictions CSV")
    parser.add_argument("--time", action="store_true",
                        help="Use time axis (seconds) instead of sample index")
    parser.add_argument("--angle", type=str, default=None, choices=['roll', 'pitch', 'yaw'],
                        help="Plot a single angle only (default: all three)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the predictions CSV
    df = pd.read_csv(args.csv)

    # Filter to test set, 1-step ahead only
    df = df[(df['set'] == 'test') & (df['step_ahead'] == 1)].reset_index(drop=True)
    print(f"Test set, 1-step ahead: {len(df)} samples")

    # Convert radians to degrees
    for col in ['roll_pred', 'roll_gt', 'pitch_pred', 'pitch_gt', 'yaw_pred', 'yaw_gt']:
        df[col + '_deg'] = np.rad2deg(df[col])

    # Filter to sample range
    df_filtered = df.iloc[args.start:args.end]
    n = len(df_filtered)

    if args.time:
        x_axis = np.arange(n) * TS
        x_label = 'Time (seconds)'
    else:
        x_axis = np.arange(args.start, args.start + n)
        x_label = 'Test Sample Index'

    # Compute RMSE per angle
    rmse = {}
    for angle in ['roll', 'pitch', 'yaw']:
        pred = df_filtered[f'{angle}_pred_deg']
        gt = df_filtered[f'{angle}_gt_deg']
        rmse[angle] = np.sqrt(np.mean((pred - gt) ** 2))

    duration_str = f" ({n * TS:.1f}s)" if args.time else ""
    print(f"Plotting {n} samples{duration_str}")
    print(f"Roll RMSE: {rmse['roll']:.3f}° | Pitch RMSE: {rmse['pitch']:.3f}° | Yaw RMSE: {rmse['yaw']:.3f}°")

    # Select which angles to plot
    all_angles = [
        ('Roll',  'roll_pred_deg',  'roll_gt_deg'),
        ('Pitch', 'pitch_pred_deg', 'pitch_gt_deg'),
        ('Yaw',   'yaw_pred_deg',   'yaw_gt_deg'),
    ]

    if args.angle:
        angles_cfg = [a for a in all_angles if a[0].lower() == args.angle]
    else:
        angles_cfg = all_angles

    num_plots = len(angles_cfg)
    fig_height = 5 if num_plots == 1 else 3 * num_plots
    fig, axes_raw = plt.subplots(num_plots, 1, figsize=(14, fig_height), sharex=True)
    if num_plots == 1:
        axes_raw = [axes_raw]

    for i, (name, pred_col, gt_col) in enumerate(angles_cfg):
        ax = axes_raw[i]
        ax.tick_params(axis='both', labelsize=12)

        ax.plot(x_axis, df_filtered[gt_col].values,
                label='Ground Truth', linewidth=2.2, color='#1f77b4',
                linestyle='--', alpha=0.8)
        ax.plot(x_axis, df_filtered[pred_col].values,
                label='LSTM Prediction', linewidth=1.2, color='#ff7f0e')

        ax.set_ylabel(f'{name} (deg)', fontsize=13)
        ax.grid(True, alpha=0.3)

        # RMSE on right y-axis label (no overlap with data)
        ax2 = ax.twinx()
        ax2.set_ylabel(f'RMSE: {rmse[name.lower()]:.3f}°', fontsize=12,
                        color='#d62728', fontweight='bold', rotation=270, labelpad=18)
        ax2.set_yticks([])

    # Shared legend and title
    angle_str = f' - {args.angle.capitalize()}' if args.angle else ''
    fig.suptitle(f'LSTM 1-Step Ahead Prediction vs Ground Truth (Test Set){angle_str}',
                 fontsize=14, fontweight='bold', y=1.02)
    handles, labels = axes_raw[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=13,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.99))
    axes_raw[-1].set_xlabel(x_label, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs('Results/LSTM', exist_ok=True)
    suffix = f'_{args.angle}' if args.angle else ''
    filename = f'Results/LSTM/lstm_predictions_test_{args.start}_{args.end}{suffix}.png'
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()


if __name__ == '__main__':
    main()
