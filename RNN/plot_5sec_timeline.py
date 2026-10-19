"""
5-Second Continuous Timeline: Model E (Attention, K=50, N=30) Predictions vs Ground Truth

Matches predictions back to the original timeline by comparing ground truth values
in the CSV with the original dataset. Plots a contiguous 5-second segment showing
how well the LSTM tracks the true attitude dynamics.

Usage:
    python3 plot_5sec_timeline.py [--start START_SEC]

Output:
    Results/LSTM/timeline_5sec_model_E.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

LOOKBACK = 50
WINDOW_SIZE = 30
TS = 0.015  # 66.6 Hz
DURATION = 5.0
CSV_PATH = '../Results/lstm_predictions_attn_N30lookback50.csv'
DATASET_PATH = '../Data/Quadcopter_Datasets/all_combined_reordered.csv'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=float, default=50.0,
                        help='Start time in seconds (default: 50)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load original dataset (ground truth)
    dataset = pd.read_csv(DATASET_PATH, header=None).values
    gt_angles = dataset[:, :3]  # roll, pitch, yaw in radians

    # Build a lookup: for each timestep, its (roll, pitch, yaw) rounded for matching
    # We'll match step_ahead=1 ground truth to find absolute position
    print("Building ground truth index...")
    # Hash the first 3 columns at each row for fast lookup
    gt_keys = {}
    for i in range(len(gt_angles)):
        key = (round(gt_angles[i, 0], 10), round(gt_angles[i, 1], 10), round(gt_angles[i, 2], 10))
        gt_keys[key] = i  # last one wins if duplicates (unlikely with float precision)

    # Load predictions
    print("Loading predictions CSV...")
    pred_df = pd.read_csv(CSV_PATH)

    # Filter to test set only
    test_df = pred_df[pred_df['set'] == 'test'].copy()
    num_steps = test_df['step_ahead'].max()
    num_test_samples = len(test_df) // num_steps
    print(f"Test set: {num_test_samples} samples, {num_steps} steps each")

    # For each test sample, find its absolute time position using step_ahead=1 ground truth
    print("Matching test samples to original timeline...")
    sample_time_map = {}  # sample_idx -> absolute_time_index of first prediction

    step1_df = test_df[test_df['step_ahead'] == 1]
    for _, row in step1_df.iterrows():
        sample_idx = int(row['sample'])
        key = (round(row['roll_gt'], 10), round(row['pitch_gt'], 10), round(row['yaw_gt'], 10))
        if key in gt_keys:
            sample_time_map[sample_idx] = gt_keys[key]

    print(f"Matched {len(sample_time_map)}/{num_test_samples} test samples to timeline")

    # Define time window
    start_idx = int(args.start / TS)
    end_idx = start_idx + int(DURATION / TS)
    end_idx = min(end_idx, len(dataset))
    window_indices = set(range(start_idx, end_idx))

    # Collect predictions that fall in this window
    # For each test sample, its predictions span [abs_time, abs_time + 29]
    pred_data = {}  # time_idx -> list of (roll_pred, pitch_pred, yaw_pred)

    for sample_idx, abs_start in sample_time_map.items():
        # Check if any of this sample's predictions fall in our window
        pred_range = set(range(abs_start, abs_start + WINDOW_SIZE))
        overlap = pred_range & window_indices
        if not overlap:
            continue

        # Get this sample's rows
        sample_rows = test_df[test_df['sample'] == sample_idx].sort_values('step_ahead')
        for _, row in sample_rows.iterrows():
            t_idx = abs_start + int(row['step_ahead']) - 1
            if t_idx in window_indices:
                if t_idx not in pred_data:
                    pred_data[t_idx] = []
                pred_data[t_idx].append((
                    np.rad2deg(row['roll_pred']),
                    np.rad2deg(row['pitch_pred']),
                    np.rad2deg(row['yaw_pred'])
                ))

    # Average predictions where multiple windows overlap the same timestep
    pred_roll = np.full(end_idx - start_idx, np.nan)
    pred_pitch = np.full(end_idx - start_idx, np.nan)
    pred_yaw = np.full(end_idx - start_idx, np.nan)

    for t_idx, preds in pred_data.items():
        j = t_idx - start_idx
        pred_roll[j] = np.mean([p[0] for p in preds])
        pred_pitch[j] = np.mean([p[1] for p in preds])
        pred_yaw[j] = np.mean([p[2] for p in preds])

    coverage = np.count_nonzero(~np.isnan(pred_roll))
    total = end_idx - start_idx
    print(f"Window: {args.start:.1f}s–{args.start + DURATION:.1f}s | "
          f"{coverage}/{total} timesteps have predictions ({100*coverage/total:.0f}%)")

    if coverage < 20:
        print("WARNING: Very few predictions in this window. Try a different --start value.")
        print("Scanning for best coverage...")
        best_start, best_count = 0, 0
        for s in range(0, len(dataset) - int(DURATION / TS), int(1.0 / TS)):
            e = s + int(DURATION / TS)
            count = sum(1 for t in range(s, e) if t in pred_data)
            if count > best_count:
                best_count = count
                best_start = s
        print(f"Best window: --start {best_start * TS:.1f} ({best_count} predictions)")
        return

    # --- Plot ---
    t = np.arange(start_idx, end_idx) * TS
    gt_r = np.rad2deg(gt_angles[start_idx:end_idx, 0])
    gt_p = np.rad2deg(gt_angles[start_idx:end_idx, 1])
    gt_y = np.rad2deg(gt_angles[start_idx:end_idx, 2])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    angles = [
        ('Roll', gt_r, pred_roll),
        ('Pitch', gt_p, pred_pitch),
        ('Yaw', gt_y, pred_yaw),
    ]

    for i, (name, gt, pred) in enumerate(angles):
        ax = axes[i]
        ax.plot(t, gt, label='Ground Truth', color='#1f77b4', linewidth=1.5, alpha=0.9)

        mask = ~np.isnan(pred)
        ax.plot(t[mask], pred[mask],
                label='LSTM Prediction (Model E)', color='#ff7f0e',
                linewidth=1.2, alpha=0.85)

        ax.set_ylabel(f'{name} (deg)', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

    fig.suptitle('Model E (Attention, K=50, N=30): 5-Second Prediction Timeline',
                 fontsize=14, fontweight='bold', y=1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=13,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.99))
    axes[2].set_xlabel('Time (seconds)', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs('Results/LSTM', exist_ok=True)
    output = 'Results/LSTM/timeline_5sec_model_E.png'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.show()


if __name__ == '__main__':
    main()
