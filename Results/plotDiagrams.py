import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot roll and pitch estimation comparison')
parser.add_argument('--filter', choices=['complementary', 'mahony', 'ekf', 'explicitcf'],
                    default='mahony', help='Filter to plot (default: mahony)')
parser.add_argument('--start', type=int, default=200, help='Start sample index (default: 200)')
parser.add_argument('--end', type=int, default=1000, help='End sample index (default: 1000)')
parser.add_argument('--full', action='store_true', help='Show full data range (ignores --start and --end)')
args = parser.parse_args()

# Filter configuration
filter_config = {
    'complementary': {
        'roll_file': 'Results/Results/ComplementaryRoll_a_0_79.txt',
        'pitch_file': 'Results/Results/ComplementaryPitch_a_0_79.txt',
        'name': 'Complementary Filter',
        'param': 'a = 0.79',
        'roll_rmse': 0.820,
        'pitch_rmse': 0.771
    },
    'mahony': {
        'roll_file': 'Results/Results/MahonyRoll_kp_11.txt',
        'pitch_file': 'Results/Results/MahonyPitch_kp_11.txt',
        'name': 'Mahony Filter',
        'param': 'kp = 11',
        'roll_rmse': 0.614,
        'pitch_rmse': 0.756
    },
    'ekf': {
        'roll_file': 'Results/Results/EkfRoll.txt',
        'pitch_file': 'Results/Results/EkfPitch.txt',
        'name': 'Extended Kalman Filter',
        'param': '',
        'roll_rmse': 0.298,
        'pitch_rmse': 0.720
    },
    'explicitcf': {
        'roll_file': 'Results/Results/ExplicitComplementaryRoll.txt',
        'pitch_file': 'Results/Results/ExplicitComplementaryPitch.txt',
        'name': 'Explicit Complementary Filter',
        'param': '',
        'roll_rmse': 0.554,
        'pitch_rmse': 0.752
    }
}

config = filter_config[args.filter]

# Load ground truth
rollGroundTruth = np.loadtxt('Results/ExpectedResults/expected_roll.txt')
pitchGroundTruth = np.loadtxt('Results/ExpectedResults/expected_pitch.txt')

# Load predictions
rollPredicted = np.loadtxt(config['roll_file'])
pitchPredicted = np.loadtxt(config['pitch_file'])

# Define zoom region
if args.full:
    start_idx = 0
    end_idx = len(rollGroundTruth)
else:
    start_idx = args.start
    end_idx = min(args.end, len(rollGroundTruth))

# Extract zoomed data
roll_pred_zoom = rollPredicted[start_idx:end_idx]
roll_truth_zoom = rollGroundTruth[start_idx:end_idx]
pitch_pred_zoom = pitchPredicted[start_idx:end_idx]
pitch_truth_zoom = pitchGroundTruth[start_idx:end_idx]
time_axis = np.arange(start_idx, end_idx)


# Create 2-row subplot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].tick_params(axis='both', labelsize=14)
axes[1].tick_params(axis='both', labelsize=14)

# Panel 1: Roll
axes[0].plot(time_axis, roll_pred_zoom, label='Estimated Roll', linestyle='-', linewidth=1.5, color='#ff7f0e')
axes[0].plot(time_axis, roll_truth_zoom, label='Ground Truth', linestyle='--', linewidth=1.5, color='#1f77b4')
axes[0].set_ylabel('Roll Angle (degrees)', fontsize=14)
axes[0].set_title(f'{config["name"]} - Roll & Pitch Estimation (samples {start_idx}-{end_idx})', fontsize=14)
axes[0].legend(loc='upper right', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Add RMSE text for roll
rmse_text = f'Roll RMSE: {config["roll_rmse"]:.3f} deg'
if config['param']:
    rmse_text += f'  |  {config["param"]}'
axes[0].text(0.02, 0.95, rmse_text, transform=axes[0].transAxes, fontsize=13,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Pitch
axes[1].plot(time_axis, pitch_pred_zoom, label='Estimated Pitch', linestyle='-', linewidth=1.5, color='#ff7f0e')
axes[1].plot(time_axis, pitch_truth_zoom, label='Ground Truth', linestyle='--', linewidth=1.5, color='#1f77b4')
axes[1].set_ylabel('Pitch Angle (degrees)', fontsize=16)
axes[1].set_xlabel('Sample Index', fontsize=16)
axes[1].legend(loc='upper right', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Add RMSE text for pitch
axes[1].text(0.02, 0.95, f'Pitch RMSE: {config["pitch_rmse"]:.3f} deg', transform=axes[1].transAxes,
             fontsize=13, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.tight_layout()

# Save the plot
output_file = f'Results/Figures/{args.filter.capitalize()}_RollPitch_{start_idx}_{end_idx}.png'
plt.savefig(output_file, dpi=150)
print(f"Plot saved: {output_file}")
