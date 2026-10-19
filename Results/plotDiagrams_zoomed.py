import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot zoomed comparison of filter estimation vs ground truth')
parser.add_argument('angle', choices=['roll', 'pitch'], help='Angle to plot (roll or pitch)')
parser.add_argument('--filter', choices=['complementary', 'mahony', 'ekf', 'explicitcf'],
                    default='complementary', help='Filter to plot (default: complementary)')
parser.add_argument('--start', type=int, default=300, help='Start sample index (default: 300)')
parser.add_argument('--end', type=int, default=700, help='End sample index (default: 700)')
args = parser.parse_args()

angle = args.angle
angle_cap = angle.capitalize()
start_time = args.start
end_time = args.end

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

# File paths based on angle selection
if angle == 'roll':
    groundTruthFile = 'Results/ExpectedResults/expected_roll.txt'
    predictedFile = config['roll_file']
    rmse = config['roll_rmse']
else:  # pitch
    groundTruthFile = 'Results/ExpectedResults/expected_pitch.txt'
    predictedFile = config['pitch_file']
    rmse = config['pitch_rmse']

# Read data from files
predictedData = np.loadtxt(predictedFile)
groundTruthData = np.loadtxt(groundTruthFile)

# Extract zoomed data
predictedData_zoom = predictedData[start_time:end_time]
groundTruthData_zoom = groundTruthData[start_time:end_time]
time_axis = np.arange(start_time, end_time)

# Create a plot with larger figure size
plt.figure(figsize=(10, 7))
plt.tick_params(axis='both', labelsize=14)

plt.plot(time_axis, predictedData_zoom, label=f'Estimated {angle_cap}', linestyle='-', linewidth=2, markersize=4, color='#ff7f0e')
plt.plot(time_axis, groundTruthData_zoom, label=f'Ground Truth', linestyle='--', linewidth=2, markersize=4, color='#1f77b4')

# Add labels, title, and grid
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel(f'{angle_cap} Angle (degrees)', fontsize=16)
plt.title(f"Zoomed View: {angle_cap} - {config['name']} (samples {start_time}-{end_time})", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)

# Dynamically set y-axis limits with some padding
all_data = np.concatenate([predictedData_zoom, groundTruthData_zoom])
y_min, y_max = all_data.min(), all_data.max()
y_margin = (y_max - y_min) * 0.1  # 10% margin
plt.ylim(y_min - y_margin, y_max + y_margin)

# Display RMSE and param on the plot
plt.text(0.05, 0.95, f'{angle_cap} RMSE (full data): {rmse:.3f} deg',
         transform=plt.gca().transAxes,
         fontsize=14, color='black', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

if config['param']:
    plt.text(0.05, 0.89, config['param'],
             transform=plt.gca().transAxes,
             fontsize=14, color='black', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Reduce whitespace margins
plt.tight_layout()

# Save the plot
output_file = f"Results/Figures/{args.filter.capitalize()}_{angle_cap}_{start_time}_{end_time}.png"
plt.savefig(output_file, dpi=150)
print(f"Zoomed plot saved: {output_file}")
