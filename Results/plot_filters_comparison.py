import matplotlib.pyplot as plt
import numpy as np

start = 300
end = 500
time_axis = np.arange(start, end)

# Ground truth
gt_roll = np.loadtxt('Results/ExpectedResults/expected_roll.txt')[start:end]
gt_pitch = np.loadtxt('Results/ExpectedResults/expected_pitch.txt')[start:end]

# Filter results
filters = {
    'Complementary': {
        'roll': np.loadtxt('Results/Results/ComplementaryRoll_a_0_79.txt')[start:end],
        'pitch': np.loadtxt('Results/Results/ComplementaryPitch_a_0_79.txt')[start:end],
        'color': '#e41a1c',
    },
    'Mahony': {
        'roll': np.loadtxt('Results/Results/MahonyRoll_kp_11.txt')[start:end],
        'pitch': np.loadtxt('Results/Results/MahonyPitch_kp_11.txt')[start:end],
        'color': '#377eb8',
    },
    'Explicit CF': {
        'roll': np.loadtxt('Results/Results/ExplicitComplementaryRoll.txt')[start:end],
        'pitch': np.loadtxt('Results/Results/ExplicitComplementaryPitch.txt')[start:end],
        'color': '#4daf4a',
    },
    'EKF': {
        'roll': np.loadtxt('Results/Results/EkfRoll.txt')[start:end],
        'pitch': np.loadtxt('Results/Results/EkfPitch.txt')[start:end],
        'color': '#984ea3',
    },
}

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Roll comparison
ax = axes[0]
ax.plot(time_axis, gt_roll, 'k--', linewidth=2, label='Ground Truth')
for name, data in filters.items():
    ax.plot(time_axis, data['roll'], linewidth=1.5, color=data['color'], label=name)
ax.set_ylabel('Roll Angle (degrees)', fontsize=16)
ax.set_title('Roll Estimation — All Classical Filters (Samples 300–500)', fontsize=14)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

# Pitch comparison
ax = axes[1]
ax.plot(time_axis, gt_pitch, 'k--', linewidth=2, label='Ground Truth')
for name, data in filters.items():
    ax.plot(time_axis, data['pitch'], linewidth=1.5, color=data['color'], label=name)
ax.set_ylabel('Pitch Angle (degrees)', fontsize=16)
ax.set_xlabel('Sample Index', fontsize=16)
ax.set_title('Pitch Estimation — All Classical Filters (Samples 300–500)', fontsize=14)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('Results/Figures/AllFilters_Comparison_300_500.png', dpi=150)
print("Saved: Results/Figures/AllFilters_Comparison_300_500.png")
plt.show()
