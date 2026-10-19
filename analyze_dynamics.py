import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Analyze filter performance vs sensor dynamics')
parser.add_argument('angle', choices=['roll', 'pitch'], help='Angle to analyze (roll or pitch)')
args = parser.parse_args()

angle = args.angle
angle_cap = angle.capitalize()

# Load datasets based on angle selection
if angle == 'roll':
    dataComplementary = np.loadtxt('Results/Results/ComplementaryRoll_a_0_79.txt')
    dataMahony = np.loadtxt('Results/Results/MahonyRoll_kp_11.txt')
    dataEKF = np.loadtxt('Results/Results/EkfRoll.txt')
    dataExplicitCF = np.loadtxt('Results/Results/ExplicitComplementaryRoll.txt')
    dataGroundTruth = np.loadtxt('Results/ExpectedResults/expected_roll.txt')
else:  # pitch
    dataComplementary = np.loadtxt('Results/Results/ComplementaryPitch_a_0_79.txt')
    dataMahony = np.loadtxt('Results/Results/MahonyPitch_kp_11.txt')
    dataEKF = np.loadtxt('Results/Results/EkfPitch.txt')
    dataExplicitCF = np.loadtxt('Results/Results/ExplicitComplementaryPitch.txt')
    dataGroundTruth = np.loadtxt('Results/ExpectedResults/expected_pitch.txt')

# Load sensor data
gyroData = np.loadtxt('Data/gyro.csv', delimiter=',')
accelData = np.loadtxt('Data/accel.csv', delimiter=',')

# Same sample window as before
total_samples = len(dataGroundTruth)
sample_size = total_samples
start_idx = 0
end_idx = total_samples

# Extract samples
comp_sample = dataComplementary[start_idx:end_idx]
mahony_sample = dataMahony[start_idx:end_idx]
ekf_sample = dataEKF[start_idx:end_idx]
explicitCF_sample = dataExplicitCF[start_idx:end_idx]
truth_sample = dataGroundTruth[start_idx:end_idx]
time_indices = np.arange(start_idx, end_idx)

# Calculate errors
error_complementary = comp_sample - truth_sample
error_mahony = mahony_sample - truth_sample
error_ekf = ekf_sample - truth_sample
error_explicitCF = explicitCF_sample - truth_sample

# Absolute errors for plotting
abs_error_complementary = np.abs(error_complementary)
abs_error_mahony = np.abs(error_mahony)
abs_error_ekf = np.abs(error_ekf)
abs_error_explicitCF = np.abs(error_explicitCF)

# Extract gyro and accel for this window
gyro_sample = gyroData[start_idx:end_idx, :]
accel_sample = accelData[start_idx:end_idx, :]

# Calculate gyro magnitude (total angular rate)
gyro_magnitude = np.linalg.norm(gyro_sample, axis=1)

# Calculate accel magnitude
accel_magnitude = np.linalg.norm(accel_sample, axis=1)

# Calculate RMSE for legend labels
rmse_complementary = np.sqrt(np.mean(error_complementary**2))
rmse_mahony = np.sqrt(np.mean(error_mahony**2))
rmse_ekf = np.sqrt(np.mean(error_ekf**2))
rmse_explicitCF = np.sqrt(np.mean(error_explicitCF**2))

# Create multi-panel figure
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Estimation errors
panel_number = 0
axes[panel_number].plot(time_indices, abs_error_complementary, 'b-', linewidth=1.5, label=f'Complementary - RMSE: {rmse_complementary:.3f} deg', alpha=0.7)
axes[panel_number].plot(time_indices, abs_error_mahony, 'r-', linewidth=1.5, label=f'Mahony - RMSE: {rmse_mahony:.3f} deg', alpha=0.7)
axes[panel_number].plot(time_indices, abs_error_ekf, 'g-', linewidth=1.5, label=f'EKF - RMSE: {rmse_ekf:.3f} deg', alpha=0.7)
axes[panel_number].plot(time_indices, abs_error_explicitCF, 'c-', linewidth=1.5, label=f'Explicit CF - RMSE: {rmse_explicitCF:.3f} deg', alpha=0.7)
axes[panel_number].set_ylabel('Absolute Error (deg)', fontsize=11)
axes[panel_number].set_title(f'{angle_cap} Angle Estimation Error with Sensor Dynamics', fontsize=13)
axes[panel_number].legend(loc='best')
axes[panel_number].grid(True, alpha=0.3)

# Panel 2: Gyro magnitude (angular rate)
panel_number = 1
axes[panel_number].plot(time_indices, gyro_magnitude, 'g-', linewidth=1.5, label='Gyro Magnitude')
axes[panel_number].set_ylabel('Angular Rate (rad/s)', fontsize=11)
axes[panel_number].legend(loc='best')
axes[panel_number].grid(True, alpha=0.3)

# Panel 3: Accel magnitude
panel_number = 2
axes[panel_number].plot(time_indices, accel_magnitude, 'm-', linewidth=1.5, label='Accel Magnitude')
axes[panel_number].set_ylabel('Acceleration (m/s²)', fontsize=11)
axes[panel_number].set_xlabel('Sample Index', fontsize=12)
axes[panel_number].legend(loc='best')
axes[panel_number].grid(True, alpha=0.3)

plt.tight_layout()
output_file = f"Results/Figures/{angle_cap}_Dynamics_Analysis.png"
plt.savefig(output_file, dpi=150)
print(f"Dynamics analysis plot saved: {output_file}")

# Print RMSE statistics
print(f"\n=== {angle_cap} Performance Analysis ===")
print(f"Sample window: {start_idx} to {end_idx} ({sample_size} samples)")
print(f"\nComplementary Filter:")
print(f"  RMSE: {rmse_complementary:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_complementary):.4f} deg")
print(f"\nMahony Filter:")
print(f"  RMSE: {rmse_mahony:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_mahony):.4f} deg")
print(f"\nEKF:")
print(f"  RMSE: {rmse_ekf:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_ekf):.4f} deg")
print(f"\nExplicit Complementary Filter:")
print(f"  RMSE: {rmse_explicitCF:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_explicitCF):.4f} deg")

# Identify high-dynamics periods (top 25% gyro magnitude)
high_dynamics_threshold = np.percentile(gyro_magnitude, 75)
high_dynamics_mask = gyro_magnitude > high_dynamics_threshold
low_dynamics_mask = ~high_dynamics_mask

# Calculate RMSE for high/low dynamics periods
rmse_comp_high = np.sqrt(np.mean(error_complementary[high_dynamics_mask]**2))
rmse_mahony_high = np.sqrt(np.mean(error_mahony[high_dynamics_mask]**2))
rmse_ekf_high = np.sqrt(np.mean(error_ekf[high_dynamics_mask]**2))
rmse_explicitCF_high = np.sqrt(np.mean(error_explicitCF[high_dynamics_mask]**2))
rmse_comp_low = np.sqrt(np.mean(error_complementary[low_dynamics_mask]**2))
rmse_mahony_low = np.sqrt(np.mean(error_mahony[low_dynamics_mask]**2))
rmse_ekf_low = np.sqrt(np.mean(error_ekf[low_dynamics_mask]**2))
rmse_explicitCF_low = np.sqrt(np.mean(error_explicitCF[low_dynamics_mask]**2))

print(f"\n=== High Dynamics vs Low Dynamics Performance ===")
print(f"High dynamics threshold: {high_dynamics_threshold:.4f} rad/s")
print(f"High dynamics samples: {np.sum(high_dynamics_mask)} ({np.sum(high_dynamics_mask)/len(high_dynamics_mask)*100:.1f}%)")

print(f"\nHigh Dynamics Period:")
print(f"  Complementary RMSE: {rmse_comp_high:.4f} deg")
print(f"  Mahony RMSE: {rmse_mahony_high:.4f} deg")
print(f"  EKF RMSE: {rmse_ekf_high:.4f} deg")
print(f"  Explicit CF RMSE: {rmse_explicitCF_high:.4f} deg")
high_rmses = {'Complementary': rmse_comp_high, 'Mahony': rmse_mahony_high, 'EKF': rmse_ekf_high, 'Explicit CF': rmse_explicitCF_high}
best_high = min(high_rmses, key=high_rmses.get)
second_best_high = sorted(high_rmses.values())[1]
print(f"  Best performer: {best_high} (advantage: {second_best_high - high_rmses[best_high]:.4f} deg)")

print(f"\nLow Dynamics Period:")
print(f"  Complementary RMSE: {rmse_comp_low:.4f} deg")
print(f"  Mahony RMSE: {rmse_mahony_low:.4f} deg")
print(f"  EKF RMSE: {rmse_ekf_low:.4f} deg")
print(f"  Explicit CF RMSE: {rmse_explicitCF_low:.4f} deg")
low_rmses = {'Complementary': rmse_comp_low, 'Mahony': rmse_mahony_low, 'EKF': rmse_ekf_low, 'Explicit CF': rmse_explicitCF_low}
best_low = min(low_rmses, key=low_rmses.get)
second_best_low = sorted(low_rmses.values())[1]
print(f"  Best performer: {best_low} (advantage: {second_best_low - low_rmses[best_low]:.4f} deg)")
