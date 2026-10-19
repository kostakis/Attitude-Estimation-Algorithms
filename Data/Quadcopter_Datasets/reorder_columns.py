import pandas as pd

df = pd.read_csv('all_combined.csv', header=None)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Current:  [roll_angle, roll_rate, pitch_angle, pitch_rate, yaw_angle, yaw_rate, roll_torque, pitch_torque, yaw_torque]
# Target:   [roll_angle, pitch_angle, yaw_angle, roll_rate, pitch_rate, yaw_rate, roll_torque, pitch_torque, yaw_torque]
df_reordered = df[[0, 2, 4, 1, 3, 5, 6, 7, 8]]

df_reordered.to_csv('all_combined_reordered.csv', header=False, index=False)
print(f"Saved: all_combined_reordered.csv")
print(f"First row: {df_reordered.iloc[0].values}")

