import pandas as pd
import math

dataset = pd.read_csv("all_combined_reordered.csv")

print(f"Loaded dataset {dataset.shape}")


rollMax = dataset[dataset.columns[0]].max()
rollMin = dataset[dataset.columns[0]].min()
print(f"Max Roll = {math.degrees(rollMax)}, Min Roll = {math.degrees(rollMin)}")


pitchMax = dataset[dataset.columns[1]].max()
pitchMin = dataset[dataset.columns[1]].min()
print(f"Max Pitch = {math.degrees(pitchMax)}, Min Pitch = {math.degrees(pitchMin)}")


yawMax = dataset[dataset.columns[2]].max()
yawMin = dataset[dataset.columns[2]].min()


print(f"Max Yaw = {math.degrees(yawMax)}, Min Yaw = {math.degrees(yawMin)}")