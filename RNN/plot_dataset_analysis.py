#!/usr/bin/env python3
"""Plot roll, pitch, yaw angles from a UAV dataset CSV file.

Usage:
    python3 plot_dataset_analysis.py                          # default: dataset_1 (grouped)
    python3 plot_dataset_analysis.py ../Data/dataset_2.csv interleaved

Layouts:
    grouped     - [roll, pitch, yaw, wr, wp, wy, tr, tp, ty]  (dataset_1)
    interleaved - [roll, wr, pitch, wp, yaw, wy, tr, tp, ty]  (dataset_2+)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Column indices for the angle columns per layout
LAYOUTS = {
    "grouped":     {"roll": 0, "pitch": 1, "yaw": 2},
    "interleaved": {"roll": 0, "pitch": 2, "yaw": 4},
}

csv_path = sys.argv[1] if len(sys.argv) > 1 else "../Data/dataset_1.csv"
layout   = sys.argv[2] if len(sys.argv) > 2 else "grouped"

if layout not in LAYOUTS:
    print(f"Unknown layout '{layout}'. Use: {list(LAYOUTS.keys())}")
    sys.exit(1)

data = np.loadtxt(csv_path, delimiter=",")
cols = LAYOUTS[layout]
Ts = 0.03
t = np.arange(len(data)) * Ts

roll  = np.degrees(data[:, cols["roll"]])
pitch = np.degrees(data[:, cols["pitch"]])
yaw   = np.degrees(data[:, cols["yaw"]])

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax in axes:
    ax.tick_params(axis='both', labelsize=14)

axes[0].plot(t, roll, color='#1f77b4', linewidth=1.5)
axes[0].set_ylabel(r'Roll $\phi$ (degrees)', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, pitch, color='#ff7f0e', linewidth=1.5)
axes[1].set_ylabel(r'Pitch $\theta$ (degrees)', fontsize=14)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, yaw, color='#2ca02c', linewidth=1.5)
axes[2].set_ylabel(r'Yaw $\psi$ (degrees)', fontsize=14)
axes[2].set_xlabel('Time (s)', fontsize=16)
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

axes[0].set_title(f'UAV Attitude Angles ({len(data)} samples at {1/Ts:.1f} Hz)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("dataset_angles.png", dpi=150, bbox_inches='tight')
plt.savefig("dataset_angles.pdf", bbox_inches='tight')
print(f"Saved: dataset_angles.png and .pdf")
plt.show()
