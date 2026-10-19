"""
Grouped Bar Chart: Overall Test RMSE across 6 LSTM configurations.

Groups bars by prediction horizon (N=10 vs N=30) and look-back window (K=10 vs K=50).
Colors distinguish Attention vs No Attention models.

Models:
  A: Attn,    K=10, N=10  -> 0.437
  B: No Attn, K=10, N=10  -> 0.444
  C: Attn,    K=10, N=30  -> 0.521
  D: No Attn, K=10, N=30  -> 0.502
  E: Attn,    K=50, N=30  -> 0.420
  F: No Attn, K=50, N=30  -> 0.440

Usage:
    python3 plot_rmse_bar_chart.py

Output:
    Results/LSTM/rmse_bar_chart.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data ---
groups = ['N=10, K=10', 'N=30, K=10', 'N=30, K=50']
attention_rmse = [0.437, 0.521, 0.420]   # Models A, C, E
no_attention_rmse = [0.444, 0.502, 0.440] # Models B, D, F
attention_labels = ['A', 'C', 'E']
no_attention_labels = ['B', 'D', 'F']

# --- Style ---
COLOR_ATTN = '#2166ac'      # Blue for attention
COLOR_NO_ATTN = '#b2182b'   # Red for no attention
BAR_WIDTH = 0.32

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(groups))

bars_attn = ax.bar(x - BAR_WIDTH/2, attention_rmse, BAR_WIDTH,
                   label='With Attention', color=COLOR_ATTN, edgecolor='black', linewidth=0.8)
bars_no = ax.bar(x + BAR_WIDTH/2, no_attention_rmse, BAR_WIDTH,
                 label='Without Attention', color=COLOR_NO_ATTN, edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bar, label in zip(bars_attn, attention_labels):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.004,
            f'{label}\n{height:.3f}°', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar, label in zip(bars_no, no_attention_labels):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.004,
            f'{label}\n{height:.3f}°', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best model (E)
bars_attn[2].set_edgecolor('#ff7f00')
bars_attn[2].set_linewidth(2.5)

ax.set_ylabel('Overall Test RMSE (degrees)', fontsize=14)
ax.set_xlabel('Configuration (Horizon N, Look-back K)', fontsize=14)
ax.set_title('LSTM Model Comparison: Effect of Attention, Look-back Window, and Prediction Horizon',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=13)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=13, loc='upper left')

# Set y-axis to start near the data range for better visual contrast
ax.set_ylim(0.38, 0.56)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add a subtle annotation for the best model
ax.annotate('Best', xy=(x[2] - BAR_WIDTH/2, 0.420),
            xytext=(x[2] - BAR_WIDTH/2 - 0.35, 0.39),
            fontsize=11, fontweight='bold', color='#ff7f00',
            arrowprops=dict(arrowstyle='->', color='#ff7f00', lw=1.5))

plt.tight_layout()

# Save
os.makedirs('Results/LSTM', exist_ok=True)
output_path = 'Results/LSTM/rmse_bar_chart.png'
fig.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.show()
