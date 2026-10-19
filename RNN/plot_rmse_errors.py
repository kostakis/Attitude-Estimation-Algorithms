import matplotlib.pyplot as plt
import re
import sys

if len(sys.argv) < 2:
    print("Usage: python3 plot_rmse_errors.py <training_log.txt>")
    print("Example: python3 plot_rmse_errors.py threesplitdata.txt")
    sys.exit(1)

log_file = sys.argv[1]

epochs = []
train_losses = []
val_losses = []

with open(log_file, 'r') as f:
    for line in f:
        match = re.match(r'Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))

print(f"Parsed {len(epochs)} epochs from {log_file}")
print(f"Epoch range: {epochs[0]} to {epochs[-1]}")
print(f"Final Train Loss: {train_losses[-1]:.6f}")
print(f"Final Val Loss:   {val_losses[-1]:.6f}")

# Plot train vs val loss
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, train_losses, linewidth=2, label='Train Loss', color='blue')
ax.plot(epochs, val_losses, linewidth=2, label='Validation Loss', color='orange')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

filename = 'Results/LSTM/training_curve.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {filename}")

plt.show()
