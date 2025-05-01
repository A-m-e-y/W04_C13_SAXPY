import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
df = pd.read_csv("timing.csv")

# Extract columns
x = df['N']
kernel = df['KernelTime_ms']
total = df['TotalTime_ms']
xticks_labels = [f"2^{p}" for p in x]

# Bar positions
bar_width = 0.35
x_indices = np.arange(len(x))

# Plot
plt.figure(figsize=(12, 6))
plt.bar(x_indices - bar_width/2, kernel, width=bar_width, label='Kernel Time (ms)', color='skyblue')
plt.bar(x_indices + bar_width/2, total, width=bar_width, label='Total Time (ms)', color='salmon')

plt.yscale('log')
plt.xlabel("log₂(N)")
plt.ylabel("Time (ms) (log scale)")
plt.title("SAXPY: Kernel vs Total Execution Time (Log Y-Axis) - RTX-3050 4GB VRAM")
plt.xticks(x_indices, xticks_labels)
plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("plots/saxpy_batch_gpu_plot.png")
plt.show()
