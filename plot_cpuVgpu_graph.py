import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("timing_cpuVgpu.csv")

# Extract values
powers = df['PowerOf2']
gpu_total = df['TotalTime_ms']
cpu_time = df['CPUTime_ms']
labels = [f"2^{p}" for p in powers]

# Bar positions
x = np.arange(len(powers))
bar_width = 0.35

# Plot setup
plt.figure(figsize=(12, 6))

# Bars
plt.bar(x - bar_width/2, cpu_time, width=bar_width, label="CPU Time (ms)", color='lightgreen')
plt.bar(x + bar_width/2, gpu_total, width=bar_width, label="GPU Total Time (ms)", color='salmon')

# Log scale
plt.yscale("log")
plt.ylabel("Execution Time (ms) [log scale]")
plt.xlabel("N (as power of 2)")
plt.title("SAXPY: CPU vs GPU Total Execution Time (Log Scale) - RTX-3050 4GB VRAM")

# X-axis
plt.xticks(x, labels)
plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("plots/cpu_vs_gpu_total_plot.png")
plt.show()
