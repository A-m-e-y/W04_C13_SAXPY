import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("timing_cpuVgpu.csv")

powers = df['PowerOf2']
labels = [f"2^{p}" for p in powers]
x = np.arange(len(powers))

# Times
cpu = df['CPUTime_ms']
gpu_kernel = df['KernelTime_ms']
gpu_total = df['TotalTime_ms']

bar_width = 0.25

# Plot setup
plt.figure(figsize=(14, 6))

# Bars
plt.bar(x - bar_width, cpu, width=bar_width, label="CPU Time (ms)", color='lightgreen')
plt.bar(x, gpu_kernel, width=bar_width, label="GPU Kernel Time (ms)", color='skyblue')
plt.bar(x + bar_width, gpu_total, width=bar_width, label="GPU Total Time (ms)", color='salmon')

# Annotations
def annotate_bars(values, x_pos, offset=bar_width):
    for i, val in enumerate(values):
        plt.text(x_pos[i], val, f"{val:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)

annotate_bars(cpu, x - bar_width)
annotate_bars(gpu_kernel, x)
annotate_bars(gpu_total, x + bar_width)

# Log scale
plt.yscale("log")
plt.ylabel("Execution Time (ms) [log scale]")
plt.xlabel("N (as power of 2)")
plt.title("SAXPY: CPU vs GPU Execution Times - RTX-3050 vs i5-12450H")
plt.xticks(x, labels)
# plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("plots/saxpy_full_benchmark_plot.png")
plt.show()
