import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
df = pd.read_csv("timing_gpu_details.csv")

# X-axis: powers of 2
powers = df['PowerOf2']
labels = [f"2^{p}" for p in powers]
x = np.arange(len(powers))

# GPU time components
malloc_time = df['MallocTime_ms']
h2d_time = df['H2DTime_ms']
kernel_time = df['KernelTime_ms']
d2h_time = df['D2HTime_ms']
free_time = df['FreeTime_ms']

# Stack components bottom-up
bottom_malloc = malloc_time
bottom_h2d = bottom_malloc + h2d_time
bottom_kernel = bottom_h2d + kernel_time
bottom_d2h = bottom_kernel + d2h_time

# Plot setup
plt.figure(figsize=(14, 7))

# Stacked bars
plt.bar(x, malloc_time, label='cudaMalloc', color='#9c27b0')
plt.bar(x, h2d_time, bottom=malloc_time, label='H2D Memcpy', color='#03a9f4')
plt.bar(x, kernel_time, bottom=bottom_h2d, label='Kernel', color='#4caf50')
plt.bar(x, d2h_time, bottom=bottom_kernel, label='D2H Memcpy', color='#fbc02d')
plt.bar(x, free_time, bottom=bottom_d2h, label='cudaFree', color='#ef5350')

# Labels and formatting
plt.xticks(x, labels)
plt.xlabel("N (as power of 2)")
plt.ylabel("Time (ms)")
plt.title("Breakdown of GPU Total Time - SAXPY (RTX-3050 4GB VRAM)")
plt.legend(loc='upper left')
plt.yscale("log")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/saxpy_gpu_time_breakdown.png")
plt.show()
