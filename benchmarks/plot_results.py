import csv
from pathlib import Path

import matplotlib.pyplot as plt


benchmark_dir = Path(__file__).parent
csv_path = benchmark_dir / "results.csv"
graphs_dir = benchmark_dir / "graphs"

graphs_dir.mkdir(exist_ok=True)

resolutions = []
cpu_times = []
gpu_kernel_times = []
gpu_end_to_end_times = []
kernel_speedups = []
end_to_end_speedups = []

with open(csv_path, newline="") as file:
	reader = csv.DictReader(file)

	for row in reader:
		width = int(row["width"])
		height = int(row["height"])

		resolutions.append(f"{width}×{height}")
		cpu_times.append(float(row["cpu_ms"]))
		gpu_kernel_times.append(float(row["gpu_kernel_ms"]))
		gpu_end_to_end_times.append(float(row["gpu_end_to_end_ms"]))
		kernel_speedups.append(float(row["kernel_speedup"]))
		end_to_end_speedups.append(float(row["end_to_end_speedup"]))


plt.figure(figsize=(9, 5.5))

plt.plot(resolutions, cpu_times, marker="o", label="CPU pipeline")
plt.plot(resolutions, gpu_end_to_end_times, marker="o", label="CUDA end-to-end")
plt.plot(resolutions, gpu_kernel_times, marker="o", label="CUDA kernels")

plt.yscale("log")
plt.xlabel("Image resolution")
plt.ylabel("Median execution time (ms)")
plt.title("CPU vs CUDA Image Processing Performance")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(graphs_dir / "execution_times.png", dpi=200)
plt.close()


plt.figure(figsize=(9, 5.5))

plt.plot(resolutions, kernel_speedups, marker="o", label="Kernel-only speed-up")
plt.plot(resolutions, end_to_end_speedups, marker="o", label="End-to-end speed-up")

plt.xlabel("Image resolution")
plt.ylabel("Speed-up over CPU (×)")
plt.title("CUDA Pipeline Speed-up")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(graphs_dir / "speedup.png", dpi=200)
plt.close()


print("Graphs saved successfully:")
print(graphs_dir / "execution_times.png")
print(graphs_dir / "speedup.png")