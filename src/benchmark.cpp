#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "cpu_filters.hpp"
#include "cuda_filters.cuh"

double median(std::vector<double> values)
{
	std::sort(values.begin(), values.end());

	int size = values.size();

	if (size % 2 == 0)
	{
		return (values[size / 2 - 1] + values[size / 2]) / 2.0;
	}

	return values[size / 2];
}

int main()
{
	const int trials = 10;

	const int resolutions[][2] = {
		{640, 480},
		{1280, 720},
		{1920, 1080},
		{2560, 1440},
		{3840, 2160}
	};

	std::ofstream csv("../benchmarks/results.csv");

	csv << "width,height,cpu_ms,gpu_kernel_ms,gpu_end_to_end_ms,kernel_speedup,end_to_end_speedup\n";

	std::cout << "\n=== CUDA Image Processing Benchmark ===\n\n";

	for (const auto& resolution : resolutions)
	{
		int width = resolution[0];
		int height = resolution[1];
		int num_pixels = width * height;

		std::vector<unsigned char> input(num_pixels * 3);
		std::vector<unsigned char> greyscale_cpu_output(num_pixels);
		std::vector<unsigned char> blur_cpu_output(num_pixels);
		std::vector<unsigned char> sobel_cpu_output(num_pixels);
		std::vector<unsigned char> cuda_output(num_pixels);

		// Generate deterministic RGB test data.
		for (int pixel = 0; pixel < num_pixels; pixel++)
		{
			input[pixel * 3] = static_cast<unsigned char>((pixel * 3) % 256);
			input[pixel * 3 + 1] = static_cast<unsigned char>((pixel * 5) % 256);
			input[pixel * 3 + 2] = static_cast<unsigned char>((pixel * 7) % 256);
		}

		// Warm-up CPU.
		grayscale_cpu(input.data(), greyscale_cpu_output.data(), width, height);
		gaussian_blur_cpu(greyscale_cpu_output.data(), blur_cpu_output.data(), width, height);
		sobel_cpu(blur_cpu_output.data(), sobel_cpu_output.data(), width, height);

		// Warm-up GPU.
		float warmup_kernel_ms = 0.0f;
		double warmup_end_to_end_ms = 0.0;

		run_cuda_pipeline(
			input.data(),
			cuda_output.data(),
			width,
			height,
			&warmup_kernel_ms,
			&warmup_end_to_end_ms
		);

		std::vector<double> cpu_times;
		std::vector<double> gpu_kernel_times;
		std::vector<double> gpu_end_to_end_times;

		// CPU benchmark.
		for (int trial = 0; trial < trials; trial++)
		{
			auto start = std::chrono::steady_clock::now();

			grayscale_cpu(input.data(), greyscale_cpu_output.data(), width, height);
			gaussian_blur_cpu(greyscale_cpu_output.data(), blur_cpu_output.data(), width, height);
			sobel_cpu(blur_cpu_output.data(), sobel_cpu_output.data(), width, height);

			auto stop = std::chrono::steady_clock::now();

			double cpu_ms = std::chrono::duration<double, std::milli>(stop - start).count();
			cpu_times.push_back(cpu_ms);
		}

		// GPU benchmark.
		for (int trial = 0; trial < trials; trial++)
		{
			float kernel_ms = 0.0f;
			double end_to_end_ms = 0.0;

			run_cuda_pipeline(
				input.data(),
				cuda_output.data(),
				width,
				height,
				&kernel_ms,
				&end_to_end_ms
			);

			gpu_kernel_times.push_back(kernel_ms);
			gpu_end_to_end_times.push_back(end_to_end_ms);
		}

		double cpu_ms = median(cpu_times);
		double gpu_kernel_ms = median(gpu_kernel_times);
		double gpu_end_to_end_ms = median(gpu_end_to_end_times);

		double kernel_speedup = cpu_ms / gpu_kernel_ms;
		double end_to_end_speedup = cpu_ms / gpu_end_to_end_ms;

		std::cout << width << "x" << height << '\n';
		std::cout << "\tCPU: " << cpu_ms << " ms\n";
		std::cout << "\tGPU kernels: " << gpu_kernel_ms << " ms\n";
		std::cout << "\tGPU end-to-end: " << gpu_end_to_end_ms << " ms\n";
		std::cout << "\tKernel speed-up: " << kernel_speedup << "x\n";
		std::cout << "\tEnd-to-end speed-up: " << end_to_end_speedup << "x\n\n";

		csv << width << ','
			<< height << ','
			<< cpu_ms << ','
			<< gpu_kernel_ms << ','
			<< gpu_end_to_end_ms << ','
			<< kernel_speedup << ','
			<< end_to_end_speedup << '\n';
	}

	csv.close();

	std::cout << "Benchmark results saved to benchmarks/results.csv\n";

	return 0;
}
