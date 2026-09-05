#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda_filters.cuh"

float median(std::vector<float> values)
{
	std::sort(values.begin(), values.end());

	int size = values.size();

	if (size % 2 == 0)
	{
		return (values[size / 2 - 1] + values[size / 2]) / 2.0f;
	}

	return values[size / 2];
}

int main()
{
	int width = 3840;
	int height = 2160;
	int num_pixels = width * height;
	int trials = 20;

	size_t rgb_bytes = num_pixels * 3 * sizeof(unsigned char);
	size_t grey_bytes = num_pixels * sizeof(unsigned char);
	size_t temp_bytes = num_pixels * sizeof(int);

	std::vector<unsigned char> input(num_pixels * 3);
	std::vector<unsigned char> baseline_output(num_pixels);
	std::vector<unsigned char> shared_output(num_pixels);
	std::vector<unsigned char> separable_output(num_pixels);

	for (int pixel = 0; pixel < num_pixels; pixel++)
	{
		input[pixel * 3] = static_cast<unsigned char>((pixel * 3) % 256);
		input[pixel * 3 + 1] = static_cast<unsigned char>((pixel * 5) % 256);
		input[pixel * 3 + 2] = static_cast<unsigned char>((pixel * 7) % 256);
	}

	unsigned char* device_rgb;
	unsigned char* device_greyscale;
	unsigned char* device_blur_baseline;
	unsigned char* device_blur_shared;
	unsigned char* device_blur_separable;
	int* device_temp;

	cudaMalloc(&device_rgb, rgb_bytes);
	cudaMalloc(&device_greyscale, grey_bytes);
	cudaMalloc(&device_blur_baseline, grey_bytes);
	cudaMalloc(&device_blur_shared, grey_bytes);
	cudaMalloc(&device_blur_separable, grey_bytes);
	cudaMalloc(&device_temp, temp_bytes);

	cudaMemcpy(device_rgb, input.data(), rgb_bytes, cudaMemcpyHostToDevice);

	// Generate one identical greyscale input for every blur implementation.
	launch_grayscale_kernel(device_rgb, device_greyscale, width, height);
	cudaDeviceSynchronize();

	// Warm-up.
	launch_gaussian_blur_kernel(device_greyscale, device_blur_baseline, width, height);
	launch_gaussian_blur_shared_kernel(device_greyscale, device_blur_shared, width, height);
	launch_gaussian_blur_separable_kernel(device_greyscale, device_temp, device_blur_separable, width, height);
	cudaDeviceSynchronize();

	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	std::vector<float> baseline_times;
	std::vector<float> shared_times;
	std::vector<float> separable_times;

	for (int trial = 0; trial < trials; trial++)
	{
		float elapsed_ms = 0.0f;

		cudaEventRecord(start);
		launch_gaussian_blur_kernel(device_greyscale, device_blur_baseline, width, height);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_ms, start, stop);
		baseline_times.push_back(elapsed_ms);

		cudaEventRecord(start);
		launch_gaussian_blur_shared_kernel(device_greyscale, device_blur_shared, width, height);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_ms, start, stop);
		shared_times.push_back(elapsed_ms);

		cudaEventRecord(start);
		launch_gaussian_blur_separable_kernel(device_greyscale, device_temp, device_blur_separable, width, height);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_ms, start, stop);
		separable_times.push_back(elapsed_ms);
	}

	float baseline_ms = median(baseline_times);
	float shared_ms = median(shared_times);
	float separable_ms = median(separable_times);

	cudaMemcpy(baseline_output.data(), device_blur_baseline, grey_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(shared_output.data(), device_blur_shared, grey_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(separable_output.data(), device_blur_separable, grey_bytes, cudaMemcpyDeviceToHost);

	int shared_mismatches = 0;
	int shared_max_difference = 0;

	int separable_mismatches = 0;
	int separable_max_difference = 0;

	for (int pixel = 0; pixel < num_pixels; pixel++)
	{
		int shared_difference = std::abs(static_cast<int>(baseline_output[pixel]) - static_cast<int>(shared_output[pixel]));
		int separable_difference = std::abs(static_cast<int>(baseline_output[pixel]) - static_cast<int>(separable_output[pixel]));

		if (shared_difference > 0)
		{
			shared_mismatches++;
		}

		if (shared_difference > shared_max_difference)
		{
			shared_max_difference = shared_difference;
		}

		if (separable_difference > 0)
		{
			separable_mismatches++;
		}

		if (separable_difference > separable_max_difference)
		{
			separable_max_difference = separable_difference;
		}
	}

	double shared_speedup = baseline_ms / shared_ms;
	double separable_speedup = baseline_ms / separable_ms;

	double shared_reduction = ((baseline_ms - shared_ms) / baseline_ms) * 100.0;
	double separable_reduction = ((baseline_ms - separable_ms) / baseline_ms) * 100.0;

	std::cout << "\n=== 4K Gaussian Blur Optimisation ===\n";
	std::cout << "Baseline 5x5: " << baseline_ms << " ms\n";
	std::cout << "Shared-memory 5x5: " << shared_ms << " ms\n";
	std::cout << "Separable 5+5: " << separable_ms << " ms\n";

	std::cout << "\n=== Performance vs Baseline ===\n";
	std::cout << "Shared-memory speed-up: " << shared_speedup << "x\n";
	std::cout << "Shared-memory time reduction: " << shared_reduction << "%\n";
	std::cout << "Separable speed-up: " << separable_speedup << "x\n";
	std::cout << "Separable time reduction: " << separable_reduction << "%\n";

	std::cout << "\n=== Correctness Validation ===\n";
	std::cout << "Shared-memory mismatches: " << shared_mismatches << '\n';
	std::cout << "Shared-memory max difference: " << shared_max_difference << '\n';
	std::cout << "Separable mismatches: " << separable_mismatches << '\n';
	std::cout << "Separable max difference: " << separable_max_difference << '\n';

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(device_rgb);
	cudaFree(device_greyscale);
	cudaFree(device_blur_baseline);
	cudaFree(device_blur_shared);
	cudaFree(device_blur_separable);
	cudaFree(device_temp);

	return 0;
}
