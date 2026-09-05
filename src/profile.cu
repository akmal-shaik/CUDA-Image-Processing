#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "cuda_filters.cuh"

int main()
{
	int width = 3840;
	int height = 2160;
	int num_pixels = width * height;

	size_t rgb_bytes = num_pixels * 3 * sizeof(unsigned char);
	size_t grey_bytes = num_pixels * sizeof(unsigned char);

	std::vector<unsigned char> input(num_pixels * 3);

	for (int pixel = 0; pixel < num_pixels; pixel++)
	{
		input[pixel * 3] = static_cast<unsigned char>((pixel * 3) % 256);
		input[pixel * 3 + 1] = static_cast<unsigned char>((pixel * 5) % 256);
		input[pixel * 3 + 2] = static_cast<unsigned char>((pixel * 7) % 256);
	}

	unsigned char* device_rgb;
	unsigned char* device_greyscale;
	unsigned char* device_blur;
	unsigned char* device_sobel;

	cudaMalloc(&device_rgb, rgb_bytes);
	cudaMalloc(&device_greyscale, grey_bytes);
	cudaMalloc(&device_blur, grey_bytes);
	cudaMalloc(&device_sobel, grey_bytes);

	cudaMemcpy(device_rgb, input.data(), rgb_bytes, cudaMemcpyHostToDevice);

	// Warm-up
	launch_grayscale_kernel(device_rgb, device_greyscale, width, height);
	launch_gaussian_blur_kernel(device_greyscale, device_blur, width, height);
	launch_sobel_kernel(device_blur, device_sobel, width, height);
	cudaDeviceSynchronize();

	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float greyscale_ms;
	float blur_ms;
	float sobel_ms;

	// Greyscale
	cudaEventRecord(start);
	launch_grayscale_kernel(device_rgb, device_greyscale, width, height);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&greyscale_ms, start, stop);

	// Gaussian blur
	cudaEventRecord(start);
	launch_gaussian_blur_kernel(device_greyscale, device_blur, width, height);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&blur_ms, start, stop);

	// Sobel
	cudaEventRecord(start);
	launch_sobel_kernel(device_blur, device_sobel, width, height);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&sobel_ms, start, stop);

	float total_ms = greyscale_ms + blur_ms + sobel_ms;

	std::cout << "\n=== 4K CUDA Kernel Profile ===\n";
	std::cout << "Greyscale: " << greyscale_ms << " ms\n";
	std::cout << "Gaussian blur: " << blur_ms << " ms\n";
	std::cout << "Sobel: " << sobel_ms << " ms\n";
	std::cout << "Total: " << total_ms << " ms\n";

	std::cout << "\n=== Kernel Time Share ===\n";
	std::cout << "Greyscale: " << (greyscale_ms / total_ms) * 100.0f << "%\n";
	std::cout << "Gaussian blur: " << (blur_ms / total_ms) * 100.0f << "%\n";
	std::cout << "Sobel: " << (sobel_ms / total_ms) * 100.0f << "%\n";

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(device_rgb);
	cudaFree(device_greyscale);
	cudaFree(device_blur);
	cudaFree(device_sobel);

	return 0;
}
