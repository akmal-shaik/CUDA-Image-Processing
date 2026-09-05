#include "cuda_filters.cuh"
#include <cuda_runtime.h>
#include <chrono>

void run_cuda_pipeline(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	float* kernel_ms,
	double* end_to_end_ms
)
{
	int num_pixels = width * height;

	size_t rgb_bytes = num_pixels * 3 * sizeof(unsigned char);
	size_t grey_bytes = num_pixels * sizeof(unsigned char);

	unsigned char* device_rgb;
	unsigned char* device_greyscale;
	unsigned char* device_blur;
	unsigned char* device_sobel;

	cudaMalloc(&device_rgb, rgb_bytes);
	cudaMalloc(&device_greyscale, grey_bytes);
	cudaMalloc(&device_blur, grey_bytes);
	cudaMalloc(&device_sobel, grey_bytes);

	cudaEvent_t start_event;
	cudaEvent_t stop_event;

	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	auto end_to_end_start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(device_rgb, input, rgb_bytes, cudaMemcpyHostToDevice);

	cudaEventRecord(start_event);

	launch_grayscale_kernel(device_rgb, device_greyscale, width, height);
	launch_gaussian_blur_kernel(device_greyscale, device_blur, width, height);
	launch_sobel_kernel(device_blur, device_sobel, width, height);

	cudaEventRecord(stop_event);

	cudaMemcpy(output, device_sobel, grey_bytes, cudaMemcpyDeviceToHost);

	auto end_to_end_stop = std::chrono::high_resolution_clock::now();

	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(kernel_ms, start_event, stop_event);

	*end_to_end_ms = std::chrono::duration<double, std::milli>(end_to_end_stop - end_to_end_start).count();

	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

	cudaFree(device_rgb);
	cudaFree(device_greyscale);
	cudaFree(device_blur);
	cudaFree(device_sobel);
}
