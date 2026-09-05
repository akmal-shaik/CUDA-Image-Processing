#include "cuda_filters.cuh"
#include <cuda_runtime.h>

__constant__ int gaussian_kernel[25] = {
	1,  4,  6,  4, 1,
	4, 16, 24, 16, 4,
	6, 24, 36, 24, 6,
	4, 16, 24, 16, 4,
	1,  4,  6,  4, 1
};

__global__ void gaussian_blur_kernel(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	int sum = 0;

	for (int ky = -2; ky <= 2; ky++)
	{
		for (int kx = -2; kx <= 2; kx++)
		{
			int neighbour_x = x + kx;
			int neighbour_y = y + ky;

			if (neighbour_x < 0) neighbour_x = 0;
			if (neighbour_x >= width) neighbour_x = width - 1;

			if (neighbour_y < 0) neighbour_y = 0;
			if (neighbour_y >= height) neighbour_y = height - 1;

			int input_index = neighbour_y * width + neighbour_x;
			int kernel_index = (ky + 2) * 5 + (kx + 2);

			sum += input[input_index] * gaussian_kernel[kernel_index];
		}
	}

	int output_index = y * width + x;
	output[output_index] = static_cast<unsigned char>(sum / 256);
}

void launch_gaussian_blur_kernel(
	const unsigned char* device_input,
	unsigned char* device_output,
	int width,
	int height
)
{
	dim3 threads_per_block(16, 16);

	dim3 blocks(
		(width + threads_per_block.x - 1) / threads_per_block.x,
		(height + threads_per_block.y - 1) / threads_per_block.y
	);

	gaussian_blur_kernel<<<blocks, threads_per_block>>>(device_input, device_output, width, height);
}

void gaussian_blur_cuda(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
)
{
	int num_pixels = width * height;
	size_t image_bytes = num_pixels * sizeof(unsigned char);

	unsigned char* device_input;
	unsigned char* device_output;

	cudaMalloc(&device_input, image_bytes);
	cudaMalloc(&device_output, image_bytes);

	cudaMemcpy(device_input, input, image_bytes, cudaMemcpyHostToDevice);

	dim3 threads_per_block(16, 16);

	dim3 blocks(
		(width + threads_per_block.x - 1) / threads_per_block.x,
		(height + threads_per_block.y - 1) / threads_per_block.y
	);

	launch_gaussian_blur_kernel(device_input, device_output, width, height);

	cudaDeviceSynchronize();

	cudaMemcpy(output, device_output, image_bytes, cudaMemcpyDeviceToHost);

	cudaFree(device_input);
	cudaFree(device_output);
}
