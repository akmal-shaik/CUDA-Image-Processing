#include "cuda_filters.cuh"
#include <cuda_runtime.h>
#include <cmath>

__constant__ int sobel_gx[9] = {
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

__constant__ int sobel_gy[9] = {
	-1, -2, -1,
	 0,  0,  0,
	 1,  2,  1
};

__global__ void sobel_kernel(
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

	int gx = 0;
	int gy = 0;

	for (int ky = -1; ky <= 1; ky++)
	{
		for (int kx = -1; kx <= 1; kx++)
		{
			int neighbour_x = x + kx;
			int neighbour_y = y + ky;

			if (neighbour_x < 0) neighbour_x = 0;
			if (neighbour_x >= width) neighbour_x = width - 1;
			if (neighbour_y < 0) neighbour_y = 0;
			if (neighbour_y >= height) neighbour_y = height - 1;

			int input_index = neighbour_y * width + neighbour_x;
			int kernel_index = (ky + 1) * 3 + (kx + 1);

			int pixel = input[input_index];

			gx += pixel * sobel_gx[kernel_index];
			gy += pixel * sobel_gy[kernel_index];
		}
	}

	int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));

	if (magnitude > 255)
	{
		magnitude = 255;
	}

	int output_index = y * width + x;
	output[output_index] = static_cast<unsigned char>(magnitude);
}

void sobel_cuda(
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

	sobel_kernel<<<blocks, threads_per_block>>>(
		device_input,
		device_output,
		width,
		height
	);

	cudaDeviceSynchronize();

	cudaMemcpy(output, device_output, image_bytes, cudaMemcpyDeviceToHost);

	cudaFree(device_input);
	cudaFree(device_output);
}

