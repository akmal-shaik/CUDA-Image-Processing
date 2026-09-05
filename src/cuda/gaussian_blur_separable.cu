#include "cuda_filters.cuh"
#include <cuda_runtime.h>

__constant__ int gaussian_1d_weights[5] = {
	1, 4, 6, 4, 1
};

__global__ void gaussian_horizontal_kernel(
	const unsigned char* input,
	int* temp,
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

	for (int k = -2; k <= 2; k++)
	{
		int neighbour_x = x + k;

		if (neighbour_x < 0) neighbour_x = 0;
		if (neighbour_x >= width) neighbour_x = width - 1;

		int input_index = y * width + neighbour_x;
		int weight_index = k + 2;

		sum += input[input_index] * gaussian_1d_weights[weight_index];
	}

	temp[y * width + x] = sum;
}

__global__ void gaussian_vertical_kernel(
	const int* temp,
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

	for (int k = -2; k <= 2; k++)
	{
		int neighbour_y = y + k;

		if (neighbour_y < 0) neighbour_y = 0;
		if (neighbour_y >= height) neighbour_y = height - 1;

		int temp_index = neighbour_y * width + x;
		int weight_index = k + 2;

		sum += temp[temp_index] * gaussian_1d_weights[weight_index];
	}

	output[y * width + x] = static_cast<unsigned char>(sum / 256);
}

void launch_gaussian_blur_separable_kernel(
	const unsigned char* device_input,
	int* device_temp,
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

	gaussian_horizontal_kernel<<<blocks, threads_per_block>>>(device_input, device_temp, width, height);
	gaussian_vertical_kernel<<<blocks, threads_per_block>>>(device_temp, device_output, width, height);
}
