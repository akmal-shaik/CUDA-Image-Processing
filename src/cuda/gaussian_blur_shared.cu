#include "cuda_filters.cuh"
#include <cuda_runtime.h>

constexpr int BLUR_BLOCK_SIZE = 16;
constexpr int BLUR_RADIUS = 2;
constexpr int BLUR_TILE_SIZE = BLUR_BLOCK_SIZE + 2 * BLUR_RADIUS;

__constant__ int gaussian_shared_weights[25] = {
	1,  4,  6,  4, 1,
	4, 16, 24, 16, 4,
	6, 24, 36, 24, 6,
	4, 16, 24, 16, 4,
	1,  4,  6,  4, 1
};

__global__ void gaussian_blur_shared_kernel(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
)
{
	__shared__ unsigned char tile[BLUR_TILE_SIZE][BLUR_TILE_SIZE];

	int x = blockIdx.x * BLUR_BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLUR_BLOCK_SIZE + threadIdx.y;

	int local_thread = threadIdx.y * BLUR_BLOCK_SIZE + threadIdx.x;
	int threads_per_block = BLUR_BLOCK_SIZE * BLUR_BLOCK_SIZE;
	int tile_elements = BLUR_TILE_SIZE * BLUR_TILE_SIZE;

	for (int index = local_thread; index < tile_elements; index += threads_per_block)
	{
		int tile_x = index % BLUR_TILE_SIZE;
		int tile_y = index / BLUR_TILE_SIZE;

		int global_x = blockIdx.x * BLUR_BLOCK_SIZE + tile_x - BLUR_RADIUS;
		int global_y = blockIdx.y * BLUR_BLOCK_SIZE + tile_y - BLUR_RADIUS;

		if (global_x < 0) global_x = 0;
		if (global_x >= width) global_x = width - 1;
		if (global_y < 0) global_y = 0;
		if (global_y >= height) global_y = height - 1;

		tile[tile_y][tile_x] = input[global_y * width + global_x];
	}

	__syncthreads();

	if (x >= width || y >= height)
	{
		return;
	}

	int sum = 0;

	for (int ky = -2; ky <= 2; ky++)
	{
		for (int kx = -2; kx <= 2; kx++)
		{
			int tile_x = threadIdx.x + BLUR_RADIUS + kx;
			int tile_y = threadIdx.y + BLUR_RADIUS + ky;
			int kernel_index = (ky + 2) * 5 + (kx + 2);

			sum += tile[tile_y][tile_x] * gaussian_shared_weights[kernel_index];
		}
	}

	output[y * width + x] = static_cast<unsigned char>(sum / 256);
}

void launch_gaussian_blur_shared_kernel(
	const unsigned char* device_input,
	unsigned char* device_output,
	int width,
	int height
)
{
	dim3 threads_per_block(BLUR_BLOCK_SIZE, BLUR_BLOCK_SIZE);

	dim3 blocks(
		(width + BLUR_BLOCK_SIZE - 1) / BLUR_BLOCK_SIZE,
		(height + BLUR_BLOCK_SIZE - 1) / BLUR_BLOCK_SIZE
	);

	gaussian_blur_shared_kernel<<<blocks, threads_per_block>>>(device_input, device_output, width, height);
}
