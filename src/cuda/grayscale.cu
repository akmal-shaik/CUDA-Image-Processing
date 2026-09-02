#include "cuda_filters.cuh"
#include <cuda_runtime.h>

__global__ void grayscale_kernel(
	const unsigned char* input,
	unsigned char* output,
	int num_pixels
)
{
	int pixel = blockIdx.x * blockDim.x + threadIdx.x;

	if( pixel < num_pixels)
        {

    	int rgb_index = pixel * 3;

    	const int r = input[rgb_index];
    	const int g = input[rgb_index + 1];
    	const int b = input[rgb_index + 2];

    	float grayscale = 0.299f * r + 0.587f * g + 0.114f * b;

    	output[pixel] = static_cast<unsigned char>(grayscale);
        }
}

void grayscale_cuda(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
)
{
	int num_pixels = width * height;

	unsigned char* device_input;
	unsigned char* device_output;

	size_t input_bytes = num_pixels * 3 * sizeof(unsigned char);

	size_t output_bytes = num_pixels * sizeof(unsigned char);

	cudaMalloc(&device_input, input_bytes);
	cudaMalloc(&device_output, output_bytes);

	cudaMemcpy(
	    device_input,
	    input,
	    input_bytes,
	    cudaMemcpyHostToDevice
	);

	int threads_per_block = 256;

	int blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

	grayscale_kernel<<<blocks, threads_per_block>>>(
	    device_input,
	    device_output,
	    num_pixels
	);

	cudaDeviceSynchronize();

	cudaMemcpy(
	    output,
	    device_output,
	    output_bytes,
	    cudaMemcpyDeviceToHost
	);

	cudaFree(device_input);
	cudaFree(device_output);
}
