#pragma once

void greyscale_cuda(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
);

void gaussian_blur_cuda(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
);

void sobel_cuda(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
);	

void launch_greyscale_kernel(
	const unsigned char* device_input,
	unsigned char* device_output,
	int width,
	int height
);

void launch_gaussian_blur_kernel(
	const unsigned char* device_input,
	unsigned char* device_output,
	int width,
	int height
);

void launch_sobel_kernel(
	const unsigned char* device_input,
	unsigned char* device_output,
	int width,
	int height
);

void run_cuda_pipeline(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	float* kernel_ms,
	double* end_to_end_ms
);

void launch_gaussian_blur_shared_kernel(
	const unsigned char* device_input,
	unsigned char* device_output,
	int width,
	int height
);

void launch_gaussian_blur_separable_kernel(
	const unsigned char* device_input,
	int* device_temp,
	unsigned char* device_output,
	int width,
	int height
);
