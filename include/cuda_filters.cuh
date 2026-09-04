#pragma once

void grayscale_cuda(
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
	
