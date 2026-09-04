#include <iostream>
#include <vector>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cpu_filters.hpp"
#include "cuda_filters.cuh"

int main()
{
	int width;
	int height;
	int channels;

	unsigned char* image = stbi_load("../images/input/test.jpg", &width, &height, &channels, 3);

	if (image == nullptr)
	{
		std::cerr << "Failed to load image\n";
		return 1;
	}

	std::cout << "Image loaded successfully\n";
	std::cout << "Width: " << width << '\n';
	std::cout << "Height: " << height << '\n';
	std::cout << "Original channels: " << channels << '\n';

	// Output buffers
	std::vector<unsigned char> greyscale_cpu_output(width * height);
	std::vector<unsigned char> greyscale_cuda_output(width * height);

	std::vector<unsigned char> blur_cpu_output(width * height);
	std::vector<unsigned char> blur_cuda_output(width * height);

	std::vector<unsigned char> sobel_cpu_output(width * height);
	std::vector<unsigned char> sobel_cuda_output(width * height);

	// Extra buffers used only for isolated validation
	std::vector<unsigned char> blur_cuda_isolated(width * height);
	std::vector<unsigned char> sobel_cuda_isolated(width * height);

	// Complete CPU and CUDA pipelines
	grayscale_cpu(image, greyscale_cpu_output.data(), width, height);
	grayscale_cuda(image, greyscale_cuda_output.data(), width, height);

	gaussian_blur_cpu(greyscale_cpu_output.data(), blur_cpu_output.data(), width, height);
	gaussian_blur_cuda(greyscale_cuda_output.data(), blur_cuda_output.data(), width, height);

	sobel_cpu(blur_cpu_output.data(), sobel_cpu_output.data(), width, height);
	sobel_cuda(blur_cuda_output.data(), sobel_cuda_output.data(), width, height);

	// Isolated CUDA tests using the same CPU-generated input
	gaussian_blur_cuda(greyscale_cpu_output.data(), blur_cuda_isolated.data(), width, height);
	sobel_cuda(blur_cpu_output.data(), sobel_cuda_isolated.data(), width, height);

	// Greyscale validation
	int greyscale_mismatches = 0;
	int greyscale_max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(greyscale_cpu_output[pixel]) - static_cast<int>(greyscale_cuda_output[pixel]));

		if (difference > 1)
		{
			greyscale_mismatches++;
		}

		if (difference > greyscale_max_difference)
		{
			greyscale_max_difference = difference;
		}
	}

	// Gaussian blur isolated validation
	int blur_mismatches = 0;
	int blur_max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(blur_cpu_output[pixel]) - static_cast<int>(blur_cuda_isolated[pixel]));

		if (difference > 1)
		{
			blur_mismatches++;
		}

		if (difference > blur_max_difference)
		{
			blur_max_difference = difference;
		}
	}

	// Sobel isolated validation
	int sobel_mismatches = 0;
	int sobel_max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(sobel_cpu_output[pixel]) - static_cast<int>(sobel_cuda_isolated[pixel]));

		if (difference > 1)
		{
			sobel_mismatches++;
		}

		if (difference > sobel_max_difference)
		{
			sobel_max_difference = difference;
		}
	}

	// Complete CPU vs CUDA pipeline validation
	int pipeline_mismatches = 0;
	int pipeline_max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(sobel_cpu_output[pixel]) - static_cast<int>(sobel_cuda_output[pixel]));

		if (difference > 1)
		{
			pipeline_mismatches++;
		}

		if (difference > pipeline_max_difference)
		{
			pipeline_max_difference = difference;
		}
	}

	// Print validation results
	std::cout << "\n=== Isolated Stage Validation ===\n";

	std::cout << "Greyscale mismatches (>1): " << greyscale_mismatches << '\n';
	std::cout << "Greyscale max difference: " << greyscale_max_difference << '\n';

	std::cout << "Gaussian blur mismatches (>1): " << blur_mismatches << '\n';
	std::cout << "Gaussian blur max difference: " << blur_max_difference << '\n';

	std::cout << "Sobel mismatches (>1): " << sobel_mismatches << '\n';
	std::cout << "Sobel max difference: " << sobel_max_difference << '\n';

	std::cout << "\n=== End-to-End Pipeline Validation ===\n";

	std::cout << "Pipeline mismatches (>1): " << pipeline_mismatches << '\n';
	std::cout << "Pipeline max difference: " << pipeline_max_difference << '\n';

	// Save CPU greyscale image
	int cpu_success = stbi_write_png("../images/output/grayscale_cpu.png", width, height, 1, greyscale_cpu_output.data(), width);

	if (cpu_success == 0)
	{
		std::cerr << "Failed to save CPU greyscale image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "\nCPU greyscale image saved successfully\n";

	// Save CUDA greyscale image
	int cuda_success = stbi_write_png("../images/output/grayscale_cuda.png", width, height, 1, greyscale_cuda_output.data(), width);

	if (cuda_success == 0)
	{
		std::cerr << "Failed to save CUDA greyscale image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CUDA greyscale image saved successfully\n";

	// Save CPU Gaussian blur image
	int blur_success = stbi_write_png("../images/output/blur_cpu.png", width, height, 1, blur_cpu_output.data(), width);

	if (blur_success == 0)
	{
		std::cerr << "Failed to save CPU Gaussian blur image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CPU Gaussian blur image saved successfully\n";

	// Save CUDA Gaussian blur image
	int blur_cuda_success = stbi_write_png("../images/output/blur_cuda.png", width, height, 1, blur_cuda_output.data(), width);

	if (blur_cuda_success == 0)
	{
		std::cerr << "Failed to save CUDA Gaussian blur image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CUDA Gaussian blur image saved successfully\n";

	// Save CPU Sobel image
	int sobel_cpu_success = stbi_write_png("../images/output/sobel_cpu.png", width, height, 1, sobel_cpu_output.data(), width);

	if (sobel_cpu_success == 0)
	{
		std::cerr << "Failed to save CPU Sobel image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CPU Sobel image saved successfully\n";

	// Save CUDA Sobel image
	int sobel_cuda_success = stbi_write_png("../images/output/sobel_cuda.png", width, height, 1, sobel_cuda_output.data(), width);

	if (sobel_cuda_success == 0)
	{
		std::cerr << "Failed to save CUDA Sobel image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CUDA Sobel image saved successfully\n";

	stbi_image_free(image);

	return 0;
}
