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

	std::vector<unsigned char> grayscale_cpu_output(width * height);
	std::vector<unsigned char> grayscale_cuda_output(width * height);
	std::vector<unsigned char> blur_cpu_output(width * height);
	std::vector<unsigned char> blur_cuda_output(width * height);
	std::vector<unsigned char> sobel_cpu_output(width * height);
	std::vector<unsigned char> sobel_cuda_output(width * height);

	grayscale_cpu(image, grayscale_cpu_output.data(), width, height);

	grayscale_cuda(image, grayscale_cuda_output.data(), width, height);

	gaussian_blur_cpu(grayscale_cpu_output.data(), blur_cpu_output.data(), width, height);

	gaussian_blur_cuda(grayscale_cuda_output.data(), blur_cuda_output.data(), width, height);

	sobel_cpu(blur_cpu_output.data(), sobel_cpu_output.data(), width, height);

	sobel_cuda(blur_cuda_output.data(), sobel_cuda_output.data(), width, height);

	int mismatches = 0;
	int max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(grayscale_cpu_output[pixel]) - static_cast<int>(grayscale_cuda_output[pixel]));

		if (difference > 1)
		{
			mismatches++;
		}

		if (difference > max_difference)
		{
			max_difference = difference;
		}
	}

	std::cout << "CPU/GPU grayscale mismatches: " << mismatches << '\n';
	std::cout << "Maximum grayscale pixel difference: " << max_difference << '\n';

	int cpu_success = stbi_write_png("../images/output/grayscale_cpu.png", width, height, 1, grayscale_cpu_output.data(), width);

	if (cpu_success == 0)
	{
		std::cerr << "Failed to save CPU grayscale image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CPU grayscale image saved successfully\n";

	int cuda_success = stbi_write_png("../images/output/grayscale_cuda.png", width, height, 1, grayscale_cuda_output.data(), width);

	if (cuda_success == 0)
	{
		std::cerr << "Failed to save CUDA grayscale image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CUDA grayscale image saved successfully\n";


	int blur_mismatches = 0;
	int blur_max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(blur_cpu_output[pixel]) - static_cast<int>(blur_cuda_output[pixel]));

		if (difference > 1)
		{
			blur_mismatches++;
		}

		if (difference > blur_max_difference)
		{
			blur_max_difference = difference;
		}
	}

	std::cout << "CPU/GPU blur mismatches: " << blur_mismatches << '\n';
	std::cout << "Maximum blur pixel difference: " << blur_max_difference << '\n';

	int blur_success = stbi_write_png("../images/output/blur_cpu.png", width, height, 1, blur_cpu_output.data(), width);

	if (blur_success == 0)
	{
		std::cerr << "Failed to save CPU Gaussian blur image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CPU Gaussian blur image saved successfully\n";

	int blur_cuda_success = stbi_write_png("../images/output/blur_cuda.png", width, height, 1, blur_cuda_output.data(), width);

	if (blur_cuda_success == 0)
	{
		std::cerr << "Failed to save CUDA Gaussian blur image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CUDA Gaussian blur image saved successfully\n";

	int sobel_mismatches = 0;
	int sobel_max_difference = 0;

	for (int pixel = 0; pixel < width * height; pixel++)
	{
		int difference = std::abs(static_cast<int>(sobel_cpu_output[pixel]) - static_cast<int>(sobel_cuda_output[pixel]));

		if (difference > 1)
		{
			sobel_mismatches++;
		}

		if (difference > sobel_max_difference)
		{
			sobel_max_difference = difference;
		}
	}

	std::cout << "CPU/GPU Sobel mismatches: " << sobel_mismatches << '\n';
	std::cout << "Maximum Sobel pixel difference: " << sobel_max_difference << '\n';

	int sobel_cpu_success = stbi_write_png("../images/output/sobel_cpu.png", width, height, 1, sobel_cpu_output.data(), width);

	if (sobel_cpu_success == 0)
	{
		std::cerr << "Failed to save CPU Sobel image\n";
		stbi_image_free(image);
		return 1;
	}

	std::cout << "CPU Sobel image saved successfully\n";

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
