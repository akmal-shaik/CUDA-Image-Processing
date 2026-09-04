#include "cpu_filters.hpp"
#include <cmath>

void sobel_cpu(
	const unsigned char* input,
	unsigned char* output,
	int width,
	int height
)
{
	const int gx_kernel[9] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	const int gy_kernel[9] = {
		-1, -2, -1,
		 0,  0,  0,
		 1,  2,  1
	};

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
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

					gx += pixel * gx_kernel[kernel_index];
					gy += pixel * gy_kernel[kernel_index];
				}
			}

			int magnitude = static_cast<int>(std::sqrt(static_cast<float>(gx * gx + gy * gy)));

			if (magnitude > 255) magnitude = 255;

			int output_index = y * width + x;
			output[output_index] = static_cast<unsigned char>(magnitude);

		}
	}
}
