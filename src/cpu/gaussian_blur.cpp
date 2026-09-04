#include "cpu_filters.hpp"

void gaussian_blur_cpu(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    const int kernel[25] = {
         1,  4,  6,  4, 1,
         4, 16, 24, 16, 4,
         6, 24, 36, 24, 6,
         4, 16, 24, 16, 4,
         1,  4,  6,  4, 1
    };

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
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

                    sum += input[input_index] * kernel[kernel_index];
                }
            }

            int output_index = y * width + x;
            output[output_index] = static_cast<unsigned char>(sum / 256);
        }
    }
}
