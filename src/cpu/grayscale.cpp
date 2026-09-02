#include "cpu_filters.hpp"

void grayscale_cpu(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    int num_pixels = width * height;

    for(int pixel = 0; pixel < num_pixels; pixel++)
    {
    	int rgb_index = pixel * 3;

    	const int r = input[rgb_index];
    	const int g = input[rgb_index + 1];
    	const int b = input[rgb_index + 2];

    	float grayscale = 0.299f * r + 0.587f * g + 0.114f * b;

    	output[pixel] = static_cast<unsigned char>(grayscale);
    }
}
