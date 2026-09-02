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

    // Load the input image and force it into 3-channel RGB format.
    unsigned char* image = stbi_load(
        "../images/input/test.jpg",
        &width,
        &height,
        &channels,
        3
    );

    if (image == nullptr)
    {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    std::cout << "Image loaded successfully\n";
    std::cout << "Width: " << width << '\n';
    std::cout << "Height: " << height << '\n';
    std::cout << "Original channels: " << channels << '\n';

    // Create separate output buffers for CPU and CUDA results.
    std::vector<unsigned char> grayscale_cpu_output(width * height);
    std::vector<unsigned char> grayscale_cuda_output(width * height);

    // Run sequential CPU grayscale.
    grayscale_cpu(
        image,
        grayscale_cpu_output.data(),
        width,
        height
    );

    // Run CUDA grayscale.
    grayscale_cuda(
        image,
        grayscale_cuda_output.data(),
        width,
        height
    );

    // Compare CPU and GPU outputs.
    int mismatches = 0;
    int max_difference = 0;

    for (int pixel = 0; pixel < width * height; pixel++)
    {
        int difference = std::abs(
            static_cast<int>(grayscale_cpu_output[pixel]) -
            static_cast<int>(grayscale_cuda_output[pixel])
        );

        if (difference > 1)
        {
            mismatches++;
        }

        if (difference > max_difference)
        {
            max_difference = difference;
        }
    }

    std::cout << "CPU/GPU mismatches: "
              << mismatches << '\n';

    std::cout << "Maximum pixel difference: "
              << max_difference << '\n';

    // Save CPU grayscale image.
    int cpu_success = stbi_write_png(
        "../images/output/grayscale_cpu.png",
        width,
        height,
        1,
        grayscale_cpu_output.data(),
        width
    );

    if (cpu_success == 0)
    {
        std::cerr << "Failed to save CPU grayscale image\n";
        stbi_image_free(image);
        return 1;
    }

    std::cout << "CPU grayscale image saved successfully\n";

    // Save CUDA grayscale image.
    int cuda_success = stbi_write_png(
        "../images/output/grayscale_cuda.png",
        width,
        height,
        1,
        grayscale_cuda_output.data(),
        width
    );

    if (cuda_success == 0)
    {
        std::cerr << "Failed to save CUDA grayscale image\n";
        stbi_image_free(image);
        return 1;
    }

    std::cout << "CUDA grayscale image saved successfully\n";

    // Free memory allocated by stb_image.
    stbi_image_free(image);

    return 0;
}
