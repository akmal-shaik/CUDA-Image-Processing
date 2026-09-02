#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cpu_filters.hpp"

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

    // Make sure the image loaded successfully.
    if (image == nullptr)
    {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    std::cout << "Image loaded successfully\n";
    std::cout << "Width: " << width << '\n';
    std::cout << "Height: " << height << '\n';
    std::cout << "Original channels: " << channels << '\n';

    // Allocate one output byte for every pixel.
    std::vector<unsigned char> grayscale(width * height);

    // Run our sequential CPU grayscale implementation.
    grayscale_cpu(
        image,
        grayscale.data(),
        width,
        height
    );

    // Save the 1-channel grayscale image.
    int gray_success = stbi_write_png(
        "../images/output/grayscale_cpu.png",
        width,
        height,
        1,
        grayscale.data(),
        width
    );

    if (gray_success == 0)
    {
        std::cerr << "Failed to save grayscale image\n";
        stbi_image_free(image);
        return 1;
    }

    std::cout << "Grayscale image saved successfully\n";

    // Free the memory allocated by stb_image.
    stbi_image_free(image);

    return 0;
}
