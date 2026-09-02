#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main()
{
    int width;
    int height;
    int channels;

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

    std::cout << "\nFirst 5 pixels:\n";

    for (int pixel = 0; pixel < 5; pixel++)
    {
    	int index = pixel * 3;
    	
    	int r =image[index];
    	int g = image[index + 1];
    	int b = image[index + 2];

    	std::cout <<"Pixel" << pixel
    	          <<": R=" << r
                  <<" G=" << g
                  <<" B=" << b
                  << '\n';
    }

    int success = stbi_write_png(
    	"../images/output/roundtrip.png",
        width,
        height,
        3,
        image,
        width * 3
    );

    if (success == 0)
    {
    	std::cerr << "Failed to save image\n";
    	stbi_image_free(image);
    	return 1;
    }

    std::cout << "\nImage saved successfully\n";

    stbi_image_free(image);

    return 0;
}

