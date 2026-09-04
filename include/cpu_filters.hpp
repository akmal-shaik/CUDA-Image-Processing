#pragma once

void grayscale_cpu(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
);

void gaussian_blur_cpu(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
);
