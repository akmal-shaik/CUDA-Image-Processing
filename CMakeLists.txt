cmake_minimum_required(VERSION 3.18)

project(cuda_image_processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_check
    src/cuda-check.cu
)

add_executable(image_pipeline
    src/main.cpp
    src/cpu/grayscale.cpp
    src/cuda/grayscale.cu
)

target_include_directories(image_pipeline PRIVATE
    include
    third_party/stb
)
