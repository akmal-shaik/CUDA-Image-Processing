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
	src/cpu/gaussian_blur.cpp
	src/cpu/sobel.cpp
	src/cuda/grayscale.cu
	src/cuda/gaussian_blur.cu
	src/cuda/sobel.cu
	src/cuda/pipeline.cu
)

target_include_directories(image_pipeline PRIVATE
	include
	third_party/stb
)

add_executable(benchmark_pipeline
	src/benchmark.cpp
	src/cpu/grayscale.cpp
	src/cpu/gaussian_blur.cpp
	src/cpu/sobel.cpp
	src/cuda/grayscale.cu
	src/cuda/gaussian_blur.cu
	src/cuda/sobel.cu
	src/cuda/pipeline.cu
)

target_include_directories(benchmark_pipeline PRIVATE
	include
)

add_executable(profile_pipeline
	src/profile.cu
	src/cuda/grayscale.cu
	src/cuda/gaussian_blur.cu
	src/cuda/sobel.cu
	src/cuda/pipeline.cu
	src/cuda/gaussian_blur_shared.cu
	src/cuda/gaussian_blur_separable.cu
)

target_include_directories(profile_pipeline PRIVATE
	include
)
