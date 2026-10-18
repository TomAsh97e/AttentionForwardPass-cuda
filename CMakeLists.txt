cmake_minimum_required(VERSION 3.28)

project(cuda_attention LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(cuda_attention
        src/main.cpp
        src/attention_naive.cu
)

target_include_directories(cuda_attention PRIVATE include)
target_link_libraries(cuda_attention PRIVATE CUDA::cudart)
