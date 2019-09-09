cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED)

cuda_add_executable(cuda-test extra/cuda-test.cu)

cuda_add_executable(cube Lesson1/cube.cu)
