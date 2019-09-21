cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(CUDA_tuts)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
  # compared to class settings, we let NVidia's FindCUDA CMake detect
  # whether to build x64.  We tell it to support most devices, though,
  # to make sure more people can easily run class code without knowing
  # about this compiler argument
  set(CUDA_NVCC_FLAGS "
  -std=c++11
  #-ccbin /usr/bin/clang;
  #-gencode;arch=compute_20,code=sm_20;
  -gencode;arch=compute_30,code=sm_30;
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_35,code=compute_35;")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  cuda_add_executable(cuda-test extra/cuda-test.cu)

  ####################### LESSON 1 #############################################
  cuda_add_executable(cube            Lesson1/cube.cu)
  add_subdirectory(Lesson1/Problem_1)

  ####################### LESSON 2 #############################################
  cuda_add_executable(hello_cuda      Lesson2/hello_cuda.cu)
  cuda_add_executable(associative     Lesson2/associative.cu)
  cuda_add_executable(atomics         Lesson2/atomics.cu)
  cuda_add_executable(hello_blockIdx  Lesson2/hello_blockIdx.cu)
  cuda_add_executable(hello_threadIdx Lesson2/hello_threadIdx.cu)
  cuda_add_executable(memory          Lesson2/memory.cu)
  add_subdirectory(Lesson2/Problem_2)

  ####################### LESSON 3 #############################################
  cuda_add_executable(histo           Lesson3/histo.cu)
  cuda_add_executable(reduce          Lesson3/reduce.cu)
  add_subdirectory(Lesson3/Problem_3)

  ####################### LESSON 4 #############################################
  add_subdirectory(Lesson4/Problem_4)

  ####################### LESSON 5 #############################################
  cuda_add_executable(transpose       Lesson5/transpose.cu Lesson5/gputimer.h)
  cuda_add_executable(deviceQuery_simplified Lesson5/deviceQuery_simplified.cpp)

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
