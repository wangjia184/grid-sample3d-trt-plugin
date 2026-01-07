cmake_minimum_required(VERSION 3.10)

set(PARENT_PROJECT_NAME ${PROJECT_NAME})

project(test LANGUAGES CXX CUDA)
enable_language(CUDA)
find_package(CUDAToolkit)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(TEST_GRID_SAMPLE test_grid_sample)

add_executable(${TEST_GRID_SAMPLE} test.cpp)

target_include_directories(${TEST_GRID_SAMPLE} PUBLIC 
    ${PROJECT_INCLUDE_DIR} 
    ${CUDA_ROOT}/include
    /usr/local/cuda-13.1/targets/x86_64-linux/include/
)

link_directories(
    /usr/local/cuda-13.1/targets/x86_64-linux/lib/
    /usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs/
    /usr/local/tensorrt/lib/
    /usr/local/tensorrt/lib/stubs/
)


target_link_libraries(${TEST_GRID_SAMPLE} PRIVATE
    ${PARENT_PROJECT_NAME} 
    nvJitLink_static 
    cudart_static
    
)

set_target_properties(${TEST_GRID_SAMPLE} PROPERTIES CUDA_ARCHITECTURES "80;86;89;90;100")