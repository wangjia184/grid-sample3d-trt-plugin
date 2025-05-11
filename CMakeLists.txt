cmake_minimum_required(VERSION 3.10)

project(grid_sample_3d_plugin LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(TensorRT_ROOT)
    message(STATUS "TensorRT_ROOT: ${TensorRT_ROOT}")
    if(NOT TensorRT_INCLUDE_DIR)
        set(TensorRT_INCLUDE_DIR ${TensorRT_ROOT}/include)
    endif()
    if(NOT TensorRT_LIB_DIR)
        set(TensorRT_LIB_DIR ${TensorRT_ROOT}/lib)
    endif()
endif()

file(GLOB SOURCES "./src/*.cpp")
file(GLOB CU_SOURCE "./src/*.cu")

message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

link_directories(
    /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/
    /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/stubs/
    /usr/local/cuda-12.9/targets/x86_64-linux/lib/
    /usr/local/cuda-12.9/targets/x86_64-linux/lib/stubs/
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CU_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE 
    "./src" 
    /usr/local/tensorrt/include/
    /usr/local/cuda-12.9/targets/x86_64-linux/include/
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    nvinfer
    nvinfer_plugin
    nvonnxparser
    onnx_proto
    nvinfer_vc_plugin
    nvinfer_dispatch
    nvinfer_lean

    nvrtc
    nvrtc-builtins
    #nvrtc_static
    #nvrtc-builtins_static
    #nvptxcompiler
    nvJitLink
 
)



set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;80;86;89")

add_subdirectory(test)
enable_testing()