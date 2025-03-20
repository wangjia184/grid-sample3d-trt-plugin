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
    /usr/local/tensorrt/lib/
    /usr/local/tensorrt/lib/stubs/
    /usr/local/cuda-12.1/targets/x86_64-linux/lib/
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CU_SOURCE})

target_include_directories(${PROJECT_NAME} PRIVATE "./src" ${CUDAToolkit_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer)



set(COMMON_LIBS
    cufft
    curand
    cusparse
    dl
    rt
    pthread
)

 
target_link_libraries(${PROJECT_NAME} PRIVATE
    # TensorRT Static
    nvinfer_static
    nvinfer_plugin_static
    nvonnxparser_static
    nvparsers_static
    nvinfer_lean_static
    nvinfer_vc_plugin_static
    
    # CUDA Related
    cublas_static_stub_trt
    cublasLt_static_stub_trt
    cudnn_static_stub_trt
    cudart_static
    
    # Others
    #nvrtc_static
    #nvptxcompiler_static
    #onnx_proto
    #culibos 
    
    -static-libgcc
    -static-libstdc++
    
    #${COMMON_LIBS}
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;80;86;89")

add_subdirectory(test)
enable_testing()