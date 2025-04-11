## TensorRT plugin for 3D GridSample Operator

At current stage, TensorRT(up to version 8.6.1) does not support 3D GridSample operator.

This plugin is a custom implementation of the 3D GridSample operator for TensorRT. It is inspired by the [GridSample operator](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) from PyTorch, and the code structure is inspired by project [onnxparser-trt-plugin-sample](https://github.com/TrojanXu/onnxparser-trt-plugin-sample).

### Build statically linked shared library

1. Let cmake find nvcc

Base image `nvcr.io/nvidia/pytorch:23.04-py3` https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-07.html

Go to https://developer.nvidia.com/nvidia-tensorrt-8x-download, download "TensorRT 8.6 GA for Linux x86_64 and CUDA 12.0 and 12.1 TAR Package"

Extract to disk,  start docker container to build

```
docker run -ti --name tensorrt --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/:/src \
    -v /mnt/f/TensorRT-8.6.1.6/:/usr/local/tensorrt/ \
    nvcr.io/nvidia/pytorch:23.04-py3
```

2. Build the plugin with the following commands:
```shell
mkdir -p build
cd build
cmake .. -DTensorRT_ROOT=/usr/local/tensorrt -DCMAKE_BUILD_TYPE=Release
make
cp -f libgrid_sample_3d_plugin.so ../../bin/
```
 
### Usage 

for python code (only on Linux platform), load the plugin with:

```python
import ctypes
success = ctypes.CDLL("build/libgrid_sample_3d_plugin.so", mode = ctypes.RTLD_GLOBAL)
```

see [test_grid_sample3d.py](./test/test_grid_sample3d_plugin.py) for more details.