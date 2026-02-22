# CUDA Optimization Examples
Simple vector addition with CUDA C++ using CLion + CMake.

## Requirements
- CUDA Toolkit 13.0+
- CMake 3.20+
- NVIDIA GPU with compute capability >= 8.6

## Build
```bash
mkdir build && cd build
cmake ..
make -j
./cuda_test