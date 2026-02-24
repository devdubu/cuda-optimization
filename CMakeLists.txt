cmake_minimum_required(VERSION 3.20)
project(CudaTest LANGUAGES CXX CUDA)

# NVCC 강제 지정 (이미 잘 잡혔으니 유지)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 서버 GPU 아키텍처 (로그에서 sm_52 fallback 피하려면 필수!)
# nvidia-smi로 GPU 모델 확인 후 맞춰주세요 (RTX 30/40 시리즈 → 86, 4090 → 89, A100 → 80)
set(CMAKE_CUDA_ARCHITECTURES 86)

# ★ 핵심: OBJECT 제거하고 executable 하나만!
add_executable(cuda_test main.cu)
add_executable(cuda_test_pipeline multiStagePipeline.cu)

# separable compilation 활성화 (커널 분리 컴파일 추천)
set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_test_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# 필요 시 추가 라이브러리 (지금은 없음)
# target_link_libraries(cuda_test PRIVATE cuda cudart)