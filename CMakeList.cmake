cmake_minimum_required(VERSION 3.18)
project(CudaTest LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # GPU에 맞게 수정 (RTX 30/40 → 86 or 89, A100 → 80 등)

add_executable(cuda_test
        main.cu          # 만약 main이 .cu라면
        kernel.cu        # ← 여기 추가! (당신이 만든 파일 이름)
)

set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)