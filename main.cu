#include <iostream>
#include <cooperative_groups.h>
#include <cuda/pipeline>

__global__ void asyncCopyKernel(int *d_out, const int *d_in, int n) {
    extern __shared__ int smem[];
    auto block = cooperative_groups::this_thread_block();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 파이프라인 상태 객체 생성
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // 2. 백그라운드 비동기 복사 지시
    pipeline.producer_acquire();
    // ★ 수정된 부분: block 전체가 협력해서 n개의 int를 한 번에 복사하도록 '시작 주소'와 '전체 크기'를 전달합니다.
    cuda::memcpy_async(block, smem, d_in, n * sizeof(int), pipeline);
    pipeline.producer_commit();

    // 3. 실제 데이터가 공유 메모리에 도착할 때까지 스레드 대기
    pipeline.consumer_wait();

    // 4. 연산 후 결과 저장 (안전하게 범위 체크)
    if (idx < n) {
        d_out[idx] = smem[threadIdx.x] * 2;
    }

    pipeline.consumer_release();
}

int main() {
    int n = 64;
    int size = n * sizeof(int);

    int *h_in = new int[n], *h_out = new int[n];
    for(int i=0; i<n; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    asyncCopyKernel<<<1, n, n * sizeof(int)>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "결과 확인 (첫 5개 값, 2배수 기대): ";
    for(int i=0; i<5; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}