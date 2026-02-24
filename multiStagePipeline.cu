#include <iostream>
#include <cooperative_groups.h>
#include <cuda/pipeline>

__global__ void multiStagePipelineKernel(int *d_out, const int *d_in, int total_batches, int batch_size) {
    // 스테이지가 2개이므로 공유 메모리도 2배(2 * batch_size) 공간을 사용합니다.
    extern __shared__ int smem[]; 
    auto block = cooperative_groups::this_thread_block();
    int tid = threadIdx.x;

    // 1. 2단계 파이프라인 상태 객체 생성
    constexpr int stages = 2;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stages> shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // [프롤로그] 0번 배치를 먼저 백그라운드로 가져오기 시작!
    pipeline.producer_acquire();
    cuda::memcpy_async(block, &smem, &d_in, batch_size * sizeof(int), pipeline);
    pipeline.producer_commit();

    // [메인 루프] 다음 배치를 가져오는 동시에(Copy), 이전 배치를 계산(Compute)
    for (int batch = 1; batch < total_batches; ++batch) {
        int compute_stage = (batch - 1) % stages; // 지금 계산할 구역 (0 또는 1)
        int copy_stage = batch % stages;          // 새 데이터를 받을 구역 (1 또는 0)
        
        // 2. [가져오기] 다음 배치 데이터를 백그라운드 복사 지시
        pipeline.producer_acquire();
        cuda::memcpy_async(block, &smem[copy_stage * batch_size], &d_in[batch * batch_size], batch_size * sizeof(int), pipeline);
        pipeline.producer_commit();

        // 3. [기다리기] 이전 배치 데이터가 안전하게 도착했는지 대기
        pipeline.consumer_wait();
        
        // 4. [계산하기] 이전 배치 데이터 연산
        d_out[(batch - 1) * batch_size + tid] = smem[compute_stage * batch_size + tid] * 2;
        pipeline.consumer_release(); // 계산이 끝난 구역을 해제하여 다음 복사에 쓸 수 있게 함
    }

    // [에필로그] 마지막 남은 배치 계산
    pipeline.consumer_wait();
    int last_compute_stage = (total_batches - 1) % stages;
    d_out[(total_batches - 1) * batch_size + tid] = smem[last_compute_stage * batch_size + tid] * 2;
    pipeline.consumer_release();
}

int main() {
    int batch_size = 64;
    int total_batches = 4;
    int n = batch_size * total_batches; // 총 256개 데이터
    int size = n * sizeof(int);

    int *h_in = new int[n], *h_out = new int[n];
    for(int i=0; i<n; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // ★ 공유 메모리 크기를 '스테이지 수(2) * 배치 크기'로 넉넉히 잡아줍니다.
    int shared_mem_size = 2 * batch_size * sizeof(int);
    multiStagePipelineKernel<<<1, batch_size, shared_mem_size>>>(d_out, d_in, total_batches, batch_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 결과 확인 (두 번째 배치인 인덱스 64번부터 5개 값 확인. 128, 130, 132... 기대)
    std::cout << "결과 확인 (두 번째 배치 첫 5개 값): ";
    for(int i=64; i<69; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}