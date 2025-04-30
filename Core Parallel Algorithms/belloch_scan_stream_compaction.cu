#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error " << cudaGetErrorString(err_) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

template <typename T>
__global__ void inclusive_scan_kernel(T* input, T* output, int n) {
    extern __shared__ T temp[];
    int tid = threadIdx.x;
    int offset = 1;

    temp[2*tid] = (2*tid < n) ? input[2*tid] : 0;
    temp[2*tid+1] = (2*tid+1 < n) ? input[2*tid+1] : 0;

    // Up-sweep phase
    for(int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if(tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Down-sweep phase
    if(tid == 0) temp[n-1] = 0;
    
    for(int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if(tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    if(2*tid < n) output[2*tid] = temp[2*tid];
    if(2*tid+1 < n) output[2*tid+1] = temp[2*tid+1];
}

template <typename T>
void inclusive_scan(T* d_input, T* d_output, int n) {
    const int threads = 256;
    const int elements_per_block = 2 * threads;
    int blocks = (n + elements_per_block - 1) / elements_per_block;
    
    inclusive_scan_kernel<T><<<blocks, threads, elements_per_block * sizeof(T)>>>(
        d_input, d_output, n
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void exclusive_scan(T* d_input, T* d_output, int n) {
    inclusive_scan(d_input, d_output, n);
    
    thrust::device_ptr<T> dev_ptr(d_output);
    thrust::transform(dev_ptr, dev_ptr + n, dev_ptr, 
        [=] __device__ (T x) { return x - *thrust::device_pointer_cast(d_input); });
}

template <typename T, typename Predicate>
__global__ void stream_compaction_kernel(T* input, T* output, int* flags, int* indices, int n, Predicate pred) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    
    flags[idx] = pred(input[idx]) ? 1 : 0;
    
    __syncthreads();
    
    if(flags[idx]) {
        output[indices[idx]] = input[idx];
    }
}

template <typename T, typename Predicate>
int stream_compaction(T* d_input, T* d_output, int n, Predicate pred) {
    int* d_flags, *d_indices;
    CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices, n * sizeof(int)));
    
    // 1. Create flags array
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    stream_compaction_kernel<T, Predicate><<<blocks, threads>>>(
        d_input, d_output, d_flags, d_indices, n, pred
    );
    
    // 2. Perform exclusive scan on flags
    exclusive_scan(d_flags, d_indices, n);
    
    // 3. Get total number of elements passing predicate
    int total;
    CUDA_CHECK(cudaMemcpy(&total, d_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int last_flag;
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    total += last_flag;
    
    // 4. Compact elements
    stream_compaction_kernel<T, Predicate><<<blocks, threads>>>(
        d_input, d_output, d_flags, d_indices, n, pred
    );
    
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_indices));
    
    return total;
}

int main() {
    const int n = 16;
    std::vector<int> h_input(n);
    for(int i = 0; i < n; ++i) {
        h_input[i] = i % 4;  // Sample data: [0,1,2,3,0,1,2,3,...]
    }

    // Allocate device memory
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // 1. Test inclusive scan
    inclusive_scan(d_input, d_output, n);
    std::vector<int> h_scan(n);
    CUDA_CHECK(cudaMemcpy(h_scan.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Inclusive Scan: ";
    for(int val : h_scan) std::cout << val << " ";
    std::cout << "\n";

    // 2. Test exclusive scan
    exclusive_scan(d_input, d_output, n);
    CUDA_CHECK(cudaMemcpy(h_scan.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Exclusive Scan: ";
    for(int val : h_scan) std::cout << val << " ";
    std::cout << "\n";

    // 3. Test stream compaction (filter non-zero elements)
    int* d_compacted;
    CUDA_CHECK(cudaMalloc(&d_compacted, n * sizeof(int)));
    auto is_non_zero = [] __device__ (int x) { return x != 0; };
    int compacted_size = stream_compaction<int>(d_input, d_compacted, n, is_non_zero);
    
    std::vector<int> h_compacted(compacted_size);
    CUDA_CHECK(cudaMemcpy(h_compacted.data(), d_compacted, compacted_size * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Compacted (" << compacted_size << " elements): ";
    for(int val : h_compacted) std::cout << val << " ";
    std::cout << "\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_compacted));

    return 0;
}
