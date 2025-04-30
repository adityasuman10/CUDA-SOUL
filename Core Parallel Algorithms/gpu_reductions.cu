#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

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
__device__ T warpReduce(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        if (offset == 16) {  // Handle non-32-bit types properly
            val += tmp;
        } else {
            val += tmp;
        }
    }
    return val;
}

template <typename T, typename Op>
__global__ void reduceKernel(const T* arr, T* result, int n, Op op, T init) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    T val = init;
    if (i < n) val = arr[i];
    if (i + blockDim.x < n) val = op(val, arr[i + blockDim.x]);
    
    for (int offset = blockDim.x/2; offset >= warpSize; offset >>= 1) {
        if (tid < offset && i + offset < n) {
            val = op(val, arr[i + offset]);
        }
    }
    
    val = warpReduce(val);
    
    if (tid == 0) {
        sdata[0] = val;
    }
    __syncthreads();
    
    if (blockIdx.x == 0) {
        T blockResult = (tid < blockDim.x) ? sdata[tid] : init;
        T finalVal = warpReduce(blockResult);
        
        if (tid == 0) {
            result[0] = finalVal;
        }
    }
}

template <typename T>
T reduce(const T* d_arr, int n, const char* op) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shmem_size = threads * sizeof(T);
    
    T* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(T)));
    
    // First reduction step
    if (strcmp(op, "sum") == 0) {
        reduceKernel<<<blocks, threads, shmem_size>>>(
            d_arr, d_partial, n, [] __device__ (T a, T b) { return a + b; }, T(0)
        );
    } else if (strcmp(op, "max") == 0) {
        reduceKernel<<<blocks, threads, shmem_size>>>(
            d_arr, d_partial, n, [] __device__ (T a, T b) { return max(a, b); }, std::numeric_limits<T>::lowest()
        );
    } else if (strcmp(op, "min") == 0) {
        reduceKernel<<<blocks, threads, shmem_size>>>(
            d_arr, d_partial, n, [] __device__ (T a, T b) { return min(a, b); }, std::numeric_limits<T>::max()
        );
    }
    
    // Final reduction
    T final_result;
    if (blocks > 1) {
        final_result = reduce(d_partial, blocks, op);
    } else {
        CUDA_CHECK(cudaMemcpy(&final_result, d_partial, sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_partial));
    return final_result;
}

__global__ void dotProductKernel(const float* a, const float* b, float* temp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        temp[i] = a[i] * b[i];
    }
}

float dotProduct(const float* d_a, const float* d_b, int n) {
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(float)));
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dotProductKernel<<<blocks, threads>>>(d_a, d_b, d_temp, n);
    
    float result = reduce<float>(d_temp, n, "sum");
    CUDA_CHECK(cudaFree(d_temp));
    return result;
}

int main() {
    const int n = 1 << 20;  // 1M elements
    std::vector<float> h_a(n), h_b(n);
    
    // Initialize with sample data
    for (int i = 0; i < n; ++i) {
        h_a[i] = (i % 100) * 0.1f;
        h_b[i] = (i % 50) * 0.2f;
    }
    
    // Allocate device memory
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute reductions
    float total = reduce<float>(d_a, n, "sum");
    float max_val = reduce<float>(d_a, n, "max");
    float min_val = reduce<float>(d_a, n, "min");
    float dot = dotProduct(d_a, d_b, n);
    
    std::cout << "Array Statistics:\n"
              << "Sum: " << total << "\n"
              << "Max: " << max_val << "\n"
              << "Min: " << min_val << "\n"
              << "Dot Product: " << dot << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    
    return 0;
}
