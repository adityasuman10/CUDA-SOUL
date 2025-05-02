#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Naive transpose kernel (non-coalesced writes)
__global__ void naiveTranspose(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Optimized transpose using shared memory
__global__ void optimizedTranspose(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Read input matrix into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Write transposed data to global memory
    int x_transposed = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_transposed = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x_transposed < height && y_transposed < width) {
        output[y_transposed * height + x_transposed] = tile[threadIdx.x][threadIdx.y];
    }
}

// Verify transpose correctness
bool verifyTranspose(float *original, float *transposed, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (original[y * width + x] != transposed[x * height + y]) {
                std::cerr << "Mismatch at (" << y << "," << x << ")" << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const size_t size = width * height * sizeof(float);
    
    // Allocate host memory
    float *h_input = new float[width * height];
    float *h_naive = new float[width * height];
    float *h_optimized = new float[width * height];
    
    // Initialize input matrix
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = y * width + x; // Unique values
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    
    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, 
                 (height + TILE_DIM - 1) / TILE_DIM);
    
    // Run naive transpose
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    naiveTranspose<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Naive transpose time: " << ms << " ms" << std::endl;
    
    // Copy naive result back
    CHECK_CUDA(cudaMemcpy(h_naive, d_output, size, cudaMemcpyDeviceToHost));
    
    // Run optimized transpose
    CHECK_CUDA(cudaEventRecord(start));
    optimizedTranspose<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Optimized transpose time: " << ms << " ms" << std::endl;
    
    // Copy optimized result back
    CHECK_CUDA(cudaMemcpy(h_optimized, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    std::cout << "Verifying naive transpose... "
              << (verifyTranspose(h_input, h_naive, width, height) ? "OK" : "FAIL")
              << std::endl;
    
    std::cout << "Verifying optimized transpose... "
              << (verifyTranspose(h_input, h_optimized, width, height) ? "OK" : "FAIL")
              << std::endl;
    
    // Cleanup
    delete[] h_input;
    delete[] h_naive;
    delete[] h_optimized;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
