#include <iostream>
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

// Vector addition kernel
__global__ void vectorAdd(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}

// Vector multiplication kernel
__global__ void vectorMult(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] * b[i];
    }
}

// Exponential kernel
__global__ void vectorExp(const float* a, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = expf(a[i]);
    }
}

// Natural logarithm kernel
__global__ void vectorLog(const float* a, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = logf(a[i]);
    }
}

// Sine kernel
__global__ void vectorSin(const float* a, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = sinf(a[i]);
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);

    // Host memory allocation
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_addResult = new float[n];
    float *h_multResult = new float[n];
    float *h_expResult = new float[n];
    float *h_logResult = new float[n];
    float *h_sinResult = new float[n];

    // Initialize input vectors with safe values for log
    for (int i = 0; i < n; ++i) {
        h_a[i] = 0.1f * (i + 1);  
        h_b[i] = 0.2f * (i + 1);
    }

    // Device memory allocation
    float *d_a, *d_b, *d_addResult, *d_multResult, *d_expResult, *d_logResult, *d_sinResult;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_addResult, size));
    CUDA_CHECK(cudaMalloc(&d_multResult, size));
    CUDA_CHECK(cudaMalloc(&d_expResult, size));
    CUDA_CHECK(cudaMalloc(&d_logResult, size));
    CUDA_CHECK(cudaMalloc(&d_sinResult, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Kernel configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernels
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_addResult, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vectorMult<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_multResult, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vectorExp<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_expResult, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vectorLog<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_logResult, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vectorSin<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_sinResult, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_addResult, d_addResult, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_multResult, d_multResult, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_expResult, d_expResult, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logResult, d_logResult, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sinResult, d_sinResult, size, cudaMemcpyDeviceToHost));

    // Verify and print first 5 elements of each result
    std::cout << "Vector Addition (first 5 elements):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_addResult[i] << " ";
    }

    std::cout << "\n\nVector Multiplication (first 5 elements):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_multResult[i] << " ";
    }

    std::cout << "\n\nExponential (first 5 elements):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_expResult[i] << " ";
    }

    std::cout << "\n\nNatural Logarithm (first 5 elements):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_logResult[i] << " ";
    }

    std::cout << "\n\nSine (first 5 elements):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_sinResult[i] << " ";
    }
    std::cout << "\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_addResult));
    CUDA_CHECK(cudaFree(d_multResult));
    CUDA_CHECK(cudaFree(d_expResult));
    CUDA_CHECK(cudaFree(d_logResult));
    CUDA_CHECK(cudaFree(d_sinResult));

    delete[] h_a;
    delete[] h_b;
    delete[] h_addResult;
    delete[] h_multResult;
    delete[] h_expResult;
    delete[] h_logResult;
    delete[] h_sinResult;

    return 0;
}
