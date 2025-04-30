#include <iostream>
#include <cuda_runtime.h>
#include <thrust/scan.h>

const int BITS = 4;
const int BINS = 1 << BITS;

__global__ void computeHistogram(int *input, int *histogram, int n, int bit) {
    extern __shared__ int sHist[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (threadIdx.x < BINS) sHist[threadIdx.x] = 0;
    __syncthreads();

    if (idx < n) {
        int digit = (input[idx] >> bit) & (BINS - 1);
        atomicAdd(&sHist[digit], 1);
    }
    __syncthreads();

    // Update global histogram
    if (threadIdx.x < BINS) {
        atomicAdd(&histogram[threadIdx.x], sHist[threadIdx.x]);
    }
}

__global__ void scatterKernel(int *input, int *output, int *prefixSum, int n, int bit) {
    extern __shared__ int shared[];
    int *sHist = shared;
    int *sPrefix = shared + BINS;
    int *sCounter = shared + 2 * BINS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (threadIdx.x < BINS) {
        sHist[threadIdx.x] = 0;
        sCounter[threadIdx.x] = 0;
    }
    __syncthreads();

    // Compute local histogram
    if (idx < n) {
        int digit = (input[idx] >> bit) & (BINS - 1);
        atomicAdd(&sHist[digit], 1);
    }
    __syncthreads();

    // Compute local prefix sum
    if (threadIdx.x == 0) {
        sPrefix[0] = 0;
        for (int i = 1; i < BINS; ++i)
            sPrefix[i] = sPrefix[i-1] + sHist[i-1];
    }
    __syncthreads();

    // Scatter elements
    if (idx < n) {
        int digit = (input[idx] >> bit) & (BINS - 1);
        int offset = atomicAdd(&sCounter[digit], 1);
        output[prefixSum[digit] + sPrefix[digit] + offset] = input[idx];
    }
}

void radixSort(int *hArray, int n) {
    int *dInput, *dTemp, *dHist, *dPrefix;
    cudaMalloc(&dInput, n * sizeof(int));
    cudaMalloc(&dTemp, n * sizeof(int));
    cudaMalloc(&dHist, BINS * sizeof(int));
    cudaMalloc(&dPrefix, BINS * sizeof(int));

    cudaMemcpy(dInput, hArray, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int bit = 0; bit < 32; bit += BITS) {
        cudaMemset(dHist, 0, BINS * sizeof(int));
        computeHistogram<<<blocks, threads, BINS * sizeof(int)>>>(dInput, dHist, n, bit);
        thrust::exclusive_scan(thrust::device, dHist, dHist + BINS, dPrefix);
        scatterKernel<<<blocks, threads, 3 * BINS * sizeof(int)>>>(dInput, dTemp, dPrefix, n, bit);
        std::swap(dInput, dTemp);
    }

    cudaMemcpy(hArray, dInput, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dInput); cudaFree(dTemp); cudaFree(dHist); cudaFree(dPrefix);
}

int main() {
    int n = 1 << 20;
    int *hArray = new int[n];
    // Initialize array...
    radixSort(hArray, n);
    // Verify sorted array...
    delete[] hArray;
    return 0;
}