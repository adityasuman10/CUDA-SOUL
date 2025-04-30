#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void bitonicSortBlock(int *arr, int n, int blockSize) {
    extern __shared__ int sArray[];
    int idx = blockIdx.x * blockSize + threadIdx.x;
    if (idx >= n) return;

    sArray[threadIdx.x] = arr[idx];
    __syncthreads();

    for (int stage = 0; (1 << stage) < blockSize; ++stage) {
        for (int step = stage; step >= 0; --step) {
            int j = 1 << step;
            int ixj = threadIdx.x ^ j;
            if (ixj > threadIdx.x) {
                if ((threadIdx.x & (1 << (stage + 1))) == 0) {
                    if (sArray[threadIdx.x] > sArray[ixj]) {
                        int temp = sArray[threadIdx.x];
                        sArray[threadIdx.x] = sArray[ixj];
                        sArray[ixj] = temp;
                    }
                } else {
                    if (sArray[threadIdx.x] < sArray[ixj]) {
                        int temp = sArray[threadIdx.x];
                        sArray[threadIdx.x] = sArray[ixj];
                        sArray[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    arr[idx] = sArray[threadIdx.x];
}

__global__ void mergeKernel(int *input, int *output, int n, int blockSize) {
    int segment = blockIdx.x;
    int start = segment * 2 * blockSize;
    int mid = min(start + blockSize, n);
    int end = min(start + 2 * blockSize, n);

    int aPtr = start, bPtr = mid;
    int outPtr = start;

    while (aPtr < mid && bPtr < end) {
        if (input[aPtr] < input[bPtr]) {
            output[outPtr++] = input[aPtr++];
        } else {
            output[outPtr++] = input[bPtr++];
        }
    }
    while (aPtr < mid) output[outPtr++] = input[aPtr++];
    while (bPtr < end) output[outPtr++] = input[bPtr++];
}

void mergeSort(int *hArray, int n) {
    int *dArray, *dTemp;
    cudaMalloc(&dArray, n * sizeof(int));
    cudaMalloc(&dTemp, n * sizeof(int));
    cudaMemcpy(dArray, hArray, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    dim3 blocks((n + blockSize - 1) / blockSize);
    bitonicSortBlock<<<blocks, blockSize, blockSize * sizeof(int)>>>(dArray, n, blockSize);

    while (blockSize < n) {
        dim3 mergeBlocks((n + (2 * blockSize) - 1) / (2 * blockSize));
        mergeKernel<<<mergeBlocks, 1>>>(dArray, dTemp, n, blockSize);
        std::swap(dArray, dTemp);
        blockSize *= 2;
    }

    cudaMemcpy(hArray, dArray, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dArray); cudaFree(dTemp);
}

int main() {
    int n = 1 << 20;
    int *hArray = new int[n];
    // Initialize array...
    mergeSort(hArray, n);
    // Verify sorted array...
    delete[] hArray;
    return 0;
}