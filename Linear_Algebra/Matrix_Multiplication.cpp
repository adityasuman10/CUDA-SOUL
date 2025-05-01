#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Naive matrix multiplication kernel
__global__ void naiveMatMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication kernel
__global__ void tiledMatMul(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;

    for (int ph = 0; ph < (N + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        int aCol = ph * TILE_SIZE + tx;
        int aRow = row;
        if (aRow < N && aCol < N)
            As[ty][tx] = A[aRow * N + aCol];
        else
            As[ty][tx] = 0.0f;

        int bRow = ph * TILE_SIZE + ty;
        int bCol = col;
        if (bRow < N && bCol < N)
            Bs[ty][tx] = B[bRow * N + bCol];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Matrix addition kernel
__global__ void matrixAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i * N + j] = A[i * N + j] + B[i * N + j];
}

// Matrix subtraction kernel
__global__ void matrixSub(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i * N + j] = A[i * N + j] - B[i * N + j];
}

// Strassen's algorithm host function
void strassen(float *d_A, float *d_B, float *d_C, int N) {
    if (N <= 64) {
        dim3 blocks((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
        dim3 threads(TILE_SIZE, TILE_SIZE);
        tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        return;
    }

    int newSize = N / 2;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((newSize + TILE_SIZE - 1)/TILE_SIZE, (newSize + TILE_SIZE - 1)/TILE_SIZE);

    // Submatrix pointers
    float *A11 = d_A, *A12 = d_A + newSize;
    float *A21 = d_A + newSize * N, *A22 = A21 + newSize;
    float *B11 = d_B, *B12 = d_B + newSize;
    float *B21 = d_B + newSize * N, *B22 = B21 + newSize;
    float *C11 = d_C, *C12 = d_C + newSize;
    float *C21 = d_C + newSize * N, *C22 = C21 + newSize;

    // Temporary matrices
    float *S1, *S2, *S3, *S4, *S5, *S6, *S7, *S8, *S9, *S10;
    float *P1, *P2, *P3, *P4, *P5, *P6, *P7;

    CHECK_CUDA(cudaMalloc(&S1, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S2, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S3, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S4, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S5, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S6, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S7, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S8, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S9, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&S10, newSize * newSize * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&P1, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&P2, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&P3, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&P4, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&P5, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&P6, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&P7, newSize * newSize * sizeof(float)));

    // Compute S matrices
    matrixSub<<<dimGrid, dimBlock>>>(B12, B22, S1, newSize);      // S1 = B12 - B22
    matrixAdd<<<dimGrid, dimBlock>>>(A11, A12, S2, newSize);      // S2 = A11 + A12
    matrixAdd<<<dimGrid, dimBlock>>>(A21, A22, S3, newSize);      // S3 = A21 + A22
    matrixSub<<<dimGrid, dimBlock>>>(B21, B11, S4, newSize);      // S4 = B21 - B11
    matrixAdd<<<dimGrid, dimBlock>>>(A11, A22, S5, newSize);      // S5 = A11 + A22
    matrixAdd<<<dimGrid, dimBlock>>>(B11, B22, S6, newSize);      // S6 = B11 + B22
    matrixSub<<<dimGrid, dimBlock>>>(A12, A22, S7, newSize);      // S7 = A12 - A22
    matrixAdd<<<dimGrid, dimBlock>>>(B21, B22, S8, newSize);      // S8 = B21 + B22
    matrixSub<<<dimGrid, dimBlock>>>(A11, A21, S9, newSize);      // S9 = A11 - A21
    matrixAdd<<<dimGrid, dimBlock>>>(B11, B12, S10, newSize);     // S10 = B11 + B12

    // Compute P matrices
    strassen(A11, S1, P1, newSize);      // P1 = A11 * S1
    strassen(S2, B22, P2, newSize);      // P2 = S2 * B22
    strassen(S3, B11, P3, newSize);      // P3 = S3 * B11
    strassen(A22, S4, P4, newSize);      // P4 = A22 * S4
    strassen(S5, S6, P5, newSize);       // P5 = S5 * S6
    strassen(S7, S8, P6, newSize);       // P6 = S7 * S8
    strassen(S9, S10, P7, newSize);      // P7 = S9 * S10

    // Compute C submatrices
    float *temp1, *temp2;
    CHECK_CUDA(cudaMalloc(&temp1, newSize * newSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temp2, newSize * newSize * sizeof(float)));
    
    // C11 = P5 + P4 - P2 + P6
    matrixAdd<<<dimGrid, dimBlock>>>(P5, P4, temp1, newSize);
    matrixSub<<<dimGrid, dimBlock>>>(temp1, P2, temp2, newSize);
    matrixAdd<<<dimGrid, dimBlock>>>(temp2, P6, C11, newSize);

    // C12 = P1 + P2
    matrixAdd<<<dimGrid, dimBlock>>>(P1, P2, C12, newSize);
    
    // C21 = P3 + P4
    matrixAdd<<<dimGrid, dimBlock>>>(P3, P4, C21, newSize);
    
    // C22 = P5 + P1 - P3 - P7
    matrixAdd<<<dimGrid, dimBlock>>>(P5, P1, temp1, newSize);
    matrixSub<<<dimGrid, dimBlock>>>(temp1, P3, temp2, newSize);
    matrixSub<<<dimGrid, dimBlock>>>(temp2, P7, C22, newSize);

    // Free temporary memory
    CHECK_CUDA(cudaFree(S1)); CHECK_CUDA(cudaFree(S2));
    CHECK_CUDA(cudaFree(S3)); CHECK_CUDA(cudaFree(S4));
    CHECK_CUDA(cudaFree(S5)); CHECK_CUDA(cudaFree(S6));
    CHECK_CUDA(cudaFree(S7)); CHECK_CUDA(cudaFree(S8));
    CHECK_CUDA(cudaFree(S9)); CHECK_CUDA(cudaFree(S10));
    CHECK_CUDA(cudaFree(P1)); CHECK_CUDA(cudaFree(P2));
    CHECK_CUDA(cudaFree(P3)); CHECK_CUDA(cudaFree(P4));
    CHECK_CUDA(cudaFree(P5)); CHECK_CUDA(cudaFree(P6));
    CHECK_CUDA(cudaFree(P7));
    CHECK_CUDA(cudaFree(temp1)); CHECK_CUDA(cudaFree(temp2));
}

int main() {
    const int N = 1024;  // Must be power of 2 for Strassen's
    float *h_A, *h_B, *h_C;
    
    // Allocate host memory
    h_A = new float[N*N];
    h_B = new float[N*N];
    h_C = new float[N*N];
    
    // Initialize matrices with sample values
    for (int i = 0; i < N*N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N*N*sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    // Naive multiplication
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15)/16, (N + 15)/16);
    naiveMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Naive multiplication completed." << std::endl;

    // Tiled multiplication
    tiledMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Tiled multiplication completed." << std::endl;

    // Strassen's algorithm
    strassen(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Strassen's multiplication completed." << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}