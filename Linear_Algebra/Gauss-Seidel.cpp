#include <iostream>
#include <cuda_runtime.h>

__global__ void gaussSeidelKernel(const double* A, const double* b, double* x, int n) {
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j != i) sum += A[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i * n + i];
    }
}

void gaussSeidelCUDA(const double* A, const double* b, double* x, int n,
                    double tolerance = 1e-6, int max_iterations = 1000) {
    // Allocate device memory
    double *d_A, *d_b, *d_x;
    cudaMalloc(&d_A, n*n*sizeof(double));
    cudaMalloc(&d_b, n*sizeof(double));
    cudaMalloc(&d_x, n*sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, n*sizeof(double));

    // Run iterations
    for (int iter = 0; iter < max_iterations; ++iter) {
        gaussSeidelKernel<<<1, 1>>>(d_A, d_b, d_x, n);
        cudaDeviceSynchronize();
    }

    // Copy result back
    cudaMemcpy(x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
}

int main() {
    const int n = 3;
    double A[n*n] = {4, -1, 0,
                    -1, 4, -1,
                     0, -1, 4};
    double b[n] = {15, 10, 15};
    double x[n];

    gaussSeidelCUDA(A, b, x, n);

    std::cout << "CUDA Solution:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";

    return 0;
}