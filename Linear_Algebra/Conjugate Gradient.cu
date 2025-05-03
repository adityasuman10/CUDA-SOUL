#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixVectorMul(double *A, double *p, double *Ap, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        double sum = 0.0;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col] * p[col];
        }
        Ap[row] = sum;
    }
}

__global__ void dotProduct(double *a, double *b, double *result, int n) {
    extern __shared__ double cache[];
    int tid = threadIdx.x;
    cache[tid] = 0.0;
    if (tid < n) cache[tid] = a[tid] * b[tid];
    __syncthreads();

    // Reduction
    for (int i = blockDim.x/2; i > 0; i /= 2) {
        if (tid < i && tid + i < n) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, cache[0]);
}

void conjugateGradientCUDA(double *A, double *b, double *x, int n, double tol=1e-6) {
    // Allocate device memory
    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;
    cudaMalloc(&d_A, n*n*sizeof(double));
    cudaMalloc(&d_b, n*sizeof(double));
    cudaMalloc(&d_x, n*sizeof(double));
    cudaMalloc(&d_r, n*sizeof(double));
    cudaMalloc(&d_p, n*sizeof(double));
    cudaMalloc(&d_Ap, n*sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, n*sizeof(double));

    // Initialize residuals
    cudaMemcpy(d_r, d_b, n*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_r, n*sizeof(double), cudaMemcpyDeviceToDevice);

    double rsq_old, alpha, beta;
    double *d_rsq_old, *d_alpha, *d_beta;
    cudaMalloc(&d_rsq_old, sizeof(double));
    cudaMalloc(&d_alpha, sizeof(double));
    cudaMalloc(&d_beta, sizeof(double));

    // Compute initial residual squared
    dotProduct<<<1, 256, 256*sizeof(double)>>>(d_r, d_r, d_rsq_old, n);

    for (int k = 0; k < 1000; ++k) {
        // Compute Ap = A * p
        matrixVectorMul<<<(n+255)/256, 256>>>(d_A, d_p, d_Ap, n);

        // Compute alpha
        double pAp;
        dotProduct<<<1, 256, 256*sizeof(double)>>>(d_p, d_Ap, d_alpha, n);
        cudaMemcpy(&alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToHost);
        alpha = rsq_old / alpha;

        // Update x and r
        cublasDaxpy(n, alpha, d_p, 1, d_x, 1);
        cublasDaxpy(n, -alpha, d_Ap, 1, d_r, 1);

        // Check convergence
        double rsq_new;
        dotProduct<<<1, 256, 256*sizeof(double)>>>(d_r, d_r, &rsq_new, n);
        if (sqrt(rsq_new) < tol) break;

        // Update beta and p
        beta = rsq_new / rsq_old;
        cublasDscal(n, beta, d_p, 1);
        cublasDaxpy(n, 1.0, d_r, 1, d_p, 1);
        rsq_old = rsq_new;
    }

    // Copy result back
    cudaMemcpy(x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
}
