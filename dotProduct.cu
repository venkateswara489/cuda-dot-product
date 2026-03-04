#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define N 100000000
#define THREADS 256
#define BLOCKS 256

// CUDA Kernel
__global__ void dotProductKernel(double* A, double* B, double* partial, int n)
{
    __shared__ double cache[THREADS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double temp = 0.0;

    while (gid < n)
    {
        temp += A[gid] * B[gid];
        gid += stride;
    }

    cache[tid] = temp;
    __syncthreads();

    int i = blockDim.x / 2;

    while (i != 0)
    {
        if (tid < i)
            cache[tid] += cache[tid + i];

        __syncthreads();
        i /= 2;
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


// OpenMP Version
double dotProductOMP(double* A, double* B)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; i++)
        sum += A[i] * B[i];

    return sum;
}


// CUDA Version
double dotProductCUDA(double* A, double* B)
{
    double *d_A, *d_B, *d_partial;

    double* h_partial = new double[BLOCKS];

    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_partial, BLOCKS * sizeof(double));

    cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dotProductKernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_partial, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(h_partial, d_partial, BLOCKS * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;

    for (int i = 0; i < BLOCKS; i++)
        sum += h_partial[i];

    std::cout << "Kernel Time (ms): " << kernelTime << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);

    delete[] h_partial;

    return sum;
}


// MAIN PROGRAM
int main()
{
    std::cout << "Generating vectors..." << std::endl;

    double* A = new double[N];
    double* B = new double[N];

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (long long i = 0; i < N; i++)
    {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    double cpuResult = dotProductOMP(A, B);

    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();

    double gpuResult = dotProductCUDA(A, B);

    auto t3 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cpuTime = t1 - t0;
    std::chrono::duration<double> gpuTime = t3 - t2;

    std::cout << "CPU Result: " << cpuResult << std::endl;
    std::cout << "GPU Result: " << gpuResult << std::endl;

    std::cout << "CPU Time: " << cpuTime.count() << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpuTime.count() << " seconds" << std::endl;

    std::cout << "Speedup: "
              << cpuTime.count() / gpuTime.count()
              << std::endl;

    delete[] A;
    delete[] B;

    return 0;
}
