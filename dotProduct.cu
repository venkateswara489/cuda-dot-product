#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define N 100000000
#define THREADS 256
#define BLOCKS 2048   // Increased for better GPU utilization

// ================= CUDA KERNEL =================
__global__ void dotProductKernel(double* A, double* B,
                                 double* partial, int n)
{
    __shared__ double cache[THREADS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double localSum = 0.0;

    // Grid-stride loop
    while (gid < n)
    {
        localSum += A[gid] * B[gid];
        gid += stride;
    }

    cache[tid] = localSum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            cache[tid] += cache[tid + s];

        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


// ================= OPENMP VERSION =================
double dotProductOMP(double* A, double* B)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; i++)
        sum += A[i] * B[i];

    return sum;
}


// ================= CUDA VERSION =================
double dotProductCUDA(double* A, double* B, float &kernelTimeMs)
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

    cudaEventElapsedTime(&kernelTimeMs, start, stop);

    cudaMemcpy(h_partial, d_partial,
               BLOCKS * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < BLOCKS; i++)
        sum += h_partial[i];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);
    delete[] h_partial;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return sum;
}


// ================= MAIN =================
int main()
{
    // Print GPU name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;

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

    // -------- CPU Timing --------
    auto t0 = std::chrono::high_resolution_clock::now();
    double cpuResult = dotProductOMP(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    // -------- GPU Timing --------
    auto t2 = std::chrono::high_resolution_clock::now();
    float kernelTime = 0.0f;
    double gpuResult = dotProductCUDA(A, B, kernelTime);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cpuTime = t1 - t0;
    std::chrono::duration<double> gpuTime = t3 - t2;

    // -------- Output --------
    std::cout << "\nCPU Result: " << cpuResult << std::endl;
    std::cout << "GPU Result: " << gpuResult << std::endl;

    std::cout << "\nCPU Time: " << cpuTime.count() << " seconds" << std::endl;
    std::cout << "GPU Total Time: " << gpuTime.count() << " seconds" << std::endl;
    std::cout << "GPU Kernel Time: " << kernelTime << " ms" << std::endl;

    std::cout << "\nTotal Speedup (CPU/GPU): "
              << cpuTime.count() / gpuTime.count()
              << std::endl;

    std::cout << "Kernel Speedup (CPU/kernel): "
              << (cpuTime.count() * 1000.0) / kernelTime
              << std::endl;

    delete[] A;
    delete[] B;

    return 0;
}
