#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[])
{
    int power = 15; // default
    if (argc >= 2)
    {
        power = atoi(argv[1]);
        if (power < 1)
        {
            printf("Invalid power value (must be between 1 and 30)\n");
            return -1;
        }
    }

    int N = 1 << power;
    float *x, *y, *d_x, *d_y;

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // Total time start
    cudaEventRecord(start_total);

    // Host malloc
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    // Initialize input
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Device malloc
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Start kernel-only timer
    cudaEventRecord(start_kernel);

    // Launch kernel
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    // Stop kernel-only timer
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // Copy result back
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Total time end
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Measure times
    float kernel_time = 0.0f, total_time = 0.0f;
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_time, start_total, stop_total);

    // Output
    printf("N = 2^%d (%d elements)\n", power, N);
    printf("Kernel time: %.4f ms\n", kernel_time);
    printf("Total time:  %.4f ms\n", total_time);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}
