#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// CUDA kernel for SAXPY
__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

// CPU version of SAXPY
void cpu_saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

// CPU timing utility
double get_cpu_time_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char *argv[])
{
    int start_power = 10;
    int max_power = 30;

    if (argc >= 2)
    {
        max_power = atoi(argv[1]);
        if (max_power <= start_power || max_power > 30)
        {
            printf("Invalid max power (must be > %d and <= 30)\n", start_power);
            return -1;
        }
    }

    FILE *fp = fopen("timing_gpu_details.csv", "w");
    if (!fp)
    {
        perror("Failed to open timing_gpu_details.csv");
        return -1;
    }

    fprintf(fp, "PowerOf2,N,MallocTime_ms,H2DTime_ms,KernelTime_ms,D2HTime_ms,FreeTime_ms,TotalGPUTime_ms,CPUTime_ms\n");

    for (int power = start_power; power <= max_power; power++)
    {
        int N = 1 << power;
        float *x, *y, *d_x, *d_y;
        float malloc_time = 0.0f, h2d_time = 0.0f, kernel_time = 0.0f;
        float d2h_time = 0.0f, free_time = 0.0f, total_time = 0.0f;

        // Host allocations
        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        // CPU timing
        float *x_cpu = (float *)malloc(N * sizeof(float));
        float *y_cpu = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++)
        {
            x_cpu[i] = 1.0f;
            y_cpu[i] = 2.0f;
        }
        double cpu_start = get_cpu_time_ms();
        cpu_saxpy(N, 2.0f, x_cpu, y_cpu);
        double cpu_end = get_cpu_time_ms();
        double cpu_time = cpu_end - cpu_start;
        free(x_cpu);
        free(y_cpu);

        // CUDA events
        cudaEvent_t malloc_start, malloc_stop, h2d_start, h2d_stop;
        cudaEvent_t kernel_start, kernel_stop, d2h_start, d2h_stop;
        cudaEvent_t free_start, free_stop, total_start, total_stop;

        cudaEventCreate(&malloc_start);
        cudaEventCreate(&malloc_stop);
        cudaEventCreate(&h2d_start);
        cudaEventCreate(&h2d_stop);
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
        cudaEventCreate(&d2h_start);
        cudaEventCreate(&d2h_stop);
        cudaEventCreate(&free_start);
        cudaEventCreate(&free_stop);
        cudaEventCreate(&total_start);
        cudaEventCreate(&total_stop);

        // Total start
        cudaEventRecord(total_start);

        // cudaMalloc
        cudaEventRecord(malloc_start);
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaEventRecord(malloc_stop);
        cudaEventSynchronize(malloc_stop);
        cudaEventElapsedTime(&malloc_time, malloc_start, malloc_stop);

        // cudaMemcpy H2D
        cudaEventRecord(h2d_start);
        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(h2d_stop);
        cudaEventSynchronize(h2d_stop);
        cudaEventElapsedTime(&h2d_time, h2d_start, h2d_stop);

        // Kernel execution
        cudaEventRecord(kernel_start);
        saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

        // cudaMemcpy D2H
        cudaEventRecord(d2h_start);
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(d2h_stop);
        cudaEventSynchronize(d2h_stop);
        cudaEventElapsedTime(&d2h_time, d2h_start, d2h_stop);

        // cudaFree
        cudaEventRecord(free_start);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaEventRecord(free_stop);
        cudaEventSynchronize(free_stop);
        cudaEventElapsedTime(&free_time, free_start, free_stop);

        // Total stop
        cudaEventRecord(total_stop);
        cudaEventSynchronize(total_stop);
        cudaEventElapsedTime(&total_time, total_start, total_stop);

        // Output
        printf("N = 2^%d (%d): malloc=%.2f, H2D=%.2f, kernel=%.2f, D2H=%.2f, free=%.2f, total=%.2f, CPU=%.2f ms\n",
               power, N, malloc_time, h2d_time, kernel_time, d2h_time, free_time, total_time, cpu_time);

        fprintf(fp, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                power, N, malloc_time, h2d_time, kernel_time, d2h_time, free_time, total_time, cpu_time);

        // Cleanup
        free(x);
        free(y);
        cudaEventDestroy(malloc_start);
        cudaEventDestroy(malloc_stop);
        cudaEventDestroy(h2d_start);
        cudaEventDestroy(h2d_stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_stop);
        cudaEventDestroy(free_start);
        cudaEventDestroy(free_stop);
        cudaEventDestroy(total_start);
        cudaEventDestroy(total_stop);
    }

    fclose(fp);
    return 0;
}
