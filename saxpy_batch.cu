#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[])
{
    int start_power = 10;
    int max_power = 25; // default

    if (argc >= 2)
    {
        max_power = atoi(argv[1]);
        if (max_power <= start_power || max_power > 30)
        {
            printf("Invalid max power (must be > 10 and <= 30)\n");
            return -1;
        }
    }

    FILE *fp = fopen("timing.csv", "w");
    if (!fp)
    {
        perror("Failed to open timing.csv");
        return -1;
    }
    fprintf(fp, "N,KernelTime_ms,TotalTime_ms\n");

    for (int power = start_power; power <= max_power; power += 1)
    {
        int N = 1 << power;
        float *x, *y, *d_x, *d_y;

        cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
        cudaEventCreate(&start_total);
        cudaEventCreate(&stop_total);
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        cudaEventRecord(start_total);

        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));

        for (int i = 0; i < N; i++)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));

        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(start_kernel);

        saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop_total);
        cudaEventSynchronize(stop_total);

        float kernel_time = 0.0f, total_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
        cudaEventElapsedTime(&total_time, start_total, stop_total);

        printf("N = 2^%d (%d): kernel = %.4f ms, total = %.4f ms\n",
               power, N, kernel_time, total_time);
        fprintf(fp, "%d,%.4f,%.4f\n", power, kernel_time, total_time);

        // Cleanup
        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
    }

    fclose(fp);
    return 0;
}
