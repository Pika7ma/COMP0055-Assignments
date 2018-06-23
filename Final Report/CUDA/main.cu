#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <sys/time.h>
cudaError_t maxPooling(const int n, const int kn_size, float *m, float *m_);

__global__ void maxPoolingKernel1(const int n, const int kn_size, float *dev_m, float *dev_m_tmp) {
    int offset = threadIdx.x + blockIdx.x * blockDim.x;
    if (offset > n * (n - kn_size)) {
        return;
    }
    int x = offset / (n - kn_size + 1);
    int y = offset % (n - kn_size + 1);
    float max_ = -FLT_MAX;
    for (int i = 0; i < kn_size; ++i) {
        if (max_ < dev_m[x * n + y + i]) {
            max_ = dev_m[x * n + y + i];
        }
    }
    dev_m_tmp[offset] = max_;
}

__global__ void maxPoolingKernel2(const int n, const int kn_size, float *dev_m_tmp, float *dev_m_) {
    int offset = threadIdx.x + blockIdx.x * blockDim.x;
    if (offset > (n - kn_size) * (n - kn_size)) {
        return;
    }
    int x = offset / (n - kn_size + 1);
    int y = offset % (n - kn_size + 1);
    float max_ = -FLT_MAX;
    for (int i = 0; i < kn_size; ++i) {
        if (max_ < dev_m_tmp[x * (n - kn_size + 1 + i) + y]) {
            max_ = dev_m_tmp[x * (n - kn_size + 1 + i) + y];
        }
    }
    dev_m_[offset] = max_;
}

int main() {
    const int MAX_N = 4096;
    const int KN_SIZE = 128;

    float *m = new float[MAX_N * MAX_N];
    float *m_ = new float[(MAX_N - KN_SIZE + 1) * (MAX_N - KN_SIZE + 1)];

    // Add vectors in parallel.
    cudaError_t cudaStatus = maxPooling(MAX_N, KN_SIZE, m, m_);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "maxPooling failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t maxPooling(const int n, const int kn_size, float *m, float *m_) {
    float *dev_m = 0;
    float *dev_m_tmp = 0;
    float *dev_m_ = 0;
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_m, n * n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_m_tmp, n * (n - kn_size + 1) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_m_, (n - kn_size + 1) * (n - kn_size + 1) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_m, m, n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    const long maxThread = 1024;
    maxPoolingKernel1<<<(long)ceil(n * (n - kn_size + 1) / maxThread), maxThread>>>(n, kn_size, dev_m, dev_m_tmp);


    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    maxPoolingKernel2<<<(long)ceil((n - kn_size + 1) * (n - kn_size + 1) / maxThread), maxThread>>>(n, kn_size, dev_m_tmp, dev_m_);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(m_, dev_m_, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_m);
    cudaFree(dev_m_tmp);
    cudaFree(dev_m_);

    return cudaStatus;
}
