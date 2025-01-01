#include "cudaUtils.hpp"
#include "kernel.hpp"

__global__ void _parallelAdd(const float *a, const float *b, float *c, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

cudaError_t parallelAdd(const float *a, const float *b, float *c, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    _parallelAdd<<<blocks, threads>>>(a, b, c, size);
    return cudaDeviceSynchronize();
}
