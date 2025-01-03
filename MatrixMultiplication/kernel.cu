#include "kernel.hpp"

__global__ void _parallelAdd(const float *a, const float *b, float *c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float res = 0.f;

        for (int k = 0; k < n; ++k) {
            res += a[row * n + k] * b[row * k + col];
        }

        c[row * n + col] = res;
    }
}

cudaError_t parallelSquareMatmul(const float *a, const float *b, float *c, int n) {
    const int threads = 32;
    const int blocks = (n + threads - 1) / threads;  // cover each axis by enough blocks

    static_assert(threads * threads <= 1024);

    _parallelAdd<<<dim3(blocks, blocks, 1), dim3(threads, threads, 1)>>>(a, b, c, n);
    return cudaDeviceSynchronize();
}
