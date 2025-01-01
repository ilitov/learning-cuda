#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include "cudaUtils.hpp"

cudaError_t parallelAdd(const float *a, const float *b, float *c, int size);

#endif  // !_KERNEL_HPP_
