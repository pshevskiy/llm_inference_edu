#include <metal_stdlib>
using namespace metal;

// Each lane accesses the same element in three contiguous FP32 buffers.
// n is the validated logical length; the rounded final threadgroup has a tail.
kernel void residual_add(device const float* x [[buffer(0)]],
                         device const float* residual [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    if (index < n) { y[index] = x[index] + residual[index]; }
}
