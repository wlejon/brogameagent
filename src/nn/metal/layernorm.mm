#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger LN_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

kernel void k_ln_forward(device const float* x       [[buffer(0)]],
                         device const float* gamma   [[buffer(1)]],
                         device const float* beta    [[buffer(2)]],
                         device float*       y       [[buffer(3)]],
                         device float*       xhat    [[buffer(4)]],
                         device float*       scratch [[buffer(5)]],
                         constant uint& n            [[buffer(6)]],
                         constant float& eps         [[buffer(7)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) local += x[i];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(n);

    float local_v = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = x[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(n);
    float rstd = rsqrt(var + eps);
    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }
    for (uint i = tid; i < n; i += tg_size) {
        float xh = (x[i] - mean) * rstd;
        xhat[i] = xh;
        y[i] = gamma[i] * xh + beta[i];
    }
}

kernel void k_ln_backward(device const float* dY     [[buffer(0)]],
                          device const float* xhat   [[buffer(1)]],
                          device const float* gamma  [[buffer(2)]],
                          constant float& rstd       [[buffer(3)]],
                          device float*       dX     [[buffer(4)]],
                          device float*       dGamma [[buffer(5)]],
                          device float*       dBeta  [[buffer(6)]],
                          constant uint& n           [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g = dY[i];
        dGamma[i] += g * xhat[i];
        dBeta[i]  += g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += dY[i] * gamma[i];
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += dY[i] * gamma[i] * xhat[i];
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = dY[i] * gamma[i];
        dX[i] = scale * (nf * dxh - sum_dxh - xhat[i] * sum_dxh_xhat);
    }
}

kernel void k_ln_forward_inference_batched(device const float* x      [[buffer(0)]],
                                           device const float* gamma  [[buffer(1)]],
                                           device const float* beta   [[buffer(2)]],
                                           device float*       y      [[buffer(3)]],
                                           constant uint& R           [[buffer(4)]],
                                           constant uint& D           [[buffer(5)]],
                                           constant float& eps        [[buffer(6)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    if (row >= R) return;
    device const float* xrow = x + row * D;
    device float*       yrow = y + row * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tg_size) local += xrow[i];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(D);

    float local_v = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float d = xrow[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(D);
    float rstd = rsqrt(var + eps);
    for (uint i = tid; i < D; i += tg_size) {
        float xh = (xrow[i] - mean) * rstd;
        yrow[i] = xh * gamma[i] + beta[i];
    }
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_fw, @"k_ln_forward")
DEF_PSO(pso_bw, @"k_ln_backward")
DEF_PSO(pso_fw_inf, @"k_ln_forward_inference_batched")
#undef DEF_PSO

} // namespace

void layernorm_forward_gpu(const GpuTensor& x,
                           const GpuTensor& gamma, const GpuTensor& beta,
                           GpuTensor& y, GpuTensor& xhat,
                           float& mean_out, float& rstd_out,
                           float eps) {
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (xhat.rows != x.rows || xhat.cols != x.cols) xhat.resize(x.rows, x.cols);
    if (n == 0) { mean_out = 0.0f; rstd_out = 0.0f; return; }
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:2 * sizeof(float)
                        options:MTLResourceStorageModeShared];
        float* sptr = static_cast<float*>([scratch contents]);
        sptr[0] = 0.0f; sptr[1] = 0.0f;
        id<MTLComputePipelineState> pso = pso_fw();
        id<MTLBuffer> bx = buffer_for(x);
    NSUInteger ox = buffer_offset_for(x);
        id<MTLBuffer> bg = buffer_for(gamma);
    NSUInteger og = buffer_offset_for(gamma);
        id<MTLBuffer> bb = buffer_for(beta);
    NSUInteger ob = buffer_offset_for(beta);
        id<MTLBuffer> by = buffer_for(y);
    NSUInteger oy = buffer_offset_for(y);
        id<MTLBuffer> bh = buffer_for(xhat);
    NSUInteger oh = buffer_offset_for(xhat);
        const uint32_t nu = static_cast<uint32_t>(n);
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBuffer:bh offset:oh atIndex:4];
        [enc setBuffer:scratch offset:0 atIndex:5];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&eps length:sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        mean_out = sptr[0];
        rstd_out = sptr[1];
    }
}

void layernorm_backward_gpu(const GpuTensor& dY, const GpuTensor& xhat,
                            const GpuTensor& gamma, float rstd,
                            GpuTensor& dX,
                            GpuTensor& dGamma, GpuTensor& dBeta) {
    const int n = dY.size();
    if (dX.rows != dY.rows || dX.cols != dY.cols) dX.resize(dY.rows, dY.cols);
    if (n == 0) return;
    id<MTLComputePipelineState> pso = pso_bw();
    id<MTLBuffer> bdy = buffer_for(dY);
    NSUInteger ody = buffer_offset_for(dY);
    id<MTLBuffer> bxh = buffer_for(xhat);
    NSUInteger oxh = buffer_offset_for(xhat);
    id<MTLBuffer> bg  = buffer_for(gamma);
    NSUInteger og = buffer_offset_for(gamma);
    id<MTLBuffer> bdx = buffer_for(dX);
    NSUInteger odx = buffer_offset_for(dX);
    id<MTLBuffer> bdg = buffer_for(dGamma);
    NSUInteger odg = buffer_offset_for(dGamma);
    id<MTLBuffer> bdb = buffer_for(dBeta);
    NSUInteger odb = buffer_offset_for(dBeta);
    const uint32_t nu = static_cast<uint32_t>(n);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:bxh offset:oxh atIndex:1];
        [enc setBuffer:bg offset:og atIndex:2];
        [enc setBytes:&rstd length:sizeof(float) atIndex:3];
        [enc setBuffer:bdx offset:odx atIndex:4];
        [enc setBuffer:bdg offset:odg atIndex:5];
        [enc setBuffer:bdb offset:odb atIndex:6];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void layernorm_forward_inference_batched_gpu(const GpuTensor& X_RD,
                                             const GpuTensor& gamma,
                                             const GpuTensor& beta,
                                             GpuTensor& Y_RD,
                                             float eps) {
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D) Y_RD.resize(R, D);
    if (R == 0 || D == 0) return;
    id<MTLComputePipelineState> pso = pso_fw_inf();
    id<MTLBuffer> bx = buffer_for(X_RD);
    NSUInteger ox = buffer_offset_for(X_RD);
    id<MTLBuffer> bg = buffer_for(gamma);
    NSUInteger og = buffer_offset_for(gamma);
    id<MTLBuffer> bb = buffer_for(beta);
    NSUInteger ob = buffer_offset_for(beta);
    id<MTLBuffer> by = buffer_for(Y_RD);
    NSUInteger oy = buffer_offset_for(Y_RD);
    const uint32_t Ru = static_cast<uint32_t>(R);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&Ru length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&eps length:sizeof(float) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brogameagent::nn::gpu
