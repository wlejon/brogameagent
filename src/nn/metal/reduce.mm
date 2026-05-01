#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_masked_mean_pool_fw(device const float* X    [[buffer(0)]],
                                  device const float* mask [[buffer(1)]],
                                  constant uint& has_mask  [[buffer(2)]],
                                  device float*       y    [[buffer(3)]],
                                  constant uint& K         [[buffer(4)]],
                                  constant uint& D         [[buffer(5)]],
                                  uint j [[thread_position_in_grid]]) {
    if (j >= D) return;
    uint num_valid = 0;
    if (has_mask) {
        for (uint k = 0; k < K; ++k) num_valid += (mask[k] != 0.0f) ? 1u : 0u;
    } else {
        num_valid = K;
    }
    if (num_valid == 0) { y[j] = 0.0f; return; }
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float m = has_mask ? mask[k] : 1.0f;
        if (m != 0.0f) acc += X[k * D + j];
    }
    y[j] = acc / float(num_valid);
}

kernel void k_masked_mean_pool_bw(device const float* dY   [[buffer(0)]],
                                  device const float* mask [[buffer(1)]],
                                  constant uint& has_mask  [[buffer(2)]],
                                  device float*       dX   [[buffer(3)]],
                                  constant uint& K         [[buffer(4)]],
                                  constant uint& D         [[buffer(5)]],
                                  constant uint& num_valid [[buffer(6)]],
                                  uint idx [[thread_position_in_grid]]) {
    uint total = K * D;
    if (idx >= total) return;
    uint k = idx / D;
    uint j = idx - k * D;
    float m = has_mask ? mask[k] : 1.0f;
    if (num_valid == 0 || m == 0.0f) {
        dX[idx] = 0.0f;
    } else {
        dX[idx] = dY[j] / float(num_valid);
    }
}
)msl";

id<MTLComputePipelineState> fw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_masked_mean_pool_fw"); });
    return pso;
}
id<MTLComputePipelineState> bw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_masked_mean_pool_bw"); });
    return pso;
}

void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger n,
                void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void masked_mean_pool_forward_gpu(const GpuTensor& X, const float* d_mask,
                                  GpuTensor& y) {
    const int K = X.rows;
    const int D = X.cols;
    if (y.rows != D || y.cols != 1) y.resize(D, 1);
    if (D == 0) return;
    id<MTLBuffer> bX = buffer_for(X);
    NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bY = buffer_for(y);
    NSUInteger oY = buffer_offset_for(y);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    // Provide non-nil dummy mask buffer when has_mask==0 (use bX).
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    dispatch1d(fw_pso(), D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:1];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:2];
        [enc setBuffer:bY offset:oY atIndex:3];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
    });
}

void masked_mean_pool_backward_gpu(const GpuTensor& dY, const float* d_mask,
                                   int K, GpuTensor& dX) {
    const int D = dY.size();
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    const int total = K * D;
    if (total == 0) return;
    int num_valid = 0;
    if (d_mask) {
        // Unified memory: d_mask is host-readable.
        for (int k = 0; k < K; ++k) num_valid += (d_mask[k] != 0.0f);
    } else {
        num_valid = K;
    }
    id<MTLBuffer> bdY = buffer_for(dY);
    NSUInteger odY = buffer_offset_for(dY);
    id<MTLBuffer> bdX = buffer_for(dX);
    NSUInteger odX = buffer_offset_for(dX);
    id<MTLBuffer> bM  = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bdY;
    NSUInteger oM_arg = bM ? oM : odY;
    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const uint32_t nv = static_cast<uint32_t>(num_valid);
    dispatch1d(bw_pso(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdY offset:odY atIndex:0];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:1];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:2];
        [enc setBuffer:bdX offset:odX atIndex:3];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&nv length:sizeof(uint32_t) atIndex:6];
    });
}

} // namespace brogameagent::nn::gpu
