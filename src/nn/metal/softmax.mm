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

constexpr NSUInteger SM_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint SM_BLOCK = 256;

kernel void k_softmax_fw(device const float* logits [[buffer(0)]],
                         device float*       probs  [[buffer(1)]],
                         device const float* mask   [[buffer(2)]],
                         constant uint& has_mask    [[buffer(3)]],
                         constant uint& n           [[buffer(4)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[SM_BLOCK];

    // Phase 1: max over valid.
    float local_max = -1e30f;
    for (uint i = tid; i < n; i += tg_size) {
        if (has_mask && mask[i] == 0.0f) continue;
        float v = logits[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = sdata[0];

    // Phase 2: exp(x - m), zero on masked.
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        if (has_mask && mask[i] == 0.0f) {
            probs[i] = 0.0f;
            continue;
        }
        float e = exp(logits[i] - m);
        probs[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = sdata[0];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        probs[i] = probs[i] * inv;
    }
}

kernel void k_softmax_bw(device const float* probs   [[buffer(0)]],
                         device const float* dProbs  [[buffer(1)]],
                         device float*       dLogits [[buffer(2)]],
                         constant uint& n            [[buffer(3)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[SM_BLOCK];

    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += dProbs[i] * probs[i];
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = sdata[0];

    for (uint i = tid; i < n; i += tg_size) {
        dLogits[i] = probs[i] * (dProbs[i] - dot);
    }
}
)msl";

id<MTLComputePipelineState> fw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_softmax_fw"); });
    return pso;
}
id<MTLComputePipelineState> bw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_softmax_bw"); });
    return pso;
}

void run_single_block(id<MTLComputePipelineState> pso,
                      void (^bind)(id<MTLComputeCommandEncoder>)) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        // Single threadgroup of size SM_BLOCK.
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SM_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void softmax_forward_gpu(const GpuTensor& logits, GpuTensor& probs,
                         const float* d_mask) {
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols) {
        probs.resize(logits.rows, logits.cols);
    }
    if (n == 0) return;
    id<MTLBuffer> bL = buffer_for(logits);
    NSUInteger oL = buffer_offset_for(logits);
    id<MTLBuffer> bP = buffer_for(probs);
    NSUInteger oP = buffer_offset_for(probs);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bL;
    NSUInteger oM_arg = bM ? oM : oL;
    const uint32_t nu = static_cast<uint32_t>(n);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    run_single_block(fw_pso(), ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bL offset:oL atIndex:0];
        [enc setBuffer:bP offset:oP atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:4];
    });
}

void softmax_backward_gpu(const GpuTensor& probs, const GpuTensor& dProbs,
                          GpuTensor& dLogits) {
    const int n = probs.size();
    if (dLogits.rows != probs.rows || dLogits.cols != probs.cols) {
        dLogits.resize(probs.rows, probs.cols);
    }
    if (n == 0) return;
    id<MTLBuffer> bP  = buffer_for(probs);
    NSUInteger oP = buffer_offset_for(probs);
    id<MTLBuffer> bdP = buffer_for(dProbs);
    NSUInteger odP = buffer_offset_for(dProbs);
    id<MTLBuffer> bdL = buffer_for(dLogits);
    NSUInteger odL = buffer_offset_for(dLogits);
    const uint32_t nu = static_cast<uint32_t>(n);
    run_single_block(bw_pso(), ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bP offset:oP atIndex:0];
        [enc setBuffer:bdP offset:odP atIndex:1];
        [enc setBuffer:bdL offset:odL atIndex:2];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:3];
    });
}

} // namespace brogameagent::nn::gpu
