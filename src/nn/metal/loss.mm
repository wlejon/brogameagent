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

constexpr NSUInteger LOSS_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LOSS_BLOCK = 256;

kernel void k_mse_forward(device const float* pred    [[buffer(0)]],
                          device const float* target  [[buffer(1)]],
                          device float*       out_sum [[buffer(2)]],
                          constant uint& n            [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LOSS_BLOCK];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = pred[i] - target[i];
        local += d * d;
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_sum[0] = sdata[0];
}

kernel void k_mse_backward(device const float* pred   [[buffer(0)]],
                           device const float* target [[buffer(1)]],
                           device float*       dPred  [[buffer(2)]],
                           constant uint& n           [[buffer(3)]],
                           constant float& scale      [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dPred[i] = scale * (pred[i] - target[i]);
}

kernel void k_softmax_xent_fused(device const float* logits [[buffer(0)]],
                                 device const float* target [[buffer(1)]],
                                 device const float* mask   [[buffer(2)]],
                                 constant uint& has_mask    [[buffer(3)]],
                                 device float*       probs  [[buffer(4)]],
                                 device float*       dLogits[[buffer(5)]],
                                 device float*       out_loss[[buffer(6)]],
                                 constant uint& n           [[buffer(7)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LOSS_BLOCK];

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

    float local_loss = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        if (has_mask && mask[i] == 0.0f) {
            dLogits[i] = 0.0f;
            continue;
        }
        float p = probs[i] * inv;
        probs[i] = p;
        float t = target[i];
        if (t > 0.0f) {
            float pc = p > 1e-12f ? p : 1e-12f;
            local_loss -= t * log(pc);
        }
        dLogits[i] = p - t;
    }
    sdata[tid] = local_loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out_loss[0] = sdata[0];
}

kernel void k_mse_per_sample(device const float* pred   [[buffer(0)]],
                             device const float* target [[buffer(1)]],
                             device float*       dPred  [[buffer(2)]],
                             device float*       loss   [[buffer(3)]],
                             constant uint& B           [[buffer(4)]],
                             uint b [[thread_position_in_grid]]) {
    if (b >= B) return;
    float d = pred[b] - target[b];
    dPred[b] = d;
    loss[b] = 0.5f * d * d;
}

// One threadgroup per (sample, head) tile. Slice is [off, end) of row b.
kernel void k_softmax_xent_fused_batched(device const float* logits        [[buffer(0)]],
                                         device const float* target        [[buffer(1)]],
                                         device const float* mask          [[buffer(2)]],
                                         constant uint& has_mask           [[buffer(3)]],
                                         device const int*   head_offsets  [[buffer(4)]],
                                         device float*       probs         [[buffer(5)]],
                                         device float*       dLogits       [[buffer(6)]],
                                         device atomic_float* loss_per_sample[[buffer(7)]],
                                         constant uint& B                  [[buffer(8)]],
                                         constant uint& n_act              [[buffer(9)]],
                                         uint2 gid [[threadgroup_position_in_grid]],
                                         uint2 tid2 [[thread_position_in_threadgroup]],
                                         uint2 tgsz2 [[threads_per_threadgroup]]) {
    uint tid = tid2.x;
    uint tg_size = tgsz2.x;
    threadgroup float sdata[LOSS_BLOCK];
    uint h = gid.x;
    uint b = gid.y;
    if (b >= B) return;
    int off = head_offsets[h];
    int end = head_offsets[h + 1];
    int len = end - off;
    uint row_off = b * n_act + uint(off);

    device const float* logits_row  = logits  + row_off;
    device const float* target_row  = target  + row_off;
    device const float* mask_row    = mask    + row_off;
    device float*       probs_row   = probs   + row_off;
    device float*       dLogits_row = dLogits + row_off;

    float local_max = -1e30f;
    for (int i = int(tid); i < len; i += int(tg_size)) {
        if (has_mask && mask_row[i] == 0.0f) continue;
        float v = logits_row[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], c = sdata[tid + s];
            sdata[tid] = a > c ? a : c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = sdata[0];

    float local_sum = 0.0f;
    for (int i = int(tid); i < len; i += int(tg_size)) {
        if (has_mask && mask_row[i] == 0.0f) {
            probs_row[i] = 0.0f;
            continue;
        }
        float e = exp(logits_row[i] - m);
        probs_row[i] = e;
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

    float local_loss = 0.0f;
    for (int i = int(tid); i < len; i += int(tg_size)) {
        if (has_mask && mask_row[i] == 0.0f) {
            dLogits_row[i] = 0.0f;
            continue;
        }
        float p = probs_row[i] * inv;
        probs_row[i] = p;
        float t = target_row[i];
        if (t > 0.0f) {
            float pc = p > 1e-12f ? p : 1e-12f;
            local_loss -= t * log(pc);
        }
        dLogits_row[i] = p - t;
    }
    sdata[tid] = local_loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        atomic_fetch_add_explicit(&loss_per_sample[b], sdata[0], memory_order_relaxed);
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
DEF_PSO(pso_mse_fw, @"k_mse_forward")
DEF_PSO(pso_mse_bw, @"k_mse_backward")
DEF_PSO(pso_xent, @"k_softmax_xent_fused")
DEF_PSO(pso_mse_per_sample, @"k_mse_per_sample")
DEF_PSO(pso_xent_batched, @"k_softmax_xent_fused_batched")
#undef DEF_PSO

} // namespace

float mse_vec_forward_gpu(const GpuTensor& pred, const GpuTensor& target) {
    const int n = pred.size();
    if (n == 0) return 0.0f;
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:sizeof(float)
                        options:MTLResourceStorageModeShared];
        float* sptr = static_cast<float*>([scratch contents]);
        sptr[0] = 0.0f;
        id<MTLComputePipelineState> pso = pso_mse_fw();
        id<MTLBuffer> bp = buffer_for(pred);
    NSUInteger op = buffer_offset_for(pred);
        id<MTLBuffer> bt = buffer_for(target);
    NSUInteger ot = buffer_offset_for(target);
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bt offset:ot atIndex:1];
        [enc setBuffer:scratch offset:0 atIndex:2];
        const uint32_t nu = static_cast<uint32_t>(n);
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LOSS_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return sptr[0] / static_cast<float>(n);
    }
}

void mse_vec_backward_gpu(const GpuTensor& pred, const GpuTensor& target,
                          GpuTensor& dPred) {
    const int n = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols) {
        dPred.resize(pred.rows, pred.cols);
    }
    if (n == 0) return;
    const float scale = 2.0f / static_cast<float>(n);
    const uint32_t nu = static_cast<uint32_t>(n);
    id<MTLComputePipelineState> pso = pso_mse_bw();
    id<MTLBuffer> bp  = buffer_for(pred);
    NSUInteger op = buffer_offset_for(pred);
    id<MTLBuffer> bt  = buffer_for(target);
    NSUInteger ot = buffer_offset_for(target);
    id<MTLBuffer> bdp = buffer_for(dPred);
    NSUInteger odp = buffer_offset_for(dPred);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bt offset:ot atIndex:1];
        [enc setBuffer:bdp offset:odp atIndex:2];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&scale length:sizeof(float) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

float softmax_xent_fused_gpu(const GpuTensor& logits, const GpuTensor& target,
                             const float* d_mask,
                             GpuTensor& probs, GpuTensor& dLogits) {
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols) {
        probs.resize(logits.rows, logits.cols);
    }
    if (dLogits.rows != logits.rows || dLogits.cols != logits.cols) {
        dLogits.resize(logits.rows, logits.cols);
    }
    if (n == 0) return 0.0f;
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:sizeof(float)
                        options:MTLResourceStorageModeShared];
        float* sptr = static_cast<float*>([scratch contents]);
        sptr[0] = 0.0f;
        id<MTLBuffer> bL = buffer_for(logits);
    NSUInteger oL = buffer_offset_for(logits);
        id<MTLBuffer> bT = buffer_for(target);
    NSUInteger oT = buffer_offset_for(target);
        id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
        NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
        id<MTLBuffer> bM_arg = bM ? bM : bL;
        NSUInteger oM_arg = bM ? oM : oL;
        id<MTLBuffer> bP = buffer_for(probs);
    NSUInteger oP = buffer_offset_for(probs);
        id<MTLBuffer> bdL = buffer_for(dLogits);
    NSUInteger odL = buffer_offset_for(dLogits);
        const uint32_t nu = static_cast<uint32_t>(n);
        const uint32_t has_mask = d_mask ? 1u : 0u;
        id<MTLComputePipelineState> pso = pso_xent();
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bL offset:oL atIndex:0];
        [enc setBuffer:bT offset:oT atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bP offset:oP atIndex:4];
        [enc setBuffer:bdL offset:odL atIndex:5];
        [enc setBuffer:scratch offset:0 atIndex:6];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LOSS_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return sptr[0];
    }
}

void mse_vec_per_sample_gpu(const GpuTensor& pred, const GpuTensor& target,
                            GpuTensor& dPred, GpuTensor& loss_per_sample) {
    const int B = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols)
        dPred.resize(pred.rows, pred.cols);
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1)
        loss_per_sample.resize(B, 1);
    if (B == 0) return;
    id<MTLComputePipelineState> pso = pso_mse_per_sample();
    id<MTLBuffer> bp = buffer_for(pred);
    NSUInteger op = buffer_offset_for(pred);
    id<MTLBuffer> bt = buffer_for(target);
    NSUInteger ot = buffer_offset_for(target);
    id<MTLBuffer> bdp = buffer_for(dPred);
    NSUInteger odp = buffer_offset_for(dPred);
    id<MTLBuffer> bL = buffer_for(loss_per_sample);
    NSUInteger oL = buffer_offset_for(loss_per_sample);
    const uint32_t Bu = static_cast<uint32_t>(B);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bt offset:ot atIndex:1];
        [enc setBuffer:bdp offset:odp atIndex:2];
        [enc setBuffer:bL offset:oL atIndex:3];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(B, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void softmax_xent_fused_batched_gpu(const GpuTensor& logits_BL,
                                    const GpuTensor& target_BL,
                                    const float* d_mask_BL,
                                    const int* d_head_offsets,
                                    int n_heads,
                                    GpuTensor& probs_BL,
                                    GpuTensor& dLogits_BL,
                                    GpuTensor& loss_per_sample) {
    const int B     = logits_BL.rows;
    const int n_act = logits_BL.cols;
    if (probs_BL.rows != B || probs_BL.cols != n_act)
        probs_BL.resize(B, n_act);
    if (dLogits_BL.rows != B || dLogits_BL.cols != n_act)
        dLogits_BL.resize(B, n_act);
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1)
        loss_per_sample.resize(B, 1);
    if (B == 0 || n_act == 0 || n_heads <= 0) return;

    // Zero loss accumulator (unified memory).
    for (int b = 0; b < B; ++b) loss_per_sample.data[b] = 0.0f;

    id<MTLComputePipelineState> pso = pso_xent_batched();
    id<MTLBuffer> bL = buffer_for(logits_BL);
    NSUInteger oL = buffer_offset_for(logits_BL);
    id<MTLBuffer> bT = buffer_for(target_BL);
    NSUInteger oT = buffer_offset_for(target_BL);
    id<MTLBuffer> bM = d_mask_BL ? pool_lookup(d_mask_BL) : nil;
    NSUInteger oM = d_mask_BL ? pool_lookup_offset(d_mask_BL) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bL;
    NSUInteger oM_arg = bM ? oM : oL;
    id<MTLBuffer> bH = pool_lookup(d_head_offsets);
    NSUInteger oH = pool_lookup_offset(d_head_offsets);
    id<MTLBuffer> bP = buffer_for(probs_BL);
    NSUInteger oP = buffer_offset_for(probs_BL);
    id<MTLBuffer> bdL = buffer_for(dLogits_BL);
    NSUInteger odL = buffer_offset_for(dLogits_BL);
    id<MTLBuffer> bLoss = buffer_for(loss_per_sample);
    NSUInteger oLoss = buffer_offset_for(loss_per_sample);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Nu = static_cast<uint32_t>(n_act);
    const uint32_t has_mask = d_mask_BL ? 1u : 0u;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bL offset:oL atIndex:0];
        [enc setBuffer:bT offset:oT atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bH offset:oH atIndex:4];
        [enc setBuffer:bP offset:oP atIndex:5];
        [enc setBuffer:bdL offset:odL atIndex:6];
        [enc setBuffer:bLoss offset:oLoss atIndex:7];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, B, 1)
            threadsPerThreadgroup:MTLSizeMake(LOSS_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brogameagent::nn::gpu
