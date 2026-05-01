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

kernel void k_emb_fw(device const float* table [[buffer(0)]],
                     device const int*   idx   [[buffer(1)]],
                     device float*       out   [[buffer(2)]],
                     constant uint& B          [[buffer(3)]],
                     constant uint& D          [[buffer(4)]],
                     uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    out[t] = table[uint(row) * D + j];
}

kernel void k_emb_bw(device const float* dOut          [[buffer(0)]],
                     device const int*   idx           [[buffer(1)]],
                     device atomic_float* dTable        [[buffer(2)]],
                     constant uint& B                  [[buffer(3)]],
                     constant uint& D                  [[buffer(4)]],
                     uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    atomic_fetch_add_explicit(&dTable[uint(row) * D + j],
                              dOut[t], memory_order_relaxed);
}
)msl";

id<MTLComputePipelineState> fw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_fw"); });
    return pso;
}
id<MTLComputePipelineState> bw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_bw"); });
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

void embedding_lookup_forward_gpu(const GpuTensor& table,
                                  const int32_t* d_idx, int B,
                                  GpuTensor& out) {
    const int D = table.cols;
    if (out.rows != B || out.cols != D) out.resize(B, D);
    const int total = B * D;
    if (total == 0) return;
    id<MTLBuffer> bT = buffer_for(table);
    NSUInteger oT = buffer_offset_for(table);
    id<MTLBuffer> bO = buffer_for(out);
    NSUInteger oO = buffer_offset_for(out);
    id<MTLBuffer> bI = pool_lookup(d_idx);
    NSUInteger oI = pool_lookup_offset(d_idx);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);
    dispatch1d(fw_pso(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bT offset:oT atIndex:0];
        [enc setBuffer:bI offset:oI atIndex:1];
        [enc setBuffer:bO offset:oO atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

void embedding_lookup_backward_gpu(const GpuTensor& dOut,
                                   const int32_t* d_idx, int B,
                                   GpuTensor& dTable) {
    const int D = dTable.cols;
    const int total = B * D;
    if (total == 0) return;
    id<MTLBuffer> bdO = buffer_for(dOut);
    NSUInteger odO = buffer_offset_for(dOut);
    id<MTLBuffer> bdT = buffer_for(dTable);
    NSUInteger odT = buffer_offset_for(dTable);
    id<MTLBuffer> bI  = pool_lookup(d_idx);
    NSUInteger oI = pool_lookup_offset(d_idx);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);
    dispatch1d(bw_pso(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bI  offset:oI atIndex:1];
        [enc setBuffer:bdT offset:odT atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

} // namespace brogameagent::nn::gpu
