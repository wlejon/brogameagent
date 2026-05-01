#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSGDSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void k_sgd_step(device float*       param    [[buffer(0)]],
                       device const float* grad     [[buffer(1)]],
                       device float*       velocity [[buffer(2)]],
                       constant float& lr           [[buffer(3)]],
                       constant float& momentum     [[buffer(4)]],
                       constant uint&  n            [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = momentum * velocity[i] + grad[i];
    velocity[i] = v;
    param[i]   -= lr * v;
}
)msl";

id<MTLComputePipelineState> sgd_pipeline() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kSGDSrc, @"k_sgd_step");
    });
    return pso;
}

} // namespace

void sgd_step_gpu(GpuTensor& param, GpuTensor& grad, GpuTensor& velocity,
                  float lr, float momentum) {
    const uint32_t n = static_cast<uint32_t>(param.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso = sgd_pipeline();
    id<MTLBuffer> bp = buffer_for(param);
    NSUInteger op = buffer_offset_for(param);
    id<MTLBuffer> bg = buffer_for(grad);
    NSUInteger og = buffer_offset_for(grad);
    id<MTLBuffer> bv = buffer_for(velocity);
    NSUInteger ov = buffer_offset_for(velocity);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBytes:&lr       length:sizeof(float)    atIndex:3];
        [enc setBytes:&momentum length:sizeof(float)    atIndex:4];
        [enc setBytes:&n        length:sizeof(uint32_t) atIndex:5];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brogameagent::nn::gpu
