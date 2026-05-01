#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

#include <cmath>

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kAdamSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void k_adam_step(device float*       param   [[buffer(0)]],
                        device const float* grad    [[buffer(1)]],
                        device float*       m       [[buffer(2)]],
                        device float*       v       [[buffer(3)]],
                        constant float& lr          [[buffer(4)]],
                        constant float& beta1       [[buffer(5)]],
                        constant float& beta2       [[buffer(6)]],
                        constant float& eps         [[buffer(7)]],
                        constant float& inv_bc1     [[buffer(8)]],
                        constant float& inv_bc2     [[buffer(9)]],
                        constant uint&  n           [[buffer(10)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi * inv_bc1;
    float v_hat = vi * inv_bc2;
    param[i] -= lr * m_hat / (sqrt(v_hat) + eps);
}
)msl";

id<MTLComputePipelineState> adam_pipeline() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kAdamSrc, @"k_adam_step");
    });
    return pso;
}

} // namespace

void adam_step_gpu(GpuTensor& param, const GpuTensor& grad,
                   GpuTensor& m, GpuTensor& v,
                   float lr, float beta1, float beta2, float eps, int step) {
    const uint32_t n = static_cast<uint32_t>(param.size());
    if (n == 0) return;
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;
    id<MTLComputePipelineState> pso = adam_pipeline();
    id<MTLBuffer> bp = buffer_for(param);
    NSUInteger op = buffer_offset_for(param);
    id<MTLBuffer> bg = buffer_for(grad);
    NSUInteger og = buffer_offset_for(grad);
    id<MTLBuffer> bm = buffer_for(m);
    NSUInteger om = buffer_offset_for(m);
    id<MTLBuffer> bv = buffer_for(v);
    NSUInteger ov = buffer_offset_for(v);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bp offset:op atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bm offset:om atIndex:2];
        [enc setBuffer:bv offset:ov atIndex:3];
        [enc setBytes:&lr      length:sizeof(float)    atIndex:4];
        [enc setBytes:&beta1   length:sizeof(float)    atIndex:5];
        [enc setBytes:&beta2   length:sizeof(float)    atIndex:6];
        [enc setBytes:&eps     length:sizeof(float)    atIndex:7];
        [enc setBytes:&inv_bc1 length:sizeof(float)    atIndex:8];
        [enc setBytes:&inv_bc2 length:sizeof(float)    atIndex:9];
        [enc setBytes:&n       length:sizeof(uint32_t) atIndex:10];
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
