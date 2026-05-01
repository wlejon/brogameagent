#import "internal.h"

#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <string>

// All in-process MSL kernel sources. Compiled once at first cuda_init().
// Pipelines are cached in a name → MTLComputePipelineState map.
//
// We keep the source in one big string so newLibraryWithSource: only fires
// once; individual kernels are looked up by function name.

static NSString* const kKernelsMSL = @R"msl(
#include <metal_stdlib>
using namespace metal;

// ── elementwise ─────────────────────────────────────────────────────────

kernel void k_relu_forward(device const float* x [[buffer(0)]],
                           device float*       y [[buffer(1)]],
                           constant uint& n      [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i];
    y[i] = v > 0.0f ? v : 0.0f;
}

kernel void k_relu_backward(device const float* x  [[buffer(0)]],
                            device const float* dY [[buffer(1)]],
                            device float*       dX [[buffer(2)]],
                            constant uint& n       [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
}

kernel void k_tanh_forward(device const float* x [[buffer(0)]],
                           device float*       y [[buffer(1)]],
                           constant uint& n      [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = tanh(x[i]);
}

kernel void k_tanh_backward(device const float* y  [[buffer(0)]],
                            device const float* dY [[buffer(1)]],
                            device float*       dX [[buffer(2)]],
                            constant uint& n       [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float yv = y[i];
    dX[i] = dY[i] * (1.0f - yv * yv);
}

kernel void k_sigmoid_forward(device const float* x [[buffer(0)]],
                              device float*       y [[buffer(1)]],
                              constant uint& n      [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = 1.0f / (1.0f + exp(-x[i]));
}

kernel void k_sigmoid_backward(device const float* y  [[buffer(0)]],
                               device const float* dY [[buffer(1)]],
                               device float*       dX [[buffer(2)]],
                               constant uint& n       [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float yv = y[i];
    dX[i] = dY[i] * yv * (1.0f - yv);
}

kernel void k_add_inplace(device float*       y [[buffer(0)]],
                          device const float* x [[buffer(1)]],
                          constant uint& n      [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] += x[i];
}

kernel void k_add_scalar_inplace(device float* y    [[buffer(0)]],
                                 constant float& s  [[buffer(1)]],
                                 constant uint& n   [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] += s;
}

kernel void k_scale_inplace(device float* y    [[buffer(0)]],
                            constant float& s  [[buffer(1)]],
                            constant uint& n   [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] *= s;
}

// ── linear (small kernels for forward/backward; MPSGraph would also work) ──

// dW[i, j] += dY[i] * x[j]. Threads cover (i, j) with i = tid.y, j = tid.x.
kernel void k_linear_backward_dw(device const float* dY [[buffer(0)]],
                                 device const float* x  [[buffer(1)]],
                                 device float*       dW [[buffer(2)]],
                                 constant uint& out_dim [[buffer(3)]],
                                 constant uint& in_dim  [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y;
    if (i >= out_dim || j >= in_dim) return;
    dW[(ulong)i * in_dim + j] += dY[i] * x[j];
}

// dB[i] += dY[i]
kernel void k_linear_backward_db(device const float* dY [[buffer(0)]],
                                 device float*       dB [[buffer(1)]],
                                 constant uint& out_dim [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= out_dim) return;
    dB[i] += dY[i];
}

)msl";

namespace brogameagent::nn::gpu::metal_impl {

namespace {
struct PipelineCache {
    std::unordered_map<std::string, void*> map; // void* holds +1 retain
    id<MTLLibrary> lib = nil;
};
PipelineCache& cache() { static PipelineCache c; return c; }
} // namespace

id<MTLComputePipelineState> pipeline(NSString* name) {
    auto& C = cache();
    std::string key = [name UTF8String];
    auto it = C.map.find(key);
    if (it == C.map.end()) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "Metal: unknown kernel `%s` (not in pipeline cache)",
                      key.c_str());
        throw std::runtime_error(buf);
    }
    return (__bridge id<MTLComputePipelineState>)it->second;
}

id<MTLComputePipelineState> compile_pipeline(NSString* msl_source,
                                             NSString* function_name) {
    @autoreleasepool {
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> lib = [device() newLibraryWithSource:msl_source
                                                    options:opts
                                                      error:&err];
        if (!lib) {
            std::string m = "Metal: MSL compile failed for ";
            m += [function_name UTF8String];
            if (err) { m += ": "; m += [[err localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        id<MTLFunction> fn = [lib newFunctionWithName:function_name];
        if (!fn) {
            std::string m = "Metal: function not found: ";
            m += [function_name UTF8String];
            throw std::runtime_error(m);
        }
        NSError* perr = nil;
        id<MTLComputePipelineState> pso =
            [device() newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) {
            std::string m = "Metal: pipeline build failed for ";
            m += [function_name UTF8String];
            if (perr) { m += ": "; m += [[perr localizedDescription] UTF8String]; }
            throw std::runtime_error(m);
        }
        return pso;
    }
}

void dispatch1d_sync(NSString* pipeline_name,
                     NSUInteger n,
                     void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLComputePipelineState> pso = pipeline(pipeline_name);
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        MTLSize grid = MTLSizeMake(n, 1, 1);
        MTLSize tgsize = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgsize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brogameagent::nn::gpu::metal_impl

// Called once from cuda_init() (extern from runtime.mm).
extern "C++" void metal_build_pipelines_();
void metal_build_pipelines_() {
    using namespace brogameagent::nn::gpu::metal_impl;
    auto& C = cache();
    if (C.lib != nil) return;
    @autoreleasepool {
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> lib = [device() newLibraryWithSource:kKernelsMSL
                                                    options:opts
                                                      error:&err];
        if (!lib) {
            std::string m = "Metal: kernel library compile failed: ";
            if (err) m += [[err localizedDescription] UTF8String];
            throw std::runtime_error(m);
        }
        C.lib = lib;

        NSArray<NSString*>* names = @[
            @"k_relu_forward", @"k_relu_backward",
            @"k_tanh_forward", @"k_tanh_backward",
            @"k_sigmoid_forward", @"k_sigmoid_backward",
            @"k_add_inplace", @"k_add_scalar_inplace", @"k_scale_inplace",
            @"k_linear_backward_dw", @"k_linear_backward_db",
        ];
        for (NSString* n in names) {
            id<MTLFunction> fn = [lib newFunctionWithName:n];
            if (!fn) {
                std::string m = "Metal: kernel not found: ";
                m += [n UTF8String];
                throw std::runtime_error(m);
            }
            NSError* perr = nil;
            id<MTLComputePipelineState> pso =
                [device() newComputePipelineStateWithFunction:fn error:&perr];
            if (!pso) {
                std::string m = "Metal: pipeline build failed for ";
                m += [n UTF8String];
                if (perr) { m += ": "; m += [[perr localizedDescription] UTF8String]; }
                throw std::runtime_error(m);
            }
            C.map.emplace([n UTF8String], (__bridge_retained void*)pso);
        }
    }
}
