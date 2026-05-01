#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Declared at global scope; implemented in kernels.mm.
extern void metal_build_pipelines_();

namespace brogameagent::nn::gpu {

namespace {
std::atomic<bool> g_initialized{false};

id<MTLDevice>       g_device = nil;
id<MTLCommandQueue> g_queue  = nil;

void init_runtime_once() {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error(
                "brogameagent::nn::gpu::cuda_init: no Metal device found");
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            throw std::runtime_error(
                "brogameagent::nn::gpu::cuda_init: failed to create MTLCommandQueue");
        }
    }
}
} // namespace

void cuda_init() {
    if (g_initialized.load(std::memory_order_acquire)) return;
    static std::atomic<bool> in_progress{false};
    bool expected = false;
    if (!in_progress.compare_exchange_strong(expected, true)) {
        while (!g_initialized.load(std::memory_order_acquire)) {}
        return;
    }
    init_runtime_once();
    ::metal_build_pipelines_();
    g_initialized.store(true, std::memory_order_release);
    in_progress.store(false, std::memory_order_release);
}

void cuda_sync() {
    // On Metal there's no global "device synchronize" the way CUDA has it.
    // Each op submits its own command buffer and waits on it before
    // returning, so by the time control reaches here the GPU is already
    // caught up with everything we've issued. Implementing as a no-op is
    // correct for the synchronous-execution model; if we ever issue work
    // asynchronously we'd flush here by waiting on a sentinel buffer.
}

void cuda_check_throw(int err, const char* expr_text, const char* file, int line) {
    if (err == 0) return;
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "Metal/runtime error %d at %s:%d in `%s`",
                  err, file ? file : "?", line,
                  expr_text ? expr_text : "?");
    throw std::runtime_error(buf);
}

namespace metal_impl {

id<MTLDevice> device() {
    // No reentry into cuda_init() here: device()/queue() are called from
    // metal_build_pipelines_() which itself runs *during* cuda_init, when
    // g_initialized is still false. init_runtime_once() has already set
    // g_device by the time we reach this point, so a direct return is safe.
    if (!g_device) cuda_init();
    return g_device;
}

id<MTLCommandQueue> queue() {
    if (!g_queue) cuda_init();
    return g_queue;
}

id<MTLCommandBuffer> new_command_buffer() {
    return [queue() commandBuffer];
}

void run_graph_sync_off(MPSGraph* g,
                        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                        MPSGraphTensor* result,
                        id<MTLBuffer> result_buffer,
                        NSUInteger result_offset_bytes,
                        NSArray<NSNumber*>* result_shape) {
    @autoreleasepool {
        MPSGraphTensorData* result_td = tensor_data_for_offset(
            result_buffer, result_offset_bytes, result_shape, MPSDataTypeFloat32);
        MPSCommandBuffer* mps_cmd =
            [MPSCommandBuffer commandBufferFromCommandQueue:queue()];
        NSDictionary* results = @{ result : result_td };
        [g encodeToCommandBuffer:mps_cmd
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:results
             executionDescriptor:nil];
        [mps_cmd commit];
        [mps_cmd waitUntilCompleted];
    }
}

void run_graph_sync(MPSGraph* g,
                    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                    MPSGraphTensor* result,
                    id<MTLBuffer> result_buffer,
                    NSArray<NSNumber*>* result_shape) {
    run_graph_sync_off(g, feeds, result, result_buffer, 0, result_shape);
}

MPSGraphTensorData* tensor_data_for(id<MTLBuffer> buf,
                                    NSArray<NSNumber*>* shape,
                                    MPSDataType dt) {
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                   shape:shape
                                                dataType:dt];
}

MPSGraphTensorData* tensor_data_for_offset(id<MTLBuffer> buf,
                                           NSUInteger offset_bytes,
                                           NSArray<NSNumber*>* shape,
                                           MPSDataType dt) {
    if (offset_bytes == 0) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                       shape:shape
                                                    dataType:dt];
    }
    MPSShape* mpsShape = shape;
    MPSNDArrayDescriptor* desc =
        [MPSNDArrayDescriptor descriptorWithDataType:dt shape:mpsShape];
    MPSNDArray* nd = [[MPSNDArray alloc] initWithBuffer:buf
                                                 offset:offset_bytes
                                             descriptor:desc];
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:nd];
}

} // namespace metal_impl

} // namespace brogameagent::nn::gpu
