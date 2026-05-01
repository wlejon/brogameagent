#pragma once

// Internal helpers shared across the Metal backend translation units.
// NOT installed; not part of the public API.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <brogameagent/nn/gpu/tensor.h>

#include <cstddef>

namespace brogameagent::nn::gpu::metal_impl {

// Process-singleton accessors. Lazily initialized by cuda_init() (yes —
// kept that name for source compatibility with consumers; on Metal it
// initializes the MTLDevice + command queue + executor).
id<MTLDevice>       device();
id<MTLCommandQueue> queue();

// MPSGraph executor reused for graph runs. Each op constructs its own
// transient MPSGraph; this executor is the bridge to MTLBuffer-backed
// MPSGraphTensorData feeds.
//
// Returns a fresh MPSCommandBuffer wrapping a new MTLCommandBuffer drawn
// from queue(). Callers are responsible for committing.
id<MTLCommandBuffer> new_command_buffer();

// Pool of MTLBuffers keyed by (data_ptr → buffer). GpuTensor stores only
// `float* data`; on Metal that pointer is the contents of an MTLBuffer
// allocated with MTLResourceStorageModeShared (unified memory). The pool
// owns the MTLBuffer reference for the lifetime of the GpuTensor.
//
// Internal — used exclusively by tensor.mm and device_buffer.mm.
void  pool_register(void* data_ptr, id<MTLBuffer> buf);
id<MTLBuffer> pool_lookup(const void* data_ptr);
NSUInteger pool_lookup_offset(const void* data_ptr);
void  pool_release(void* data_ptr);

// Convenience: get the MTLBuffer backing a GpuTensor's data pointer.
// Returns nil if the tensor is empty.
inline id<MTLBuffer> buffer_for(const GpuTensor& t) {
    if (t.data == nullptr) return nil;
    return pool_lookup(t.data);
}
inline NSUInteger buffer_offset_for(const GpuTensor& t) {
    if (t.data == nullptr) return 0;
    return pool_lookup_offset(t.data);
}

// Bind a GpuTensor (resolves base buffer + view offset) to a compute encoder
// at the given buffer index.
inline void set_tensor(id<MTLComputeCommandEncoder> enc,
                       const GpuTensor& t,
                       NSUInteger index) {
    [enc setBuffer:buffer_for(t) offset:buffer_offset_for(t) atIndex:index];
}

// Run a single-output MPSGraph synchronously: feed the named placeholders
// from MTLBuffers, write the result to result_buffer. Synchronous — caller
// does not need a separate cuda_sync.
void run_graph_sync(MPSGraph* g,
                    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                    MPSGraphTensor* result,
                    id<MTLBuffer> result_buffer,
                    NSArray<NSNumber*>* result_shape);
void run_graph_sync_off(MPSGraph* g,
                        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                        MPSGraphTensor* result,
                        id<MTLBuffer> result_buffer,
                        NSUInteger result_offset_bytes,
                        NSArray<NSNumber*>* result_shape);

// MTLBuffer wrapper for an existing host-shared region (no copy). Used to
// build MPSGraphTensorData for tensors pulled from the pool.
MPSGraphTensorData* tensor_data_for(id<MTLBuffer> buf,
                                    NSArray<NSNumber*>* shape,
                                    MPSDataType dt = MPSDataTypeFloat32);
MPSGraphTensorData* tensor_data_for_offset(id<MTLBuffer> buf,
                                           NSUInteger offset_bytes,
                                           NSArray<NSNumber*>* shape,
                                           MPSDataType dt = MPSDataTypeFloat32);

// Return the cached compute pipeline for a named MSL kernel (precompiled in
// kernels.mm at first cuda_init()). Throws if name is unknown.
id<MTLComputePipelineState> pipeline(NSString* name);

// Compile an MSL source string and build a pipeline for a named function.
// Used by per-op-file lazy compile patterns: each op .mm holds its own MSL
// source plus a `static dispatch_once_t once + id<MTLComputePipelineState> pso;`
// — first call compiles, subsequent calls reuse.
//
// Throws std::runtime_error on compile or pipeline-build failure.
id<MTLComputePipelineState> compile_pipeline(NSString* msl_source,
                                             NSString* function_name);

// Helper to dispatch an in-process MSL kernel against a 1-D thread grid
// covering n elements. Encodes onto a fresh command buffer and commits +
// waits. For best latency, several launches in a row should batch onto one
// command buffer manually rather than going through this helper.
void dispatch1d_sync(NSString* pipeline_name,
                     NSUInteger n,
                     void (^bind)(id<MTLComputeCommandEncoder>));

} // namespace brogameagent::nn::gpu::metal_impl
