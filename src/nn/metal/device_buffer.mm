#include <brogameagent/nn/gpu/device_buffer.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

#include <cstring>
#include <stdexcept>

namespace brogameagent::nn::gpu {

// Backend hooks for the typed DeviceBuffer<T> template (defined in
// device_buffer.h). `native_out` carries the +1-retained MTLBuffer so the
// matching free can release it; on Metal we cannot reverse-lookup buffer
// from `[buf contents]` without registering, but DeviceBuffer is created in
// pairs so we just keep the buffer pointer in native_out.

void* device_alloc_bytes(std::size_t bytes, void** native_out) {
    if (native_out) *native_out = nullptr;
    if (bytes == 0) return nullptr;
    cuda_init();
    @autoreleasepool {
        id<MTLBuffer> buf = [metal_impl::device()
            newBufferWithLength:bytes
                        options:MTLResourceStorageModeShared];
        if (!buf) {
            throw std::runtime_error("Metal: device_alloc_bytes failed");
        }
        void* p = [buf contents];
        if (native_out) *native_out = (__bridge_retained void*)buf;
        // Also register in the GpuTensor pool so ops that look the buffer
        // up by pointer (e.g. when an ad-hoc DeviceBuffer<float> is passed
        // as a mask to softmax_forward_gpu) can find it.
        metal_impl::pool_register(p, buf);
        return p;
    }
}

void device_free_bytes(void* device_ptr, void* native) {
    if (!device_ptr) {
        if (native) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)native;
            (void)buf;
        }
        return;
    }
    metal_impl::pool_release(device_ptr);
    if (native) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)native;
        (void)buf;
    }
}

void device_upload_bytes(void* device_ptr, const void* host, std::size_t bytes) {
    if (bytes == 0) return;
    // Shared storage = unified memory; CPU write directly into the buffer.
    std::memcpy(device_ptr, host, bytes);
}

} // namespace brogameagent::nn::gpu
