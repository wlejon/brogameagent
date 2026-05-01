#pragma once

#include <cstddef>
#include <cstring>
#include <utility>

namespace brogameagent::nn::gpu {

// ─── Backend-neutral device buffer ─────────────────────────────────────────
//
// Tiny RAII wrapper over a typed device-resident buffer. Replaces the
// previous pattern of cudaMalloc/cudaMemcpy/cudaFree directly inlined into
// callers (tests, generic_trainer.cpp). Backend-specific allocation lives in
// device_alloc_bytes / device_free_bytes / device_upload_bytes — implemented
// by tensor.cu (CUDA) or device_buffer.mm (Metal).
//
// On CUDA: device_ptr() returns a true cudaMalloc'd pointer (host-invalid).
// On Metal (Apple Silicon): device_ptr() returns the contents of a shared
// MTLBuffer — same bytes are accessible from CPU and GPU. That fact is
// invisible at this level; callers should only use the returned pointer as
// a device pointer through nn::gpu::ops APIs.

void* device_alloc_bytes(std::size_t bytes, void** native_out);
void  device_free_bytes(void* device_ptr, void* native);
void  device_upload_bytes(void* device_ptr, const void* host, std::size_t bytes);

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release_(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : data_(o.data_), n_(o.n_), native_(o.native_) {
        o.data_ = nullptr; o.n_ = 0; o.native_ = nullptr;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release_();
            data_ = o.data_; n_ = o.n_; native_ = o.native_;
            o.data_ = nullptr; o.n_ = 0; o.native_ = nullptr;
        }
        return *this;
    }

    // Reallocates if size differs. Existing contents are not preserved.
    void resize(std::size_t n) {
        if (n == n_ && data_ != nullptr) return;
        release_();
        n_ = n;
        if (n == 0) return;
        data_ = static_cast<T*>(device_alloc_bytes(n * sizeof(T), &native_));
    }

    // Resize to n then copy host[0..n) into device.
    void upload(const T* host, std::size_t n) {
        resize(n);
        if (n == 0) return;
        device_upload_bytes(data_, host, n * sizeof(T));
    }

    T*       device_ptr()       { return data_; }
    const T* device_ptr() const { return data_; }
    std::size_t size() const    { return n_; }
    bool empty() const          { return n_ == 0; }

private:
    void release_() {
        if (data_) device_free_bytes(data_, native_);
        data_ = nullptr; n_ = 0; native_ = nullptr;
    }
    T*           data_   = nullptr;
    std::size_t  n_      = 0;
    void*        native_ = nullptr; // MTLBuffer on Metal; null on CUDA.
};

} // namespace brogameagent::nn::gpu
