#include <brogameagent/nn/gpu/tensor.h>
#include <brogameagent/nn/gpu/runtime.h>
#include <brogameagent/nn/gpu/device_buffer.h>

#import "internal.h"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace brogameagent::nn::gpu {

// ─── Pointer → MTLBuffer pool ──────────────────────────────────────────────
//
// GpuTensor's public API is just `float* data + rows + cols`. On Metal we
// allocate MTLBuffers with shared storage (unified memory on Apple Silicon)
// and use `[buf contents]` as the `data` pointer. To free the buffer when
// the GpuTensor is destroyed, we keep a global pointer→buffer map.
//
// The pool is process-wide and protected by a mutex. Hot-path ops do not
// hit this map; only allocation, resize, and destruction do.

namespace metal_impl {

namespace {
struct Pool {
    std::mutex mu;
    std::unordered_map<const void*, void*> map; // void* holds an Obj-C +1 retain
};
Pool& pool() { static Pool p; return p; }
} // namespace

void pool_register(void* data_ptr, id<MTLBuffer> buf) {
    if (!data_ptr || !buf) return;
    auto& P = pool();
    void* retained = (__bridge_retained void*)buf;
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) {
        // Replacing — release the prior retain.
        id<MTLBuffer> prior = (__bridge_transfer id<MTLBuffer>)it->second;
        (void)prior;
        it->second = retained;
    } else {
        P.map.emplace(data_ptr, retained);
    }
}

id<MTLBuffer> pool_lookup(const void* data_ptr) {
    if (!data_ptr) return nil;
    auto& P = pool();
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) {
        return (__bridge id<MTLBuffer>)it->second;
    }
    // Fallback: containment search — handles GpuTensor::view() pointers that
    // alias into the middle of an existing MTLBuffer.
    const auto* p = static_cast<const char*>(data_ptr);
    for (const auto& kv : P.map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const char* base = static_cast<const char*>([buf contents]);
        const char* end  = base + [buf length];
        if (p >= base && p < end) {
            return buf;
        }
    }
    return nil;
}

NSUInteger pool_lookup_offset(const void* data_ptr) {
    if (!data_ptr) return 0;
    auto& P = pool();
    std::lock_guard<std::mutex> lk(P.mu);
    auto it = P.map.find(data_ptr);
    if (it != P.map.end()) return 0;
    const auto* p = static_cast<const char*>(data_ptr);
    for (const auto& kv : P.map) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)kv.second;
        const char* base = static_cast<const char*>([buf contents]);
        const char* end  = base + [buf length];
        if (p >= base && p < end) {
            return static_cast<NSUInteger>(p - base);
        }
    }
    return 0;
}

void pool_release(void* data_ptr) {
    if (!data_ptr) return;
    auto& P = pool();
    void* retained = nullptr;
    {
        std::lock_guard<std::mutex> lk(P.mu);
        auto it = P.map.find(data_ptr);
        if (it == P.map.end()) return;
        retained = it->second;
        P.map.erase(it);
    }
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)retained;
    (void)buf; // released by ARC at scope end
}

} // namespace metal_impl

namespace {

float* allocate_shared(std::size_t nfloats) {
    if (nfloats == 0) return nullptr;
    cuda_init();
    @autoreleasepool {
        id<MTLBuffer> buf = [metal_impl::device()
            newBufferWithLength:nfloats * sizeof(float)
                        options:MTLResourceStorageModeShared];
        if (!buf) {
            throw std::runtime_error("Metal: failed to allocate MTLBuffer");
        }
        float* p = static_cast<float*>([buf contents]);
        metal_impl::pool_register(p, buf);
        return p;
    }
}

} // namespace

GpuTensor::GpuTensor(int r, int c) : data(nullptr), rows(r), cols(c), owns_(false) {
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    if (n == 0) return;
    data = allocate_shared(n);
    owns_ = true;
}

GpuTensor::~GpuTensor() {
    release_();
}

void GpuTensor::release_() {
    if (owns_ && data) {
        metal_impl::pool_release(data);
    }
    data = nullptr;
    rows = 0;
    cols = 0;
    owns_ = false;
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : data(other.data), rows(other.rows), cols(other.cols), owns_(other.owns_) {
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    other.owns_ = false;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        release_();
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        owns_ = other.owns_;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.owns_ = false;
    }
    return *this;
}

void GpuTensor::zero() {
    if (size() == 0) return;
    std::memset(data, 0, static_cast<size_t>(size()) * sizeof(float));
}

void GpuTensor::resize(int r, int c) {
    if (r == rows && c == cols && data != nullptr) return;
    release_();
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    rows = r;
    cols = c;
    if (n == 0) return;
    data = allocate_shared(n);
    owns_ = true;
}

GpuTensor GpuTensor::clone() const {
    GpuTensor out;
    if (size() == 0) {
        out.rows = rows;
        out.cols = cols;
        return out;
    }
    out.resize(rows, cols);
    std::memcpy(out.data, data,
                static_cast<size_t>(size()) * sizeof(float));
    return out;
}

GpuTensor GpuTensor::view(float* data, int rows, int cols) {
    GpuTensor t;
    t.data = data;
    t.rows = rows;
    t.cols = cols;
    t.owns_ = false;
    return t;
}

void upload(const Tensor& src, GpuTensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        dst.resize(src.rows, src.cols);
    }
    if (src.size() == 0) return;
    std::memcpy(dst.data, src.data.data(),
                static_cast<size_t>(src.size()) * sizeof(float));
}

void download(const GpuTensor& src, Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        dst.resize(src.rows, src.cols);
    }
    if (src.size() == 0) return;
    std::memcpy(dst.data.data(), src.data,
                static_cast<size_t>(src.size()) * sizeof(float));
}

} // namespace brogameagent::nn::gpu
