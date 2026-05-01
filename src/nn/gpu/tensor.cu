#include <brogameagent/nn/gpu/tensor.h>
#include <brogameagent/nn/gpu/runtime.h>
#include <brogameagent/nn/gpu/device_buffer.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace brogameagent::nn::gpu {

GpuTensor::GpuTensor(int r, int c) : data(nullptr), rows(r), cols(c), owns_(false) {
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    if (n == 0) return;
    cuda_init();
    void* p = nullptr;
    BGA_CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    data = static_cast<float*>(p);
    owns_ = true;
}

GpuTensor::~GpuTensor() {
    release_();
}

void GpuTensor::release_() {
    if (owns_ && data) {
        // Don't throw from a destructor — swallow the error code if any.
        cudaFree(data);
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
    BGA_CUDA_CHECK(cudaMemset(data, 0, static_cast<size_t>(size()) * sizeof(float)));
}

void GpuTensor::resize(int r, int c) {
    if (r == rows && c == cols && data != nullptr) return;
    release_();
    const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
    rows = r;
    cols = c;
    if (n == 0) return;
    cuda_init();
    void* p = nullptr;
    BGA_CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    data = static_cast<float*>(p);
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
    BGA_CUDA_CHECK(cudaMemcpy(out.data, data,
                              static_cast<size_t>(size()) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
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
    BGA_CUDA_CHECK(cudaMemcpy(dst.data, src.data.data(),
                              static_cast<size_t>(src.size()) * sizeof(float),
                              cudaMemcpyHostToDevice));
}

void download(const GpuTensor& src, Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        dst.resize(src.rows, src.cols);
    }
    if (src.size() == 0) return;
    BGA_CUDA_CHECK(cudaMemcpy(dst.data.data(), src.data,
                              static_cast<size_t>(src.size()) * sizeof(float),
                              cudaMemcpyDeviceToHost));
}

// ─── DeviceBuffer<T> backend hooks (CUDA) ──────────────────────────────────

void* device_alloc_bytes(std::size_t bytes, void** native_out) {
    if (native_out) *native_out = nullptr; // unused on CUDA
    if (bytes == 0) return nullptr;
    cuda_init();
    void* p = nullptr;
    BGA_CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}

void device_free_bytes(void* device_ptr, void* /*native*/) {
    if (device_ptr) cudaFree(device_ptr);
}

void device_upload_bytes(void* device_ptr, const void* host, std::size_t bytes) {
    if (bytes == 0) return;
    BGA_CUDA_CHECK(cudaMemcpy(device_ptr, host, bytes, cudaMemcpyHostToDevice));
}

} // namespace brogameagent::nn::gpu
