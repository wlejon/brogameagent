#include <brogameagent/nn/gpu/tensor.h>
#include <brogameagent/nn/gpu/runtime.h>

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

} // namespace brogameagent::nn::gpu
