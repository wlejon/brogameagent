#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace brogameagent::nn::gpu {

namespace {
std::atomic<bool> g_initialized{false};

int parse_device_from_env() {
    const char* v = std::getenv("BGA_CUDA_DEVICE");
    if (!v || !*v) return 0;
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (end == v) return 0;
    if (n < 0) return 0;
    return static_cast<int>(n);
}
} // namespace

void cuda_init() {
    if (g_initialized.load(std::memory_order_acquire)) return;

    int count = 0;
    BGA_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count <= 0) {
        throw std::runtime_error("brogameagent::nn::gpu::cuda_init: no CUDA devices found");
    }
    int dev = parse_device_from_env();
    if (dev >= count) dev = 0;
    BGA_CUDA_CHECK(cudaSetDevice(dev));

    // Touch the device to force lazy context creation up-front.
    BGA_CUDA_CHECK(cudaFree(nullptr));

    g_initialized.store(true, std::memory_order_release);
}

void cuda_sync() {
    BGA_CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_check_throw(int err, const char* expr_text, const char* file, int line) {
    if (err == 0) return;
    const char* es = cudaGetErrorString(static_cast<cudaError_t>(err));
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "CUDA error %d (%s) at %s:%d in `%s`",
                  err, es ? es : "?", file ? file : "?", line,
                  expr_text ? expr_text : "?");
    throw std::runtime_error(buf);
}

} // namespace brogameagent::nn::gpu
