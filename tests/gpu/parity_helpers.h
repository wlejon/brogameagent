#pragma once

// Shared helpers for GPU↔CPU parity tests.
//
// Header-only — each test executable includes this and gets its own copy of
// the inline helpers. Keeps the build wiring trivial (no extra .cpp).

#include <brogameagent/nn/gpu/runtime.h>
#include <brogameagent/nn/gpu/tensor.h>
#include <brogameagent/nn/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace bga_parity {

using brogameagent::nn::Tensor;
using brogameagent::nn::gpu::GpuTensor;
using brogameagent::nn::gpu::cuda_sync;
using brogameagent::nn::gpu::download;
using brogameagent::nn::gpu::upload;

// ─── Test registry ─────────────────────────────────────────────────────────

struct TestEntry {
    const char* name;
    void (*fn)();
};

inline std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

#define BGA_PARITY_TEST(name)                                                  \
    static void name();                                                        \
    namespace {                                                                \
    struct Reg_##name {                                                        \
        Reg_##name() { ::bga_parity::registry().push_back({#name, name}); }    \
    } reg_##name;                                                              \
    }                                                                          \
    static void name()

inline void check(bool cond, const char* msg, const char* file, int line) {
    if (!cond) {
        std::printf("    assertion failed at %s:%d: %s\n", file, line, msg);
        throw 0;
    }
}

#define BGA_CHECK(cond) ::bga_parity::check((cond), #cond, __FILE__, __LINE__)

// ─── Deterministic RNG (splitmix64 → uniform float in [-1, 1]) ─────────────

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_f01() {
        // Map top 24 bits to [0, 1).
        return static_cast<float>(next_u64() >> 40) / 16777216.0f;
    }
    float next_unit() { return next_f01() * 2.0f - 1.0f; }
};

inline void fill_random(Tensor& t, SplitMix64& rng, float scale = 1.0f) {
    for (int i = 0; i < t.size(); ++i) t.data[i] = rng.next_unit() * scale;
}

// ─── Tolerance comparison ─────────────────────────────────────────────────

inline void compare_tensors(const Tensor& cpu, const Tensor& gpu,
                            const char* tag,
                            float atol = 1e-5f, float rtol = 1e-4f) {
    if (cpu.rows != gpu.rows || cpu.cols != gpu.cols) {
        std::printf("    [%s] shape mismatch: cpu (%d,%d) vs gpu (%d,%d)\n",
                    tag, cpu.rows, cpu.cols, gpu.rows, gpu.cols);
        throw 0;
    }
    const int n = cpu.size();
    int worst_idx = -1;
    float worst_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = cpu.data[i];
        const float b = gpu.data[i];
        const float d = std::fabs(a - b);
        const float tol = atol + rtol * std::fabs(a);
        if (d > tol) {
            if (d > worst_diff) { worst_diff = d; worst_idx = i; }
        }
    }
    if (worst_idx >= 0) {
        std::printf("    [%s] mismatch at i=%d  cpu=%.7g gpu=%.7g  diff=%.3g\n",
                    tag, worst_idx,
                    cpu.data[worst_idx], gpu.data[worst_idx], worst_diff);
        throw 0;
    }
}

inline Tensor download_to_host(const GpuTensor& g) {
    Tensor h;
    download(g, h);
    cuda_sync();
    return h;
}

// ─── Test runner ──────────────────────────────────────────────────────────

inline int run_all(const char* banner) {
    std::printf("%s\n", banner);
    for (size_t i = 0; i < std::strlen(banner); ++i) std::putchar('=');
    std::putchar('\n');

    brogameagent::nn::gpu::cuda_init();

    int passed = 0;
    int total = static_cast<int>(registry().size());
    for (const auto& t : registry()) {
        try {
            t.fn();
            ++passed;
            std::printf("  PASS  %s\n", t.name);
        } catch (...) {
            std::printf("  FAIL  %s\n", t.name);
        }
    }
    std::printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

} // namespace bga_parity
