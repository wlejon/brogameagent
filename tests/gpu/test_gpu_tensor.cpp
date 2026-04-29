#include <brogameagent/nn/gpu/runtime.h>
#include <brogameagent/nn/gpu/tensor.h>
#include <brogameagent/nn/tensor.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace brogameagent::nn;
using namespace brogameagent::nn::gpu;

// Lightweight inline harness — matches the style of tests/test_main.cpp but
// kept self-contained so the GPU smoke test isolates a single concern.

static int g_failed = 0;

static void check(bool cond, const char* msg, int line) {
    if (!cond) {
        std::printf("    assertion failed at line %d: %s\n", line, msg);
        ++g_failed;
        throw 0;
    }
}

#define CHECK(cond) check(cond, #cond, __LINE__)
#define CHECK_NEAR(a, b, eps) check(std::abs((a) - (b)) < (eps), #a " ~= " #b, __LINE__)

static void test_upload_download_roundtrip() {
    Tensor h(3, 4);
    for (int i = 0; i < h.size(); ++i) h.data[i] = static_cast<float>(i) * 0.5f - 1.0f;

    GpuTensor g;
    upload(h, g);
    cuda_sync();
    CHECK(g.rows == 3);
    CHECK(g.cols == 4);
    CHECK(g.data != nullptr);

    Tensor back;
    download(g, back);
    cuda_sync();
    CHECK(back.rows == 3);
    CHECK(back.cols == 4);
    for (int i = 0; i < h.size(); ++i) {
        CHECK_NEAR(back.data[i], h.data[i], 1e-6f);
    }
}

static void test_clone() {
    Tensor h(2, 3);
    for (int i = 0; i < h.size(); ++i) h.data[i] = static_cast<float>(i + 1);

    GpuTensor a;
    upload(h, a);
    GpuTensor b = a.clone();
    CHECK(b.rows == 2);
    CHECK(b.cols == 3);
    CHECK(b.data != nullptr);
    CHECK(b.data != a.data); // distinct allocation

    // Mutate the source on host then re-upload so we can prove the clone is
    // independent.
    Tensor h2(2, 3);
    for (int i = 0; i < h2.size(); ++i) h2.data[i] = -42.0f;
    upload(h2, a);

    Tensor downB;
    download(b, downB);
    for (int i = 0; i < downB.size(); ++i) {
        CHECK_NEAR(downB.data[i], static_cast<float>(i + 1), 1e-6f);
    }
}

static void test_resize_and_zero() {
    GpuTensor g(4, 5);
    CHECK(g.rows == 4);
    CHECK(g.cols == 5);
    CHECK(g.size() == 20);

    g.resize(2, 3);
    CHECK(g.rows == 2);
    CHECK(g.cols == 3);
    CHECK(g.size() == 6);

    g.zero();
    Tensor h;
    download(g, h);
    for (int i = 0; i < h.size(); ++i) CHECK_NEAR(h.data[i], 0.0f, 1e-6f);

    // Resize to same shape is a no-op (data pointer should remain valid).
    float* prev = g.data;
    g.resize(2, 3);
    CHECK(g.data == prev);
}

static void test_view_is_non_owning() {
    GpuTensor owner(3, 3);
    owner.zero();
    {
        GpuTensor v = GpuTensor::view(owner.data, 3, 3);
        CHECK(v.data == owner.data);
        // v goes out of scope here without freeing owner.data.
    }
    // Sanity: owner still usable.
    owner.zero();
}

int main() {
    std::printf("brogameagent gpu tensor smoke test\n");
    std::printf("==================================\n");

    cuda_init();

    struct Entry { const char* name; void (*fn)(); };
    Entry tests[] = {
        {"upload_download_roundtrip", test_upload_download_roundtrip},
        {"clone",                     test_clone},
        {"resize_and_zero",           test_resize_and_zero},
        {"view_is_non_owning",        test_view_is_non_owning},
    };

    int passed = 0;
    int total = static_cast<int>(sizeof(tests) / sizeof(tests[0]));
    for (const auto& t : tests) {
        try {
            t.fn();
            ++passed;
            std::printf("  PASS  %s\n", t.name);
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s — %s\n", t.name, e.what());
        } catch (...) {
            std::printf("  FAIL  %s\n", t.name);
        }
    }

    std::printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
