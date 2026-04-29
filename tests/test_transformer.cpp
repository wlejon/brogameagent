// CPU correctness tests for MultiHeadAttention, FeedForward,
// TransformerBlock, TransformerEncoder. Finite-difference gradient checks,
// parameter-count assertions, save/load round-trip, and an end-to-end
// encoder smoke test.

#include <brogameagent/nn/feedforward.h>
#include <brogameagent/nn/multi_head_attention.h>
#include <brogameagent/nn/transformer_block.h>
#include <brogameagent/nn/transformer_encoder.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace brogameagent::nn;

namespace {

struct TestEntry {
    const char* name;
    void (*fn)();
};
std::vector<TestEntry>& registry() { static std::vector<TestEntry> r; return r; }

#define TEST(name) \
    static void name(); \
    namespace { struct Reg_##name { Reg_##name() { registry().push_back({#name, name}); } } reg_##name; } \
    static void name()

inline void check(bool cond, const char* msg, const char* file, int line) {
    if (!cond) {
        std::printf("    assertion failed at %s:%d: %s\n", file, line, msg);
        throw 0;
    }
}
#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

// SplitMix64 RNG (mirror of test_main.cpp's pattern).
struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_unit() {
        return static_cast<float>(next_u64() >> 40) / 16777216.0f * 2.0f - 1.0f;
    }
};

void fill_random(Tensor& t, SplitMix64& rng, float scale = 1.0f) {
    for (int i = 0; i < t.size(); ++i) t.data[i] = rng.next_unit() * scale;
}

// Compute scalar loss = sum(O * upstream) so dL/dO = upstream.
float dot_loss(const Tensor& O, const Tensor& upstream) {
    float s = 0.0f;
    for (int i = 0; i < O.size(); ++i) s += O.data[i] * upstream.data[i];
    return s;
}

// Finite-difference one-element gradient check on a scalar -> param.
// Calls forward_loss(perturbed_param_value) twice per element.
template <typename ForwardLoss>
void check_grad_param(Tensor& param, const Tensor& analytic_grad,
                      ForwardLoss forward_loss, const char* tag,
                      float h = 1e-3f, float atol = 5e-3f, float rtol = 5e-3f,
                      int max_idx = 8) {
    int n = param.size();
    int step = (n + max_idx - 1) / max_idx;
    if (step < 1) step = 1;
    for (int i = 0; i < n; i += step) {
        const float orig = param.data[i];
        param.data[i] = orig + h;
        float l_plus = forward_loss();
        param.data[i] = orig - h;
        float l_minus = forward_loss();
        param.data[i] = orig;
        const float fd = (l_plus - l_minus) / (2.0f * h);
        const float an = analytic_grad.data[i];
        const float diff = std::fabs(fd - an);
        const float tol = atol + rtol * std::fabs(an);
        if (diff > tol) {
            std::printf("    [%s] grad mismatch at i=%d  fd=%.6g an=%.6g diff=%.3g\n",
                        tag, i, fd, an, diff);
            throw 0;
        }
    }
}

template <typename ForwardLoss>
void check_grad_input(Tensor& X, const Tensor& analytic_grad,
                      ForwardLoss forward_loss, const char* tag,
                      float h = 1e-3f, float atol = 5e-3f, float rtol = 5e-3f,
                      int max_idx = 8) {
    int n = X.size();
    int step = (n + max_idx - 1) / max_idx;
    if (step < 1) step = 1;
    for (int i = 0; i < n; i += step) {
        const float orig = X.data[i];
        X.data[i] = orig + h;
        float l_plus = forward_loss();
        X.data[i] = orig - h;
        float l_minus = forward_loss();
        X.data[i] = orig;
        const float fd = (l_plus - l_minus) / (2.0f * h);
        const float an = analytic_grad.data[i];
        const float diff = std::fabs(fd - an);
        const float tol = atol + rtol * std::fabs(an);
        if (diff > tol) {
            std::printf("    [%s] dX mismatch at i=%d  fd=%.6g an=%.6g diff=%.3g\n",
                        tag, i, fd, an, diff);
            throw 0;
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

TEST(mha_param_count) {
    MultiHeadAttention mha;
    uint64_t s = 0x123ull;
    mha.init(4, 8, 2, s);
    // 4 * D * D = 4 * 64 = 256.
    CHECK(mha.num_params() == 4 * 8 * 8);
    CHECK(mha.num_heads() == 2);
    CHECK(mha.head_dim() == 4);
}

TEST(mha_forward_invalid_rows_zero) {
    MultiHeadAttention mha;
    uint64_t s = 0x222ull;
    mha.init(4, 8, 2, s);
    SplitMix64 rng(0xAA);
    Tensor X(4, 8); fill_random(X, rng);
    std::vector<float> mask = {1, 0, 1, 0};
    Tensor O(4, 8);
    mha.forward(X, mask.data(), O);
    for (int i = 0; i < 4; ++i) {
        if (mask[i] == 0.0f) {
            for (int c = 0; c < 8; ++c) CHECK(O(i, c) == 0.0f);
        }
    }
}

TEST(mha_grad_check_unmasked) {
    SplitMix64 rng(0x401ull);
    const int K = 3, D = 8, H = 2;
    MultiHeadAttention mha;
    uint64_t s = 0x999ull;
    mha.init(K, D, H, s);
    Tensor X(K, D), upstream(K, D), O(K, D);
    fill_random(X, rng);
    fill_random(upstream, rng);

    auto fwd_loss = [&]() {
        Tensor Otmp(K, D);
        mha.forward(X, nullptr, Otmp);
        return dot_loss(Otmp, upstream);
    };

    mha.zero_grad();
    mha.forward(X, nullptr, O);
    Tensor dX(K, D);
    mha.backward(upstream, dX);

    check_grad_input(X, dX, fwd_loss, "mha.dX");
    check_grad_param(mha.Wq(), mha.dWq(), fwd_loss, "mha.dWq");
    check_grad_param(mha.Wk(), mha.dWk(), fwd_loss, "mha.dWk");
    check_grad_param(mha.Wv(), mha.dWv(), fwd_loss, "mha.dWv");
    check_grad_param(mha.Wo(), mha.dWo(), fwd_loss, "mha.dWo");
}

TEST(mha_grad_check_masked) {
    SplitMix64 rng(0x402ull);
    const int K = 4, D = 8, H = 2;
    MultiHeadAttention mha;
    uint64_t s = 0xABCull;
    mha.init(K, D, H, s);
    Tensor X(K, D), upstream(K, D), O(K, D);
    fill_random(X, rng);
    fill_random(upstream, rng);
    std::vector<float> mask = {1, 1, 0, 1};

    auto fwd_loss = [&]() {
        Tensor Otmp(K, D);
        mha.forward(X, mask.data(), Otmp);
        return dot_loss(Otmp, upstream);
    };

    mha.zero_grad();
    mha.forward(X, mask.data(), O);
    Tensor dX(K, D);
    mha.backward(upstream, dX);

    check_grad_param(mha.Wq(), mha.dWq(), fwd_loss, "mha.dWq.mask");
    check_grad_param(mha.Wo(), mha.dWo(), fwd_loss, "mha.dWo.mask");
    // dX rows for invalid queries: forward path didn't read those rows
    // (their projections are computed but their attention is zero AND output
    // is zero, so dL/dX_invalid = 0 too — verified via FD).
    check_grad_input(X, dX, fwd_loss, "mha.dX.mask");
}

TEST(mha_save_load_roundtrip) {
    SplitMix64 rng(0x501ull);
    MultiHeadAttention a;
    uint64_t s = 0x101ull;
    a.init(3, 8, 2, s);
    Tensor X(3, 8); fill_random(X, rng);
    Tensor Oa(3, 8); a.forward(X, nullptr, Oa);

    std::vector<uint8_t> buf;
    a.save_to(buf);
    MultiHeadAttention b;
    uint64_t s2 = 0x202ull;
    b.init(3, 8, 2, s2);  // different rng → different weights
    size_t off = 0;
    b.load_from(buf.data(), off, buf.size());
    Tensor Ob(3, 8); b.forward(X, nullptr, Ob);
    for (int i = 0; i < Oa.size(); ++i) CHECK(std::fabs(Oa.data[i] - Ob.data[i]) < 1e-6f);
}

TEST(ff_param_count) {
    FeedForward ff;
    uint64_t s = 0x1ull;
    ff.init(8, 16, s);
    CHECK(ff.num_params() == 8 * 16 + 16 + 16 * 8 + 8);
}

TEST(ff_grad_check) {
    SplitMix64 rng(0x601ull);
    const int K = 4, D = 8, DF = 12;
    FeedForward ff;
    uint64_t s = 0xFEull;
    ff.init(D, DF, s);
    Tensor X(K, D), upstream(K, D), O(K, D);
    fill_random(X, rng);
    fill_random(upstream, rng);

    auto fwd_loss = [&]() {
        Tensor Otmp(K, D);
        ff.forward(X, Otmp);
        return dot_loss(Otmp, upstream);
    };

    ff.zero_grad();
    ff.forward(X, O);
    Tensor dX(K, D);
    ff.backward(upstream, dX);

    check_grad_input(X, dX, fwd_loss, "ff.dX");
    check_grad_param(ff.W1(), ff.dW1(), fwd_loss, "ff.dW1");
    check_grad_param(ff.b1(), ff.dB1(), fwd_loss, "ff.dB1");
    check_grad_param(ff.W2(), ff.dW2(), fwd_loss, "ff.dW2");
    check_grad_param(ff.b2(), ff.dB2(), fwd_loss, "ff.dB2");
}

TEST(ff_save_load_roundtrip) {
    SplitMix64 rng(0x701ull);
    FeedForward a;
    uint64_t s = 0x10ull;
    a.init(8, 12, s);
    Tensor X(3, 8); fill_random(X, rng);
    Tensor Oa(3, 8); a.forward(X, Oa);
    std::vector<uint8_t> buf; a.save_to(buf);
    FeedForward b;
    uint64_t s2 = 0x20ull;
    b.init(8, 12, s2);
    size_t off = 0;
    b.load_from(buf.data(), off, buf.size());
    Tensor Ob(3, 8); b.forward(X, Ob);
    for (int i = 0; i < Oa.size(); ++i) CHECK(std::fabs(Oa.data[i] - Ob.data[i]) < 1e-6f);
}

TEST(transformer_block_pre_norm_grad_check) {
    SplitMix64 rng(0x801ull);
    const int K = 3, D = 8, H = 2, DF = 16;
    TransformerBlock blk;
    TransformerBlock::Config cfg{};
    cfg.dim = D; cfg.num_heads = H; cfg.d_ff = DF; cfg.n_slots = K;
    cfg.norm = NormPlacement::PreNorm;
    uint64_t s = 0x55ull; blk.init(cfg, s);

    Tensor X(K, D), upstream(K, D), Y(K, D);
    fill_random(X, rng);
    fill_random(upstream, rng);

    auto fwd_loss = [&]() {
        Tensor Ytmp(K, D);
        blk.forward(X, nullptr, Ytmp);
        return dot_loss(Ytmp, upstream);
    };

    blk.zero_grad();
    blk.forward(X, nullptr, Y);
    Tensor dX(K, D);
    blk.backward(upstream, dX);
    check_grad_input(X, dX, fwd_loss, "tblk.pre.dX", 1e-3f, 1e-2f, 1e-2f, 8);
}

TEST(transformer_block_post_norm_grad_check) {
    SplitMix64 rng(0x802ull);
    const int K = 3, D = 8, H = 2, DF = 16;
    TransformerBlock blk;
    TransformerBlock::Config cfg{};
    cfg.dim = D; cfg.num_heads = H; cfg.d_ff = DF; cfg.n_slots = K;
    cfg.norm = NormPlacement::PostNorm;
    uint64_t s = 0x66ull; blk.init(cfg, s);

    Tensor X(K, D), upstream(K, D), Y(K, D);
    fill_random(X, rng);
    fill_random(upstream, rng);

    auto fwd_loss = [&]() {
        Tensor Ytmp(K, D);
        blk.forward(X, nullptr, Ytmp);
        return dot_loss(Ytmp, upstream);
    };

    blk.zero_grad();
    blk.forward(X, nullptr, Y);
    Tensor dX(K, D);
    blk.backward(upstream, dX);
    check_grad_input(X, dX, fwd_loss, "tblk.post.dX", 1e-3f, 1e-2f, 1e-2f, 8);
}

TEST(transformer_block_save_load_roundtrip) {
    SplitMix64 rng(0x803ull);
    const int K = 3, D = 8, H = 2, DF = 16;
    TransformerBlock a;
    TransformerBlock::Config cfg{}; cfg.dim = D; cfg.num_heads = H; cfg.d_ff = DF; cfg.n_slots = K;
    uint64_t s = 0x77ull; a.init(cfg, s);
    Tensor X(K, D); fill_random(X, rng);
    Tensor Ya(K, D); a.forward(X, nullptr, Ya);
    std::vector<uint8_t> buf; a.save_to(buf);

    TransformerBlock b;
    uint64_t s2 = 0x88ull; b.init(cfg, s2);
    size_t off = 0; b.load_from(buf.data(), off, buf.size());
    Tensor Yb(K, D); b.forward(X, nullptr, Yb);
    for (int i = 0; i < Ya.size(); ++i) CHECK(std::fabs(Ya.data[i] - Yb.data[i]) < 1e-5f);
}

TEST(transformer_encoder_smoke_and_save_load) {
    SplitMix64 rng(0x901ull);
    const int K = 4, D = 16, H = 4, DF = 32, N = 2;
    TransformerEncoder enc;
    TransformerEncoder::Config cfg{};
    cfg.n_layers = N; cfg.dim = D; cfg.num_heads = H; cfg.d_ff = DF; cfg.n_slots = K;
    cfg.norm = NormPlacement::PreNorm;
    uint64_t s = 0x11ull; enc.init(cfg, s);

    Tensor X(K, D); fill_random(X, rng);
    std::vector<float> mask = {1, 1, 0, 1};
    Tensor Y(K, D); enc.forward(X, mask.data(), Y);

    // num_params sanity: each block has 4*D*D + (D + D*DF + DF + DF*D + D)
    // and there are 2 LNs (each 2D) — just check it's positive.
    CHECK(enc.num_params() > 0);

    // save/load.
    std::vector<uint8_t> buf; enc.save_to(buf);
    TransformerEncoder enc2;
    uint64_t s2 = 0x22ull; enc2.init(cfg, s2);
    size_t off = 0; enc2.load_from(buf.data(), off, buf.size());
    Tensor Y2(K, D); enc2.forward(X, mask.data(), Y2);
    for (int i = 0; i < Y.size(); ++i) CHECK(std::fabs(Y.data[i] - Y2.data[i]) < 1e-5f);

    // backward smoke.
    Tensor upstream(K, D); fill_random(upstream, rng);
    Tensor dX(K, D);
    enc.zero_grad();
    enc.forward(X, mask.data(), Y);
    enc.backward(upstream, dX);
    // Just check finite values.
    for (int i = 0; i < dX.size(); ++i) CHECK(std::isfinite(dX.data[i]));
}

TEST(transformer_encoder_grad_check_pre_norm) {
    SplitMix64 rng(0x902ull);
    const int K = 3, D = 8, H = 2, DF = 16, N = 2;
    TransformerEncoder enc;
    TransformerEncoder::Config cfg{};
    cfg.n_layers = N; cfg.dim = D; cfg.num_heads = H; cfg.d_ff = DF; cfg.n_slots = K;
    cfg.norm = NormPlacement::PreNorm;
    uint64_t s = 0x33ull; enc.init(cfg, s);
    Tensor X(K, D), upstream(K, D), Y(K, D);
    fill_random(X, rng); fill_random(upstream, rng);

    auto fwd_loss = [&]() {
        Tensor Ytmp(K, D);
        enc.forward(X, nullptr, Ytmp);
        return dot_loss(Ytmp, upstream);
    };
    enc.zero_grad();
    enc.forward(X, nullptr, Y);
    Tensor dX(K, D);
    enc.backward(upstream, dX);
    check_grad_input(X, dX, fwd_loss, "enc.dX", 1e-3f, 2e-2f, 2e-2f, 8);
}

} // namespace

int main() {
    std::printf("brogameagent transformer tests\n");
    std::printf("==============================\n");
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
