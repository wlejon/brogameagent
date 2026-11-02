#include "test_common.h"

std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

// ─── PolicyValueNet head_sizes / checkpoint v2 ──────────────────────────────

TEST(policy_value_net_single_head_default_matches_num_actions) {
    nn::PolicyValueNet net;
    nn::PolicyValueNet::Config cfg;
    cfg.in_dim = 3; cfg.hidden = {4}; cfg.value_hidden = 4;
    cfg.num_actions = 5;
    net.init(cfg);
    CHECK(net.num_heads() == 1);
    CHECK(net.head_sizes()[0] == 5);
    CHECK(net.head_offsets().size() == 2);
    CHECK(net.head_offsets()[0] == 0);
    CHECK(net.head_offsets()[1] == 5);
}

TEST(policy_value_net_factored_heads_resolve_offsets) {
    nn::PolicyValueNet net;
    nn::PolicyValueNet::Config cfg;
    cfg.in_dim = 3; cfg.hidden = {4}; cfg.value_hidden = 4;
    cfg.head_sizes = {6, 2, 13, 9};      // num_actions auto-derives to 30
    net.init(cfg);
    CHECK(net.num_actions() == 30);
    CHECK(net.num_heads() == 4);
    CHECK(net.head_sizes()[0] == 6 && net.head_sizes()[3] == 9);
    CHECK(net.head_offsets()[0] == 0);
    CHECK(net.head_offsets()[1] == 6);
    CHECK(net.head_offsets()[2] == 8);
    CHECK(net.head_offsets()[3] == 21);
    CHECK(net.head_offsets()[4] == 30);
}

TEST(policy_value_net_save_load_v2_roundtrip_factored) {
    nn::PolicyValueNet a, b;
    nn::PolicyValueNet::Config cfg;
    cfg.in_dim = 4; cfg.hidden = {6}; cfg.value_hidden = 4;
    cfg.head_sizes = {3, 5}; cfg.seed = 7;
    a.init(cfg);
    b.init(cfg);

    auto blob = a.save();
    b.load(blob);

    // After loading, b's forward must match a's forward exactly.
    brotensor::Tensor x = brotensor::Tensor::vec(4);
    for (int i = 0; i < 4; ++i) x[i] = 0.1f * (i + 1);
    brotensor::Tensor la = brotensor::Tensor::vec(8), lb = brotensor::Tensor::vec(8);
    float va = 0.0f, vb = 0.0f;
    a.forward(x, va, la);
    b.forward(x, vb, lb);
    CHECK_NEAR(va, vb, 1e-6f);
    for (int i = 0; i < 8; ++i) CHECK_NEAR(la[i], lb[i], 1e-6f);
}

TEST(policy_value_net_v1_blob_loads_into_single_head_net) {
    // Build a v1-style blob by hand: magic + version=1 + weights. Today we
    // can't write v1 directly (save() always emits v2), so we synthesize one
    // by stripping the v2 head-sizes header from a single-head save.
    nn::PolicyValueNet a;
    nn::PolicyValueNet::Config cfg;
    cfg.in_dim = 3; cfg.hidden = {4}; cfg.value_hidden = 4;
    cfg.num_actions = 5; cfg.seed = 11;
    a.init(cfg);
    auto v2 = a.save();

    // v2 layout: [magic][version=2][n_heads=1][head=5][weights...]
    // v1 layout: [magic][version=1][weights...]
    // Strip the (n_heads, head) pair (8 bytes) and rewrite version.
    std::vector<uint8_t> v1;
    v1.resize(v2.size() - 8);
    std::memcpy(v1.data(), v2.data(), 8);
    const uint32_t v1_ver = 1;
    std::memcpy(v1.data() + 4, &v1_ver, 4);
    std::memcpy(v1.data() + 8, v2.data() + 16, v2.size() - 16);

    nn::PolicyValueNet b;
    b.init(cfg);
    b.load(v1);

    // Forward must match — load only restored weights, head shape was set
    // by init().
    brotensor::Tensor x = brotensor::Tensor::vec(3);
    x[0] = 0.5f; x[1] = -0.25f; x[2] = 1.0f;
    brotensor::Tensor la = brotensor::Tensor::vec(5), lb = brotensor::Tensor::vec(5);
    float va = 0.0f, vb = 0.0f;
    a.forward(x, va, la);
    b.forward(x, vb, lb);
    CHECK_NEAR(va, vb, 1e-6f);
    for (int i = 0; i < 5; ++i) CHECK_NEAR(la[i], lb[i], 1e-6f);
}

TEST(generic_trainer_factored_heads_drives_loss_down) {
    // Two-head net: heads {3, 4}. Train on a fixed (obs → per-head one-hot)
    // mapping and verify the policy loss decreases with SGD steps. This
    // exercises the per-head softmax-xent loop and the mean reduction.
    nn::PolicyValueNet net;
    nn::PolicyValueNet::Config cfg;
    cfg.in_dim = 4; cfg.hidden = {16, 16}; cfg.value_hidden = 8;
    cfg.head_sizes = {3, 4}; cfg.seed = 42;
    net.init(cfg);

    learn::GenericReplayBuffer buf(256);
    // 4 distinct (obs, target) pairs. Targets are concatenated per-head
    // one-hots, each summing to 1.0 within its segment.
    auto push = [&](std::vector<float> obs, int a0, int a1) {
        learn::GenericSituation s;
        s.obs = std::move(obs);
        s.policy_target.assign(7, 0.0f);
        s.policy_target[a0] = 1.0f;
        s.policy_target[3 + a1] = 1.0f;
        s.value_target = 0.0f;
        buf.push(std::move(s));
    };
    push({1, 0, 0, 0}, 0, 0);
    push({0, 1, 0, 0}, 1, 1);
    push({0, 0, 1, 0}, 2, 2);
    push({0, 0, 0, 1}, 0, 3);

    learn::GenericExItTrainer tr;
    learn::GenericTrainerConfig tcfg;
    tcfg.batch = 4; tcfg.lr = 0.05f; tcfg.momentum = 0.9f;
    tcfg.publish_every = 0;     // no handle, no publish
    tr.set_net(&net);
    tr.set_buffer(&buf);
    tr.set_config(tcfg);

    auto first = tr.step();
    for (int i = 0; i < 200; ++i) tr.step();
    auto last = tr.step();

    CHECK(first.samples == 4);
    CHECK(last.loss_policy < first.loss_policy * 0.5f);
}

TEST(factored_strides_row_major_last_head_fastest) {
    auto s = nn::head_strides({3, 4, 2});
    // strides = [4*2, 2, 1] = [8, 2, 1]
    CHECK(s.size() == 3);
    CHECK(s[0] == 8);
    CHECK(s[1] == 2);
    CHECK(s[2] == 1);
    CHECK(nn::flat_action_count({3, 4, 2}) == 24);
}

TEST(factored_count_rejects_overflow_and_empty_heads) {
    auto throws = [](const std::vector<int>& sizes, bool strides) {
        try {
            if (strides) nn::head_strides(sizes);
            else nn::flat_action_count(sizes);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    // 65536 * 65536 wraps a 32-bit int to 0.
    CHECK(throws({65536, 65536}, false));
    CHECK(throws({65536, 65536}, true));
    CHECK(throws({1 << 16, 1 << 15, 2}, false));
    CHECK(throws({3, 0, 2}, false));
    CHECK(throws({3, -2, 2}, true));
    CHECK(!throws({46340, 46340}, false));
    CHECK(nn::flat_action_count({46340, 46340}) == 46340 * 46340);
}

TEST(factored_encode_decode_roundtrip) {
    std::vector<int> sizes = {3, 4, 2};
    auto strides = nn::head_strides(sizes);
    int decoded[3];
    for (int flat = 0; flat < 24; ++flat) {
        nn::decode_flat_action(flat, sizes, strides, decoded);
        CHECK(decoded[0] >= 0 && decoded[0] < 3);
        CHECK(decoded[1] >= 0 && decoded[1] < 4);
        CHECK(decoded[2] >= 0 && decoded[2] < 2);
        CHECK(nn::encode_flat_action(decoded, strides, 3) == flat);
    }
}

TEST(factored_to_flat_uniform_logits_yields_uniform_prior) {
    // All logits equal → each per-head softmax is uniform → flat prior is
    // uniform 1/total over the cartesian product.
    std::vector<int> sizes = {3, 4};
    std::vector<int> offsets = {0, 3, 7};
    std::vector<float> logits(7, 0.0f);
    std::vector<float> flat(12, 0.0f);
    nn::factored_to_flat(logits.data(), sizes, offsets, flat.data());
    const float expected = 1.0f / 12.0f;
    for (int i = 0; i < 12; ++i) CHECK_NEAR(flat[i], expected, 1e-6f);
    float sum = 0.0f;
    for (float p : flat) sum += p;
    CHECK_NEAR(sum, 1.0f, 1e-5f);
}

TEST(factored_to_flat_matches_per_head_product) {
    // Hand-crafted per-head probs and verify the flat result equals the
    // explicit product. Logits are arbitrary; we compute expected by
    // softmaxing in the same way.
    std::vector<int> sizes = {2, 3};
    std::vector<int> offsets = {0, 2, 5};
    std::vector<float> logits = {1.0f, 0.0f,    // head 0
                                 2.0f, 1.0f, -1.0f}; // head 1
    std::vector<float> flat(6, 0.0f);
    nn::factored_to_flat(logits.data(), sizes, offsets, flat.data());

    // Compute expected per-head probs.
    auto sm = [](std::vector<float> v) {
        float m = v[0];
        for (float x : v) if (x > m) m = x;
        float s = 0;
        for (auto& x : v) { x = std::exp(x - m); s += x; }
        for (auto& x : v) x /= s;
        return v;
    };
    auto p0 = sm({1.0f, 0.0f});
    auto p1 = sm({2.0f, 1.0f, -1.0f});
    // Row-major: flat = a0 * 3 + a1.
    for (int a0 = 0; a0 < 2; ++a0) {
        for (int a1 = 0; a1 < 3; ++a1) {
            CHECK_NEAR(flat[a0 * 3 + a1], p0[a0] * p1[a1], 1e-6f);
        }
    }
}

TEST(factored_to_flat_honors_per_head_mask) {
    // Mask out one option per head; the flat prior should be 0 for any
    // cartesian combination involving a masked sub-action.
    std::vector<int> sizes = {2, 3};
    std::vector<int> offsets = {0, 2, 5};
    std::vector<float> logits(5, 0.0f);
    // Mask: head0 disables a0=1; head1 disables a1=2.
    std::vector<float> mask = {1, 0,    1, 1, 0};
    std::vector<float> flat(6, 0.0f);
    nn::factored_to_flat(logits.data(), sizes, offsets, flat.data(), mask.data());
    // Surviving combinations: a0=0 with a1∈{0,1}. Each = 1.0 * 0.5 = 0.5.
    CHECK_NEAR(flat[0 * 3 + 0], 0.5f, 1e-6f);
    CHECK_NEAR(flat[0 * 3 + 1], 0.5f, 1e-6f);
    CHECK_NEAR(flat[0 * 3 + 2], 0.0f, 1e-6f);
    CHECK_NEAR(flat[1 * 3 + 0], 0.0f, 1e-6f);
    CHECK_NEAR(flat[1 * 3 + 1], 0.0f, 1e-6f);
    CHECK_NEAR(flat[1 * 3 + 2], 0.0f, 1e-6f);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    printf("brogameagent tests\n");
    printf("==================\n");

    int passed = 0;
    for (const auto& t : registry()) {
        try {
            t.fn();
            passed++;
            printf("  PASS  %s\n", t.name);
        } catch (...) {
            printf("  FAIL  %s\n", t.name);
        }
    }

    int total = static_cast<int>(registry().size());
    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
