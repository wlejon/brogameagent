#include "test_common.h"

// ─── grid::ObsWindow / FrameStack Tests ────────────────────────────────────

TEST(obs_window_layout_invariants) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 2; s.cols_ahead = 3;
    s.rows_up = 1;     s.rows_down = 1;
    s.tile_channels = 2;
    s.self_block_size = 5;
    EntityLayerSpec layer;
    layer.channels = 3;
    layer.enumerate_fn = [] { return size_t{0}; };
    layer.sample_fn    = [](size_t) { return EntityCell{}; };
    ObsWindow win(s, [](int, int, float* o){ o[0]=0; o[1]=0; return true; }, {layer});
    const auto& L = win.layout();
    CHECK(L.cols == 6);
    CHECK(L.rows == 3);
    CHECK(L.tile_offset == 0);
    CHECK(L.tile_size == 6 * 3 * 2);
    CHECK(L.layers.size() == 1);
    CHECK(L.layers[0].offset == L.tile_size);
    CHECK(L.layers[0].size   == 6 * 3 * 3);
    CHECK(L.self_offset == L.tile_size + L.layers[0].size);
    CHECK(L.self_size   == 5);
    CHECK(L.total       == L.self_offset + 5);
    CHECK(win.out_dim() == L.total);
}

TEST(obs_window_tile_sampling_and_oob) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 1; s.cols_ahead = 1;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    s.oob_tile = {7.0f};
    // World is solid at col=0 only; everything else is OOB.
    auto tile = [](int col, int /*row*/, float* o) -> bool {
        if (col == 0) { o[0] = 3.0f; return true; }
        return false;   // OOB substitution path
    };
    ObsWindow win(s, tile, {});
    auto out = win.build(0, 0, {});
    // window cols = [-1, 0, 1] → [oob, 3, oob] → [7, 3, 7]
    CHECK(out.size() == 3);
    CHECK_NEAR(out[0], 7.0f, 1e-6f);
    CHECK_NEAR(out[1], 3.0f, 1e-6f);
    CHECK_NEAR(out[2], 7.0f, 1e-6f);
}

TEST(obs_window_entity_rasterizes_relative_to_ego) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 1; s.cols_ahead = 1;
    s.rows_up = 1;     s.rows_down = 1;
    s.tile_channels = 1;
    struct Ent { int col, row; float v; };
    std::vector<Ent> ents = { {5, 5, 1.0f}, {7, 6, 2.0f}, {99, 99, 9.0f} };
    EntityLayerSpec layer;
    layer.channels = 1;
    layer.enumerate_fn = [&] { return ents.size(); };
    layer.sample_fn    = [&](size_t i) {
        EntityCell c; c.col = ents[i].col; c.row = ents[i].row;
        c.values = { ents[i].v }; return c;
    };
    ObsWindow win(s, [](int,int,float* o){ o[0]=0; return true; }, {layer});
    auto out = win.build(6, 5, {});
    // window is 3 cols × 3 rows; offsets:
    //   ego at (col=1, row=1) maps to world (6,5)
    //   ent (5,5) → window (0,1) — "behind, same row"
    //   ent (7,6) → window (2,2) — "ahead, below"
    //   ent (99,99) — out of window, must not appear
    int tile_size = 3 * 3 * 1;
    int layer_off = tile_size;
    auto cell = [&](int wc, int wr) { return out[layer_off + (wr*3 + wc)]; };
    CHECK_NEAR(cell(0, 1), 1.0f, 1e-6f);
    CHECK_NEAR(cell(2, 2), 2.0f, 1e-6f);
    // Center cell (ego) should be untouched (no entity there).
    CHECK_NEAR(cell(1, 1), 0.0f, 1e-6f);
}

TEST(obs_window_entity_layer_modes) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 0; s.cols_ahead = 0;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    struct E { int col, row; float v; };
    std::vector<E> ents = { {0,0, 1.0f}, {0,0, 2.0f}, {0,0, 4.0f} };
    auto enumerate = [&] { return ents.size(); };
    auto sample = [&](size_t i) {
        EntityCell c; c.col=ents[i].col; c.row=ents[i].row; c.values={ents[i].v}; return c;
    };
    {
        EntityLayerSpec L; L.channels=1; L.overwrite=false;
        L.enumerate_fn=enumerate; L.sample_fn=sample;
        ObsWindow win(s, [](int,int,float* o){o[0]=0;return true;}, {L});
        auto out = win.build(0, 0, {});
        // additive: 1+2+4=7
        CHECK_NEAR(out[1], 7.0f, 1e-6f);   // out[0] is tile, out[1] is layer
    }
    {
        EntityLayerSpec L; L.channels=1; L.overwrite=true;
        L.enumerate_fn=enumerate; L.sample_fn=sample;
        ObsWindow win(s, [](int,int,float* o){o[0]=0;return true;}, {L});
        auto out = win.build(0, 0, {});
        // last write wins: 4
        CHECK_NEAR(out[1], 4.0f, 1e-6f);
    }
}

TEST(obs_window_normalizers_apply_after_accumulate) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 0; s.cols_ahead = 0;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    s.tile_normalize = {0.5f};
    EntityLayerSpec L;
    L.channels = 1;
    L.normalize = {0.25f};
    std::vector<std::pair<int,int>> dummy = {{0,0},{0,0}};
    L.enumerate_fn = [&]{ return dummy.size(); };
    L.sample_fn    = [&](size_t){ EntityCell c; c.col=0; c.row=0; c.values={4.0f}; return c; };
    ObsWindow win(s, [](int,int,float* o){o[0]=2.0f;return true;}, {L});
    auto out = win.build(0,0,{});
    CHECK_NEAR(out[0], 2.0f * 0.5f, 1e-6f);             // tile post-normalize
    CHECK_NEAR(out[1], (4.0f + 4.0f) * 0.25f, 1e-6f);  // accumulate then normalize
}

TEST(obs_window_self_block_copied) {
    using namespace brogameagent::grid;
    ObsWindowSpec s;
    s.cols_behind = 0; s.cols_ahead = 0;
    s.rows_up = 0;     s.rows_down = 0;
    s.tile_channels = 1;
    s.self_block_size = 4;
    ObsWindow win(s, [](int,int,float* o){o[0]=0;return true;}, {});
    auto out = win.build(0, 0, {1.0f, 2.0f, 3.0f});  // shorter than self block
    CHECK(out.size() == 5);
    CHECK_NEAR(out[1], 1.0f, 1e-6f);
    CHECK_NEAR(out[2], 2.0f, 1e-6f);
    CHECK_NEAR(out[3], 3.0f, 1e-6f);
    CHECK_NEAR(out[4], 0.0f, 1e-6f);   // missing self entry padded to 0
}

TEST(frame_stack_zero_initialized_and_pads_leading_frames) {
    using namespace brogameagent::grid;
    FrameStack fs(3, 4);
    CHECK(fs.out_dim() == 12);
    CHECK(fs.filled() == 0);
    auto out = fs.read();
    for (float v : out) CHECK_NEAR(v, 0.0f, 1e-6f);
    // Push one frame; the freshest frame must be the *last* k-block.
    float frame[] = {1.0f, 2.0f, 3.0f};
    fs.push(frame);
    out = fs.read();
    // Slots [0..2] zero (pad), slot 3 holds frame.
    for (int i = 0; i < 9; ++i) CHECK_NEAR(out[i], 0.0f, 1e-6f);
    CHECK_NEAR(out[9],  1.0f, 1e-6f);
    CHECK_NEAR(out[10], 2.0f, 1e-6f);
    CHECK_NEAR(out[11], 3.0f, 1e-6f);
}

TEST(frame_stack_chronological_order_when_full) {
    using namespace brogameagent::grid;
    FrameStack fs(1, 3);
    float a = 1.0f, b = 2.0f, c = 3.0f, d = 4.0f, e = 5.0f;
    fs.push(&a); fs.push(&b); fs.push(&c);
    auto out = fs.read();
    CHECK_NEAR(out[0], 1.0f, 1e-6f);
    CHECK_NEAR(out[1], 2.0f, 1e-6f);
    CHECK_NEAR(out[2], 3.0f, 1e-6f);   // freshest last
    fs.push(&d);  // evicts oldest (a)
    out = fs.read();
    CHECK_NEAR(out[0], 2.0f, 1e-6f);
    CHECK_NEAR(out[1], 3.0f, 1e-6f);
    CHECK_NEAR(out[2], 4.0f, 1e-6f);
    fs.push(&e);
    out = fs.read();
    CHECK_NEAR(out[0], 3.0f, 1e-6f);
    CHECK_NEAR(out[1], 4.0f, 1e-6f);
    CHECK_NEAR(out[2], 5.0f, 1e-6f);
}

TEST(frame_stack_reset_clears_history) {
    using namespace brogameagent::grid;
    FrameStack fs(1, 2);
    float a = 7.0f;
    fs.push(&a); fs.push(&a); fs.push(&a);
    fs.reset();
    CHECK(fs.filled() == 0);
    auto out = fs.read();
    CHECK_NEAR(out[0], 0.0f, 1e-6f);
    CHECK_NEAR(out[1], 0.0f, 1e-6f);
}

// ─── grid::FailureTape Tests ────────────────────────────────────────────────

TEST(failure_tape_records_only_tail_depth) {
    using namespace brogameagent::grid;
    FailureTapeConfig cfg; cfg.tape_depth = 3; cfg.ring_capacity = 16;
    cfg.penalty = 0.5f; cfg.floor = 0.05f;
    FailureTape tape(cfg);
    std::vector<FailureStep> tail;
    for (int i = 0; i < 10; ++i) tail.push_back({"S", i});
    tape.record_failure(tail);
    CHECK(tape.size() == 3);
    // Only the last 3 actions (7,8,9) should be counted; 0..6 must be 1.0.
    auto m = tape.multipliers("S", 10);
    for (int a = 0; a < 7; ++a) CHECK_NEAR(m[a], 1.0f, 1e-6f);
    CHECK_NEAR(m[7], 0.5f, 1e-6f);
    CHECK_NEAR(m[8], 0.5f, 1e-6f);
    CHECK_NEAR(m[9], 0.5f, 1e-6f);
}

TEST(failure_tape_multiplier_compounds_on_repeats_and_clamps_at_floor) {
    using namespace brogameagent::grid;
    FailureTapeConfig cfg; cfg.tape_depth = 100; cfg.ring_capacity = 256;
    cfg.penalty = 0.5f; cfg.floor = 0.1f;
    FailureTape tape(cfg);
    std::vector<FailureStep> t = { {"X", 2}, {"X", 2}, {"X", 2}, {"X", 2}, {"X", 2} };
    tape.record_failure(t);
    auto m = tape.multipliers("X", 4);
    // 0.5^5 = 0.03125 < 0.1 floor → clamps.
    CHECK_NEAR(m[2], 0.1f, 1e-6f);
    // Other actions unaffected.
    CHECK_NEAR(m[0], 1.0f, 1e-6f);
}

TEST(failure_tape_unknown_sig_is_identity) {
    using namespace brogameagent::grid;
    FailureTape tape;
    auto m = tape.multipliers("never_seen", 6);
    for (float v : m) CHECK_NEAR(v, 1.0f, 1e-6f);
}

TEST(failure_tape_ring_eviction_drops_old_counts) {
    using namespace brogameagent::grid;
    FailureTapeConfig cfg; cfg.tape_depth = 100; cfg.ring_capacity = 4;
    cfg.penalty = 0.5f; cfg.floor = 0.0f;
    FailureTape tape(cfg);
    // Fill ring with one (S,1) then push 4 (T,2) — S must be fully evicted.
    tape.record_failure({ {"S", 1} });
    tape.record_failure({ {"T", 2}, {"T", 2}, {"T", 2}, {"T", 2} });
    CHECK(tape.size() == 4);
    auto mS = tape.multipliers("S", 4);
    for (float v : mS) CHECK_NEAR(v, 1.0f, 1e-6f);
    auto mT = tape.multipliers("T", 4);
    CHECK_NEAR(mT[2], 0.5f * 0.5f * 0.5f * 0.5f, 1e-6f);
}

TEST(failure_tape_apply_priors_in_place) {
    using namespace brogameagent::grid;
    FailureTape tape;
    tape.record_failure({ {"S", 1}, {"S", 1} });
    std::vector<float> prior = {0.25f, 0.50f, 0.25f, 0.0f};
    tape.apply_priors("S", prior.data(), static_cast<int>(prior.size()));
    CHECK_NEAR(prior[0], 0.25f, 1e-6f);
    CHECK_NEAR(prior[1], 0.50f * 0.25f, 1e-6f);
    CHECK_NEAR(prior[2], 0.25f, 1e-6f);
}

TEST(failure_tape_clear_resets_state) {
    using namespace brogameagent::grid;
    FailureTape tape;
    tape.record_failure({ {"S", 1} });
    CHECK(tape.size() == 1);
    tape.clear();
    CHECK(tape.size() == 0);
    auto m = tape.multipliers("S", 4);
    for (float v : m) CHECK_NEAR(v, 1.0f, 1e-6f);
}

// ─── grid::BestCrop Tests ──────────────────────────────────────────────────

TEST(best_crop_orders_by_score) {
    using namespace brogameagent::grid;
    BestCropConfig cfg; cfg.capacity = 8; cfg.depth_bonus = 0.0f; cfg.age_decay = 0.0f;
    BestCrop pool(cfg);
    pool.push(std::any{1}, {}, /*score*/ 1.0f, 5);
    pool.push(std::any{2}, {}, /*score*/ 5.0f, 5);
    pool.push(std::any{3}, {}, /*score*/ 3.0f, 5);
    auto sorted = pool.sorted();
    CHECK(sorted.size() == 3);
    CHECK_NEAR(sorted[0]->score, 5.0f, 1e-6f);
    CHECK_NEAR(sorted[1]->score, 3.0f, 1e-6f);
    CHECK_NEAR(sorted[2]->score, 1.0f, 1e-6f);
}

TEST(best_crop_capacity_drops_worst) {
    using namespace brogameagent::grid;
    BestCropConfig cfg; cfg.capacity = 2; cfg.age_decay = 0.0f;
    BestCrop pool(cfg);
    pool.push(std::any{}, {}, 1.0f, 0);
    pool.push(std::any{}, {}, 5.0f, 0);
    pool.push(std::any{}, {}, 3.0f, 0);   // pushes capacity over → drop 1.0
    CHECK(pool.size() == 2);
    auto sorted = pool.sorted();
    CHECK_NEAR(sorted[0]->score, 5.0f, 1e-6f);
    CHECK_NEAR(sorted[1]->score, 3.0f, 1e-6f);
}

TEST(best_crop_depth_bonus_breaks_tie) {
    using namespace brogameagent::grid;
    BestCropConfig cfg; cfg.capacity = 8; cfg.depth_bonus = 0.1f; cfg.age_decay = 0.0f;
    BestCrop pool(cfg);
    pool.push(std::any{}, {}, /*score*/ 5.0f, /*depth*/ 100);
    pool.push(std::any{}, {}, /*score*/ 5.0f, /*depth*/ 10);
    auto sorted = pool.sorted();
    CHECK(sorted[0]->depth == 100);
    CHECK(sorted[1]->depth == 10);
}

TEST(best_crop_age_decay_demotes_old_entries) {
    using namespace brogameagent::grid;
    BestCropConfig cfg; cfg.capacity = 8; cfg.depth_bonus = 0.0f; cfg.age_decay = 1.0f;
    BestCrop pool(cfg);
    pool.push(std::any{}, {}, 10.0f, 0);   // age 0 at push, will increase
    // Push many newer-but-lower-score entries to age the first one out.
    for (int i = 0; i < 20; ++i) pool.push(std::any{}, {}, 0.5f, 0);
    auto sorted = pool.sorted();
    // Head entry: score 0.5 with very small age beats score 10 - age*1.0.
    // (10 - 20 = -10; vs 0.5 - 0 = 0.5)
    CHECK_NEAR(sorted[0]->score, 0.5f, 1e-6f);
}

TEST(best_crop_seed_returns_snapshot_and_prefix) {
    using namespace brogameagent::grid;
    BestCrop pool({});
    pool.push(std::any{int{42}}, {1, 2, 3}, /*score*/ 9.0f, 3);
    std::mt19937_64 rng(0xC0DEull);
    auto seed = pool.seed(rng);
    CHECK(seed.snapshot.has_value());
    CHECK(std::any_cast<int>(seed.snapshot) == 42);
    CHECK(seed.prefix.size() == 3);
    CHECK(seed.prefix[2] == 3);
}

TEST(best_crop_seed_empty_pool_is_default) {
    using namespace brogameagent::grid;
    BestCrop pool({});
    std::mt19937_64 rng(1);
    auto seed = pool.seed(rng);
    CHECK(!seed.snapshot.has_value());
    CHECK(seed.prefix.empty());
}

// ─── grid::Shaping Tests ───────────────────────────────────────────────────

TEST(potential_shaper_returns_gamma_phi_diff) {
    using namespace brogameagent::grid;
    PotentialShaper sh(0.99f);
    sh.reset(/*phi0*/ 1.0f);
    float b = sh.step(2.0f);              // 0.99*2 - 1 = 0.98
    CHECK_NEAR(b, 0.98f, 1e-5f);
    b = sh.step(2.0f);                    // 0.99*2 - 2 = -0.02
    CHECK_NEAR(b, -0.02f, 1e-5f);
}

TEST(potential_shaper_first_step_unprimed_returns_zero) {
    using namespace brogameagent::grid;
    PotentialShaper sh(0.99f);
    float b = sh.step(5.0f);              // unprimed: latches, returns 0
    CHECK_NEAR(b, 0.0f, 1e-6f);
    b = sh.step(7.0f);                    // 0.99*7 - 5 = 1.93
    CHECK_NEAR(b, 1.93f, 1e-5f);
}

TEST(stall_detector_fires_only_after_patience_samples_below_epsilon) {
    using namespace brogameagent::grid;
    StallDetector det(/*eps*/ 0.5f, /*patience*/ 4);
    CHECK(!det.tick(0.0f));   // filling
    CHECK(!det.tick(0.1f));
    CHECK(!det.tick(0.2f));
    // Now filled. Window = {0,0.1,0.2,X}; spread depends on next push.
    CHECK(det.tick(0.3f));    // {0, 0.1, 0.2, 0.3} spread 0.3 < 0.5
    CHECK(!det.tick(2.0f));   // {0.1, 0.2, 0.3, 2.0} spread 1.9 ≥ 0.5
    CHECK(!det.tick(2.1f));   // still has 0.2 in window
    CHECK(!det.tick(2.2f));   // still has 0.3
    CHECK(det.tick(2.3f));    // {2.0,2.1,2.2,2.3} spread 0.3 < 0.5
}

TEST(stall_detector_reset_clears_window) {
    using namespace brogameagent::grid;
    StallDetector det(0.5f, 3);
    det.tick(1.0f); det.tick(1.0f); det.tick(1.0f);
    CHECK(det.tick(1.0f));    // stalled
    det.reset();
    CHECK(!det.tick(1.0f));   // refilling
    CHECK(!det.tick(1.0f));
    CHECK(det.tick(1.0f));    // stalled again
}

// ─── grid::BC ingestion Tests ──────────────────────────────────────────────

namespace {
// Tiny 1D corridor env used by BC + harness tests.
//   state = int position in [0, 10]
//   actions: 0=left, 1=right, 2=stay
//   reward: +1 on reaching pos==goal, 0 otherwise; episode ends on goal
struct CorridorState {
    int pos = 0;
    int goal = 10;
    bool done = false;
};

mcts::GenericEnv make_corridor_for_bc(CorridorState& st) {
    mcts::GenericEnv env;
    env.num_actions = 3;
    env.snapshot_fn = [&]() -> std::any { return st; };
    env.restore_fn  = [&](const std::any& s) { st = std::any_cast<CorridorState>(s); };
    env.legal_actions_fn = [&]() -> std::vector<int> {
        if (st.done) return {};
        std::vector<int> a;
        if (st.pos > 0) a.push_back(0);
        a.push_back(1);
        a.push_back(2);
        return a;
    };
    env.observe_fn = [&]() -> std::vector<float> {
        return { static_cast<float>(st.pos) / static_cast<float>(st.goal) };
    };
    env.step_fn = [&](int action) -> mcts::GenericStepResult {
        mcts::GenericStepResult r;
        if (st.done) { r.done = true; return r; }
        if (action == 0 && st.pos > 0) st.pos--;
        else if (action == 1) st.pos++;
        if (st.pos >= st.goal) { st.done = true; r.reward = 1.0f; r.done = true; }
        return r;
    };
    return env;
}
} // namespace

TEST(bc_ingest_emits_situations_with_one_hot_policy) {
    using namespace brogameagent::grid;
    CorridorState st;
    auto env = make_corridor_for_bc(st);
    // Heuristic: always go right.
    auto policy = [](const std::vector<float>&, const std::vector<int>&) { return 1; };
    BCConfig cfg; cfg.rollout_horizon = 32; cfg.min_return = 0.0f; cfg.gamma = 0.9f;
    std::vector<std::any> starts = { CorridorState{} };  // start at pos=0
    auto sits = generate_bc_situations(env, policy, starts, cfg);
    CHECK(!sits.empty());
    // One step per move toward goal — corridor is length 10, so exactly 10
    // situations until goal.
    CHECK(sits.size() == 10);
    for (const auto& s : sits) {
        CHECK(s.policy_target.size() == 3);
        // One-hot at action 1 (right).
        CHECK_NEAR(s.policy_target[1], 1.0f, 1e-6f);
        CHECK_NEAR(s.policy_target[0], 0.0f, 1e-6f);
        CHECK_NEAR(s.policy_target[2], 0.0f, 1e-6f);
        CHECK(s.action_mask.size() == 3);
        // Action 1 must be legal at every step.
        CHECK_NEAR(s.action_mask[1], 1.0f, 1e-6f);
    }
    // Value-to-go should be monotonically increasing (closer to terminal
    // reward => higher discounted return).
    for (size_t i = 1; i < sits.size(); ++i)
        CHECK(sits[i].value_target >= sits[i - 1].value_target - 1e-5f);
    // Final step: reward 1 with no further discount.
    CHECK_NEAR(sits.back().value_target, 1.0f, 1e-5f);
}

TEST(bc_ingest_filters_below_min_return) {
    using namespace brogameagent::grid;
    CorridorState st;
    auto env = make_corridor_for_bc(st);
    // Heuristic that just stays — return 0, below threshold.
    auto policy = [](const std::vector<float>&, const std::vector<int>&) { return 2; };
    BCConfig cfg; cfg.rollout_horizon = 8; cfg.min_return = 0.5f;
    std::vector<std::any> starts = { CorridorState{} };
    auto sits = generate_bc_situations(env, policy, starts, cfg);
    CHECK(sits.empty());
}

TEST(bc_ingest_drops_demonstrations_with_illegal_action) {
    using namespace brogameagent::grid;
    CorridorState st;
    auto env = make_corridor_for_bc(st);
    // Heuristic always picks "left", which is illegal at pos=0 (start state).
    auto policy = [](const std::vector<float>&, const std::vector<int>&) { return 0; };
    BCConfig cfg; cfg.rollout_horizon = 8; cfg.min_return = 0.0f;
    std::vector<std::any> starts = { CorridorState{} };
    auto sits = generate_bc_situations(env, policy, starts, cfg);
    // Generation stops on the first illegal-action step; the trace was
    // empty before that, so nothing is emitted.
    CHECK(sits.empty());
}

// ─── grid::GenericRecorder Tests ───────────────────────────────────────────

TEST(generic_recorder_round_trip) {
    using namespace brogameagent::grid;
    const char* path = "test_grid_replay.bgargrid";
    {
        GenericRecorder rec;
        std::vector<FieldDef> roster = { {"id", FieldType::I32}, {"kind", FieldType::I32} };
        std::vector<FieldDef> frame  = { {"id", FieldType::I32}, {"x", FieldType::F32}, {"y", FieldType::F32} };
        std::vector<FieldDef> events = { {"target", FieldType::I32}, {"amount", FieldType::F32} };
        CHECK(rec.open(path, /*ep*/ 7, /*seed*/ 0xC0DE, /*dt*/ 1.0f / 60.0f,
                       roster, frame, events));
        rec.write_roster({
            { int32_t{0}, int32_t{1} },
            { int32_t{1}, int32_t{2} },
        });
        rec.record_frame(0, 0.0f,
            { { int32_t{0}, 1.0f, 2.0f }, { int32_t{1}, 3.0f, 4.0f } },
            {});
        rec.record_frame(1, 1.0f / 60.0f,
            { { int32_t{0}, 1.5f, 2.5f }, { int32_t{1}, 3.5f, 4.5f } },
            { { int32_t{1}, 12.0f } });
        CHECK(rec.frame_count() == 2);
        CHECK(rec.close());
    }
    {
        GenericReplayReader rr;
        CHECK(rr.open(path));
        CHECK(rr.episode_id() == 7);
        CHECK(rr.seed()       == 0xC0DEu);
        CHECK_NEAR(rr.dt(), 1.0f / 60.0f, 1e-6f);
        CHECK(rr.frame_count() == 2);
        CHECK(rr.roster().size() == 2);
        CHECK(std::get<int32_t>(rr.roster()[0][0]) == 0);
        CHECK(std::get<int32_t>(rr.roster()[1][1]) == 2);
        auto f0 = rr.frame(0);
        CHECK(f0.step_idx == 0);
        CHECK(f0.rows.size() == 2);
        CHECK_NEAR(std::get<float>(f0.rows[0][1]), 1.0f, 1e-6f);
        auto f1 = rr.frame(1);
        CHECK(f1.events.size() == 1);
        CHECK_NEAR(std::get<float>(f1.events[0][1]), 12.0f, 1e-6f);
        // Trajectory of "x" for row index 0 across both frames.
        auto traj = rr.trajectory(0, "x");
        CHECK(traj.size() == 2);
        CHECK_NEAR(std::get<float>(traj[0]), 1.0f, 1e-6f);
        CHECK_NEAR(std::get<float>(traj[1]), 1.5f, 1e-6f);
    }
    std::remove(path);
}

TEST(generic_recorder_handles_no_events_schema) {
    using namespace brogameagent::grid;
    const char* path = "test_grid_replay_no_events.bgargrid";
    {
        GenericRecorder rec;
        std::vector<FieldDef> roster = { {"id", FieldType::I32} };
        std::vector<FieldDef> frame  = { {"id", FieldType::I32}, {"score", FieldType::I64} };
        CHECK(rec.open(path, 1, 1, 1.0f, roster, frame, {}));
        rec.write_roster({ { int32_t{42} } });
        rec.record_frame(0, 0.0f, { { int32_t{42}, int64_t{100} } });
        rec.close();
    }
    {
        GenericReplayReader rr;
        CHECK(rr.open(path));
        CHECK(rr.event_schema().empty());
        auto f = rr.frame(0);
        CHECK(f.events.empty());
        CHECK(std::get<int64_t>(f.rows[0][1]) == int64_t{100});
    }
    std::remove(path);
}

// ─── grid::GridTrainer Tests ───────────────────────────────────────────────

TEST(harness_step_sync_drains_situations_and_runs_sgd) {
    using namespace brogameagent::grid;
    GridTrainerConfig cfg;
    cfg.net.in_dim      = 4;
    cfg.net.hidden      = {8, 8};
    cfg.net.value_hidden = 4;
    cfg.net.num_actions = 3;
    cfg.net.seed        = 1;
    cfg.buffer_capacity = 1024;
    cfg.trainer.batch   = 4;
    cfg.trainer.lr      = 0.01f;
    cfg.trainer.publish_every = 4;
    GridTrainer h(std::move(cfg));

    // Push some hand-built situations.
    for (int i = 0; i < 8; ++i) {
        learn::GenericSituation s;
        s.obs = {0.1f, 0.2f, 0.3f, 0.4f};
        s.policy_target = {1.0f, 0.0f, 0.0f};
        s.action_mask   = {1.0f, 1.0f, 1.0f};
        s.value_target  = 0.5f;
        h.ingest_situation(std::move(s));
    }
    h.step_sync(/*sgd_steps*/ 8);
    auto st = h.stats();
    CHECK(st.buffer_size >= 8);
    CHECK(st.total_steps >= 1);
    // publish_every=4, 8 steps → at least one publish event.
    auto evs = h.poll_events();
    bool saw_pub = false;
    for (auto& e : evs)
        if (e.kind == GridEvent::Kind::WeightsUpdated) { saw_pub = true; break; }
    CHECK(saw_pub);
    CHECK(h.weights().version() >= 1);
}

TEST(harness_episode_ingest_routes_to_best_and_failure_tape) {
    using namespace brogameagent::grid;
    GridTrainerConfig cfg;
    cfg.net.in_dim      = 1; cfg.net.hidden = {4}; cfg.net.value_hidden = 4;
    cfg.net.num_actions = 2;
    cfg.best_window     = 4;
    GridTrainer h(std::move(cfg));

    EpisodeSummary good;
    good.total_return  = 5.0f;
    good.depth         = 20;
    good.failed        = false;
    good.start_snapshot = std::any{int{1}};
    good.action_prefix = {1, 1, 1};
    h.ingest_episode(std::move(good));

    EpisodeSummary bad;
    bad.total_return  = -1.0f;
    bad.depth         = 5;
    bad.failed        = true;
    bad.failure_tail  = { {"S", 0}, {"S", 0} };
    h.ingest_episode(std::move(bad));

    h.step_sync(0);   // drain only

    CHECK(h.best_crop().size() == 1);
    CHECK(h.best_crop().sorted()[0]->depth == 20);
    auto m = h.failure_tape().multipliers("S", 2);
    CHECK(m[0] < 1.0f);   // penalty applied
    CHECK(m[1] == 1.0f);
    auto evs = h.poll_events();
    int eps = 0;
    for (auto& e : evs) if (e.kind == GridEvent::Kind::EpisodeIngested) ++eps;
    CHECK(eps == 2);
}

TEST(harness_async_thread_runs_and_stops_clean) {
    using namespace brogameagent::grid;
    GridTrainerConfig cfg;
    cfg.net.in_dim = 2; cfg.net.hidden = {4}; cfg.net.value_hidden = 4;
    cfg.net.num_actions = 2;
    cfg.trainer.batch = 2;
    cfg.trainer.publish_every = 2;
    cfg.steps_per_tick = 1;
    GridTrainer h(std::move(cfg));
    h.start();
    for (int i = 0; i < 8; ++i) {
        learn::GenericSituation s;
        s.obs = {0.0f, 1.0f};
        s.policy_target = {1.0f, 0.0f};
        s.action_mask = {1.0f, 1.0f};
        s.value_target = 0.0f;
        h.ingest_situation(std::move(s));
    }
    // Give the thread a moment to drain + train.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (std::chrono::steady_clock::now() < deadline) {
        if (h.stats().total_steps > 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK(h.stats().total_steps > 0);
    h.stop();
    CHECK(!h.running());
}
