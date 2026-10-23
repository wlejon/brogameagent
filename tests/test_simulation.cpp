// Simulation policy-map mutation during step(): add/removePolicy called from
// inside a policy are queued and applied when the outermost step ends, so a
// policy never destroys the function that is running and the step finishes
// with the policy set it started with. Core-only.

#include <brogameagent/brogameagent.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace brogameagent;

struct TestEntry {
    const char* name;
    void (*fn)();
};

static std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { Register_##name() { registry().push_back({#name, test_##name}); } } reg_##name; \
    static void test_##name()

static void check(bool cond, const char* msg, int line) {
    if (!cond) {
        printf("    assertion failed at line %d: %s\n", line, msg);
        throw 0;
    }
}

#define CHECK(cond) check(cond, #cond, __LINE__)

namespace {

struct Arena {
    World world;
    Agent a, b;
    Arena() {
        a.unit().id = 1; a.unit().hp = 100; a.unit().maxHp = 100;
        b.unit().id = 2; b.unit().hp = 100; b.unit().maxHp = 100;
        b.setPosition(3.0f, 0.0f);
        world.addAgent(&a);
        world.addAgent(&b);
    }
};

// Flags when the closure that owns it is destroyed.
struct Sentinel {
    bool* destroyed;
    explicit Sentinel(bool* d) : destroyed(d) {}
    ~Sentinel() { *destroyed = true; }
};

}  // namespace

TEST(policy_removing_itself_outlives_its_own_call) {
    Arena ar;
    Simulation sim(ar.world);
    bool destroyed = false;
    bool aliveAfterRemove = false;
    int calls = 0;
    auto sentinel = std::make_shared<Sentinel>(&destroyed);
    sim.addPolicy(1, [&, sentinel](Agent&, const World&) {
        ++calls;
        sim.removePolicy(1);
        // The closure (and the sentinel it alone owns) must still exist.
        aliveAfterRemove = !destroyed && sentinel->destroyed == &destroyed;
        return AgentAction{};
    });
    sentinel.reset();
    CHECK(!destroyed);
    CHECK(sim.hasPolicy(1));
    sim.step(0.1f);
    CHECK(aliveAfterRemove);
    CHECK(destroyed);            // applied at step end
    CHECK(!sim.hasPolicy(1));
    sim.step(0.1f);
    CHECK(calls == 1);
}

TEST(policy_replacing_itself_takes_effect_next_step) {
    Arena ar;
    Simulation sim(ar.world);
    int first = 0, second = 0;
    bool destroyed = false;
    auto sentinel = std::make_shared<Sentinel>(&destroyed);
    bool aliveAfterReplace = false;
    sim.addPolicy(1, [&, sentinel](Agent&, const World&) {
        ++first;
        sim.addPolicy(1, [&](Agent&, const World&) { ++second; return AgentAction{}; });
        aliveAfterReplace = !destroyed;
        return AgentAction{};
    });
    sentinel.reset();
    sim.step(0.1f);
    CHECK(aliveAfterReplace);
    CHECK(first == 1 && second == 0);
    sim.step(0.1f);
    CHECK(first == 1 && second == 1);
}

TEST(policy_added_for_another_agent_waits_for_next_step) {
    Arena ar;
    Simulation sim(ar.world);
    int bCalls = 0;
    sim.addPolicy(1, [&](Agent&, const World&) {
        if (!sim.hasPolicy(2)) {
            sim.addPolicy(2, [&](Agent&, const World&) { ++bCalls; return AgentAction{}; });
            // Queued changes are visible to hasPolicy immediately.
            check(sim.hasPolicy(2), "hasPolicy sees the queued add", __LINE__);
        }
        return AgentAction{};
    });
    sim.step(0.1f);   // agent 2 (after agent 1 in the roster) still scripted
    CHECK(bCalls == 0);
    sim.step(0.1f);
    CHECK(bCalls == 1);
}

TEST(queued_changes_apply_in_call_order) {
    Arena ar;
    Simulation sim(ar.world);
    int bCalls = 0;
    bool once = false;
    sim.addPolicy(1, [&](Agent&, const World&) {
        if (!once) {
            once = true;
            sim.addPolicy(2, [&](Agent&, const World&) { ++bCalls; return AgentAction{}; });
            sim.removePolicy(2);
            check(!sim.hasPolicy(2), "remove after add wins", __LINE__);
        }
        return AgentAction{};
    });
    sim.step(0.1f);
    sim.step(0.1f);
    CHECK(bCalls == 0);
    CHECK(!sim.hasPolicy(2));
}

TEST(throwing_policy_still_applies_queue_and_resets) {
    Arena ar;
    Simulation sim(ar.world);
    sim.addPolicy(1, [&](Agent&, const World&) -> AgentAction {
        sim.removePolicy(1);
        throw std::runtime_error("policy failed");
    });
    bool threw = false;
    try { sim.step(0.1f); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK(!sim.hasPolicy(1));
    // Outside a step, changes apply at once again.
    sim.addPolicy(2, [](Agent&, const World&) { return AgentAction{}; });
    CHECK(sim.hasPolicy(2));
    sim.removePolicy(2);
    CHECK(!sim.hasPolicy(2));
}

TEST(reentrant_step_defers_to_the_outermost) {
    Arena ar;
    Simulation sim(ar.world);
    int depth = 0, inner = 0;
    sim.addPolicy(1, [&](Agent&, const World&) {
        if (depth == 0) {
            ++depth;
            sim.removePolicy(1);
            sim.step(0.1f);   // nested: the policy set is unchanged in here
            --depth;
        } else {
            ++inner;
        }
        return AgentAction{};
    });
    sim.step(0.1f);
    CHECK(inner == 1);
    CHECK(!sim.hasPolicy(1));
    CHECK(sim.steps() == 2);
}

TEST(vec_simulation_parallel_determinism) {
    constexpr int kNumEnvs = 64;
    VecSimulation::Config cfg;
    cfg.numEnvs = kNumEnvs;
    VecSimulation vecSim(cfg);
    vecSim.seedAndReset(12345ULL);

    // Reference: 64 independent 1-env VecSimulations (which run the serial path).
    VecSimulation::Config singleCfg = cfg;
    singleCfg.numEnvs = 1;
    std::vector<std::unique_ptr<VecSimulation>> refSims;
    refSims.reserve(kNumEnvs);
    for (int i = 0; i < kNumEnvs; ++i) {
        auto s = std::make_unique<VecSimulation>(singleCfg);
        s->seedAndReset(12345ULL + static_cast<uint64_t>(i));
        refSims.push_back(std::move(s));
    }

    std::vector<float> obsPar(kNumEnvs * observation::TOTAL, 0.0f);
    std::vector<float> obsRef(observation::TOTAL, 0.0f);
    std::vector<float> maskPar(kNumEnvs * action_mask::TOTAL, 0.0f);
    std::vector<int> idsPar(kNumEnvs * action_mask::N_ENEMY_SLOTS, 0);
    std::vector<float> maskRef(action_mask::TOTAL, 0.0f);
    std::vector<int> idsRef(action_mask::N_ENEMY_SLOTS, 0);

    std::vector<AgentAction> heroActs(kNumEnvs);
    std::vector<AgentAction> oppActs(kNumEnvs);
    for (int i = 0; i < kNumEnvs; ++i) {
        heroActs[i].moveX = 1.0f;
        heroActs[i].attackTargetId = VecSimulation::OPPONENT_ID;
        oppActs[i].moveX = -1.0f;
        oppActs[i].attackTargetId = VecSimulation::HERO_ID;
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < 50; ++step) {
        vecSim.observe(VecSimulation::HERO_ID, obsPar.data());
        vecSim.actionMask(VecSimulation::HERO_ID, maskPar.data(), idsPar.data());
        vecSim.applyActions(VecSimulation::HERO_ID, heroActs.data());
        vecSim.applyActions(VecSimulation::OPPONENT_ID, oppActs.data());
        vecSim.step();
        vecSim.resetDone();
    }
    auto t1 = std::chrono::steady_clock::now();

    for (int step = 0; step < 50; ++step) {
        for (int i = 0; i < kNumEnvs; ++i) {
            refSims[i]->observe(VecSimulation::HERO_ID, obsRef.data());
            refSims[i]->actionMask(VecSimulation::HERO_ID, maskRef.data(), idsRef.data());
            refSims[i]->applyActions(VecSimulation::HERO_ID, &heroActs[i]);
            refSims[i]->applyActions(VecSimulation::OPPONENT_ID, &oppActs[i]);
            refSims[i]->step();
            refSims[i]->resetDone();
        }
    }

    vecSim.observe(VecSimulation::HERO_ID, obsPar.data());
    vecSim.actionMask(VecSimulation::HERO_ID, maskPar.data(), idsPar.data());
    for (int i = 0; i < kNumEnvs; ++i) {
        refSims[i]->observe(VecSimulation::HERO_ID, obsRef.data());
        refSims[i]->actionMask(VecSimulation::HERO_ID, maskRef.data(), idsRef.data());
        CHECK(std::memcmp(obsPar.data() + i * observation::TOTAL,
                          obsRef.data(),
                          observation::TOTAL * sizeof(float)) == 0);
        CHECK(std::memcmp(maskPar.data() + i * action_mask::TOTAL,
                          maskRef.data(),
                          action_mask::TOTAL * sizeof(float)) == 0);
    }
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("    64-env x 50 steps parallel VecSimulation verified in %.2f ms\n", ms);
}

int main() {
    printf("brogameagent simulation tests\n");
    printf("=============================\n");

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
