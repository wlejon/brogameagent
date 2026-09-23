// Simulation policy-map mutation during step(): add/removePolicy called from
// inside a policy are queued and applied when the outermost step ends, so a
// policy never destroys the function that is running and the step finishes
// with the policy set it started with. Core-only.

#include <brogameagent/brogameagent.h>

#include <cstdio>
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
