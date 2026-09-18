#include "brogameagent/tot_reasoning.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brogameagent::reasoning;

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
        std::exit(1);
    }
}

#define CHECK(cond) check(cond, #cond, __LINE__)
#define CHECK_NEAR(a, b, eps) check(std::abs((a) - (b)) < (eps), #a " ~= " #b, __LINE__)

// ─── Test 1: Arithmetic countdown / Game of 24 reasoning problem ─────────────
// Target number 24. StepGenerator proposes arithmetic operations (e.g. 12 * 2,
// 8 * 3, 20 + 4, 30 - 6), with reward scoring proximity to target.
TEST(game_of_24_countdown) {
    auto parseResult = [](const std::string& text) -> int {
        auto eqPos = text.rfind('=');
        if (eqPos != std::string::npos) {
            return std::stoi(text.substr(eqPos + 1));
        }
        return 0;
    };

    auto getCurrentVal = [&](const std::vector<ThoughtStep>& trajectory) -> int {
        if (trajectory.empty()) return 2; // initial start value
        return parseResult(trajectory.back().text);
    };

    StepGenerator stepGen = [&](const std::string& /*prompt*/,
                                const std::vector<ThoughtStep>& trajectory,
                                int action) -> ThoughtStep {
        int cur = getCurrentVal(trajectory);
        ThoughtStep step;
        int nextVal = 0;

        if (cur == 2) {
            // Initial step from 2
            if (action == 0) {
                step.text = "2 + 10 = 12";
                nextVal = 12;
            } else if (action == 1) {
                step.text = "2 + 6 = 8";
                nextVal = 8;
            } else if (action == 2) {
                step.text = "2 * 10 = 20";
                nextVal = 20;
            } else {
                step.text = "2 + 28 = 30";
                nextVal = 30;
            }
        } else if (cur == 12) {
            if (action == 0) {
                step.text = "12 * 2 = 24";
                nextVal = 24;
            } else if (action == 1) {
                step.text = "12 + 5 = 17";
                nextVal = 17;
            } else if (action == 2) {
                step.text = "12 - 4 = 8";
                nextVal = 8;
            } else {
                step.text = "12 * 3 = 36";
                nextVal = 36;
            }
        } else if (cur == 8) {
            if (action == 0) {
                step.text = "8 * 3 = 24";
                nextVal = 24;
            } else if (action == 1) {
                step.text = "8 + 4 = 12";
                nextVal = 12;
            } else {
                step.text = "8 - 2 = 6";
                nextVal = 6;
            }
        } else if (cur == 20) {
            if (action == 0) {
                step.text = "20 + 4 = 24";
                nextVal = 24;
            } else {
                step.text = "20 - 5 = 15";
                nextVal = 15;
            }
        } else if (cur == 30) {
            if (action == 0) {
                step.text = "30 - 6 = 24";
                nextVal = 24;
            } else {
                step.text = "30 / 2 = 15";
                nextVal = 15;
            }
        } else {
            // Dead end branch
            step.text = std::to_string(cur) + " - 1 = " + std::to_string(cur - 1);
            nextVal = cur - 1;
        }

        if (nextVal == 24) {
            step.reward = 10.0f;
            step.isTerminal = true;
        } else {
            step.reward = -0.1f * static_cast<float>(std::abs(24 - nextVal));
            step.isTerminal = false;
        }
        return step;
    };

    StateEvaluator critic = [&](const std::string& /*prompt*/,
                                const std::vector<ThoughtStep>& trajectory) -> float {
        if (trajectory.empty()) return 0.0f;
        int val = getCurrentVal(trajectory);
        if (val == 24) return 10.0f;
        return -0.1f * static_cast<float>(std::abs(24 - val));
    };

    TotSearchConfig cfg;
    cfg.maxDepth = 3;
    cfg.branchingFactor = 4;
    cfg.mctsIterations = 100;
    cfg.cPuct = 1.4f;
    cfg.gamma = 0.95f;

    auto solution = searchTreeOfThoughts("Reach target 24", stepGen, critic, cfg);

    CHECK(!solution.empty());
    CHECK(solution.size() <= 3);
    const auto& finalStep = solution.back();
    CHECK(finalStep.isTerminal);
    CHECK(parseResult(finalStep.text) == 24);
    CHECK(finalStep.reward > 5.0f);
}

// ─── Test 2: Deduction problem ───────────────────────────────────────────────
// A 3-step logic puzzle with branching choices (correct branch yields reward 1.0,
// incorrect branch yields -0.5).
TEST(deduction_problem) {
    StepGenerator stepGen = [](const std::string& /*prompt*/,
                               const std::vector<ThoughtStep>& trajectory,
                               int action) -> ThoughtStep {
        ThoughtStep step;
        int depth = static_cast<int>(trajectory.size());

        if (depth == 0) {
            if (action == 0) {
                step.text = "Step 0: Deduce suspect A was at library";
                step.reward = 1.0f;
            } else if (action == 1) {
                step.text = "Step 0: Deduce suspect A was at park";
                step.reward = -0.5f;
            } else {
                step.text = "Step 0: Deduce suspect A was at cinema";
                step.reward = -0.5f;
            }
            step.isTerminal = false;
        } else if (depth == 1) {
            bool prevCorrect = (trajectory[0].text.find("library") != std::string::npos);
            if (prevCorrect && action == 0) {
                step.text = "Step 1: Examine library logbook confirming alibi";
                step.reward = 1.0f;
            } else {
                step.text = "Step 1: Incorrect hypothesis";
                step.reward = -0.5f;
            }
            step.isTerminal = false;
        } else if (depth == 2) {
            bool prevCorrect = (trajectory[1].text.find("logbook") != std::string::npos);
            if (prevCorrect && action == 0) {
                step.text = "Step 2: Conclude suspect B is the culprit";
                step.reward = 1.0f;
            } else {
                step.text = "Step 2: Falsely accuse innocent bystander";
                step.reward = -0.5f;
            }
            step.isTerminal = true;
        }
        return step;
    };

    TotSearchConfig cfg;
    cfg.maxDepth = 3;
    cfg.branchingFactor = 3;
    cfg.mctsIterations = 150;
    cfg.cPuct = 1.4f;
    cfg.gamma = 0.95f;

    auto solution = searchTreeOfThoughts("Deduce culprit", stepGen, nullptr, cfg);

    CHECK(solution.size() == 3);
    CHECK(solution[0].text.find("library") != std::string::npos);
    CHECK_NEAR(solution[0].reward, 1.0f, 1e-4f);
    CHECK(!solution[0].isTerminal);

    CHECK(solution[1].text.find("logbook") != std::string::npos);
    CHECK_NEAR(solution[1].reward, 1.0f, 1e-4f);
    CHECK(!solution[1].isTerminal);

    CHECK(solution[2].text.find("suspect B") != std::string::npos);
    CHECK_NEAR(solution[2].reward, 1.0f, 1e-4f);
    CHECK(solution[2].isTerminal);
}

// ─── Test 3: Snapshot and restore test ───────────────────────────────────────
// Verify that state backtracking works without state leakage between tree rollouts.
TEST(snapshot_and_restore) {
    StepGenerator stepGen = [](const std::string& /*prompt*/,
                               const std::vector<ThoughtStep>& trajectory,
                               int action) -> ThoughtStep {
        ThoughtStep step;
        step.text = "step_d" + std::to_string(trajectory.size()) + "_a" + std::to_string(action);
        step.reward = static_cast<float>(action + 1);
        step.isTerminal = (trajectory.size() >= 2);
        return step;
    };

    TotState state;
    state.prompt = "Testing snapshots";
    state.trajectory = {};
    state.currentDepth = 0;
    state.done = false;
    state.cumulativeReward = 0.0f;

    auto env = makeTotEnv(state, stepGen, 3, 4);

    // Initial state observation
    auto obs0 = env.observe_fn();
    CHECK(obs0.size() == 3);
    CHECK_NEAR(obs0[0], 0.0f, 1e-4f);
    CHECK_NEAR(obs0[1], 0.0f, 1e-4f);
    CHECK_NEAR(obs0[2], 0.0f, 1e-4f);

    // Capture root snapshot
    auto snap0 = env.snapshot_fn();

    // Step 0: action 1
    auto res0 = env.step_fn(1);
    CHECK(!res0.done);
    CHECK_NEAR(res0.reward, 2.0f, 1e-4f);
    CHECK(state.currentDepth == 1);
    CHECK(state.trajectory.size() == 1);
    CHECK(state.trajectory[0].text == "step_d0_a1");
    CHECK_NEAR(state.cumulativeReward, 2.0f, 1e-4f);

    // Capture child snapshot
    auto snap1 = env.snapshot_fn();

    // Step 1: action 2
    auto res1 = env.step_fn(2);
    CHECK(!res1.done);
    CHECK_NEAR(res1.reward, 3.0f, 1e-4f);
    CHECK(state.currentDepth == 2);
    CHECK(state.trajectory.size() == 2);
    CHECK(state.trajectory[1].text == "step_d1_a2");
    CHECK_NEAR(state.cumulativeReward, 5.0f, 1e-4f);

    // Restore to child snapshot (snap1)
    env.restore_fn(snap1);
    CHECK(state.currentDepth == 1);
    CHECK(state.trajectory.size() == 1);
    CHECK(state.trajectory[0].text == "step_d0_a1");
    CHECK_NEAR(state.cumulativeReward, 2.0f, 1e-4f);
    CHECK(!state.done);

    // Step alternative action 0 from snap1
    auto resAlt = env.step_fn(0);
    CHECK(!resAlt.done);
    CHECK_NEAR(resAlt.reward, 1.0f, 1e-4f);
    CHECK(state.currentDepth == 2);
    CHECK(state.trajectory.size() == 2);
    CHECK(state.trajectory[0].text == "step_d0_a1");
    CHECK(state.trajectory[1].text == "step_d1_a0"); // No leakage of "step_d1_a2"
    CHECK_NEAR(state.cumulativeReward, 3.0f, 1e-4f);

    // Restore to root snapshot (snap0)
    env.restore_fn(snap0);
    CHECK(state.currentDepth == 0);
    CHECK(state.trajectory.empty());
    CHECK_NEAR(state.cumulativeReward, 0.0f, 1e-4f);
    CHECK(!state.done);

    // Verify legal actions
    auto legals = env.legal_actions_fn();
    CHECK(legals.size() == 3);
    for (int i = 0; i < 3; ++i) {
        CHECK(legals[i] == i);
    }

    // Set state.done and verify legal actions become empty
    state.done = true;
    CHECK(env.legal_actions_fn().empty());
}

// ─── Test 4: Leakage verification during MCTS rollouts ───────────────────────
// StepGenerator checks that every step received in trajectory is a valid prefix.
TEST(rollout_integrity_no_leakage) {
    StepGenerator stepGen = [](const std::string& /*prompt*/,
                               const std::vector<ThoughtStep>& trajectory,
                               int action) -> ThoughtStep {
        // Verify path consistency: each entry must encode depth and action correctly
        for (size_t i = 0; i < trajectory.size(); ++i) {
            std::string prefix = "d" + std::to_string(i) + "_";
            CHECK(trajectory[i].text.rfind(prefix, 0) == 0);
        }

        ThoughtStep step;
        step.text = "d" + std::to_string(trajectory.size()) + "_a" + std::to_string(action);
        step.reward = (action == 0) ? 1.0f : 0.1f;
        step.isTerminal = (trajectory.size() >= 2);
        return step;
    };

    TotSearchConfig cfg;
    cfg.maxDepth = 3;
    cfg.branchingFactor = 3;
    cfg.mctsIterations = 50;

    auto result = searchTreeOfThoughts("Check leakage", stepGen, nullptr, cfg);
    CHECK(result.size() == 3);
    CHECK(result[0].text == "d0_a0");
    CHECK(result[1].text == "d1_a0");
    CHECK(result[2].text == "d2_a0");
    CHECK(result[2].isTerminal);
}

// ─── Test 5: Edge cases (empty branching, maxDepth 0) ────────────────────────
TEST(edge_cases) {
    StepGenerator dummyGen = [](const std::string&, const std::vector<ThoughtStep>&, int) {
        return ThoughtStep{"dummy", 0.0f, true};
    };

    TotSearchConfig cfgZeroDepth;
    cfgZeroDepth.maxDepth = 0;
    auto resZeroDepth = searchTreeOfThoughts("test", dummyGen, nullptr, cfgZeroDepth);
    CHECK(resZeroDepth.empty());

    TotSearchConfig cfgZeroBranching;
    cfgZeroBranching.branchingFactor = 0;
    auto resZeroBranching = searchTreeOfThoughts("test", dummyGen, nullptr, cfgZeroBranching);
    CHECK(resZeroBranching.empty());
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    printf("brogameagent Tree-of-Thought (ToT) reasoning tests\n");
    printf("==================================================\n");

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
