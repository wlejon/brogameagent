#include "brogameagent/tot_reasoning.h"

#include <algorithm>

namespace brogameagent::reasoning {

mcts::GenericEnv makeTotEnv(
    TotState& liveState,
    StepGenerator stepGen,
    int branchingFactor,
    int maxDepth)
{
    mcts::GenericEnv env;
    env.num_actions = std::max(0, branchingFactor);

    env.snapshot_fn = [&liveState]() -> std::any {
        return std::any(liveState);
    };

    env.restore_fn = [&liveState](const std::any& s) {
        liveState = std::any_cast<TotState>(s);
    };

    env.step_fn = [&liveState, stepGen, maxDepth](int action) -> mcts::GenericStepResult {
        ThoughtStep step = stepGen(liveState.prompt, liveState.trajectory, action);
        liveState.trajectory.push_back(step);
        liveState.currentDepth++;
        liveState.cumulativeReward += step.reward;
        bool isDone = step.isTerminal || (liveState.currentDepth >= maxDepth);
        liveState.done = isDone;
        return mcts::GenericStepResult{step.reward, isDone};
    };

    env.legal_actions_fn = [&liveState, branchingFactor]() -> std::vector<int> {
        if (liveState.done || branchingFactor <= 0) {
            return {};
        }
        std::vector<int> actions(static_cast<size_t>(branchingFactor));
        for (int i = 0; i < branchingFactor; ++i) {
            actions[static_cast<size_t>(i)] = i;
        }
        return actions;
    };

    env.observe_fn = [&liveState]() -> std::vector<float> {
        return {
            static_cast<float>(liveState.currentDepth),
            static_cast<float>(liveState.trajectory.size()),
            liveState.cumulativeReward
        };
    };

    return env;
}

std::vector<ThoughtStep> searchTreeOfThoughts(
    const std::string& prompt,
    StepGenerator stepGen,
    StateEvaluator critic,
    const TotSearchConfig& cfg)
{
    TotState liveState;
    liveState.prompt = prompt;
    liveState.trajectory = {};
    liveState.currentDepth = 0;
    liveState.done = false;
    liveState.cumulativeReward = 0.0f;

    if (cfg.branchingFactor <= 0 || cfg.maxDepth <= 0) {
        return liveState.trajectory;
    }

    for (int depth = 0; depth < cfg.maxDepth; ++depth) {
        if (liveState.done) {
            break;
        }

        auto env = makeTotEnv(liveState, stepGen, cfg.branchingFactor, cfg.maxDepth);
        mcts::GenericMcts m(env);
        mcts::GenericMctsConfig mcfg;
        mcfg.iterations = cfg.mctsIterations;
        mcfg.c_puct = cfg.cPuct;
        mcfg.gamma = cfg.gamma;
        mcfg.rollout_depth = std::max(1, cfg.maxDepth - liveState.currentDepth);
        m.set_config(mcfg);

        if (critic) {
            m.set_value_fn([&liveState, critic](const std::vector<float>& /*obs*/) -> float {
                return critic(liveState.prompt, liveState.trajectory);
            });
        }

        int bestAction = m.search();
        if (bestAction < 0) {
            break;
        }

        auto stepResult = env.step_fn(bestAction);
        if (stepResult.done || liveState.done || liveState.currentDepth >= cfg.maxDepth) {
            break;
        }
    }

    return liveState.trajectory;
}

} // namespace brogameagent::reasoning
