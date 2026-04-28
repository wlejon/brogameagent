#pragma once

#include "brogameagent/generic_mcts.h"               // GenericEnv, GenericStepResult
#include "brogameagent/learn/generic_replay_buffer.h" // GenericSituation

#include <algorithm>
#include <any>
#include <functional>
#include <vector>

namespace brogameagent::grid {

// ─── Behavioral-cloning ingestion ─────────────────────────────────────────
//
// Run a hand-coded heuristic policy from each of `starts`, record one
// rollout per start, and turn the records into GenericSituation tuples
// suitable for createGenericReplayBuffer / GenericExItTrainer. Trajectories
// whose total discounted return is below `min_return` are dropped — this
// is the lever for "only learn from competent demonstrations."
//
// Each emitted situation has:
//   - obs          = the observation seen at decision time
//   - policy_target = one-hot at the heuristic's chosen action
//   - action_mask  = the legal mask at decision time (so the value head
//                    isn't trained against actions that weren't even
//                    available)
//   - value_target = clipped per-step discounted return-to-go
//                    (Σ_{k≥t} γ^(k-t) · r_k, then clamped to [-1, 1])
//
// Use this to escape cold-start traps where a freshly-initialized net
// never stumbles into reward and exploration alone can't bootstrap.

using HeuristicPolicyFn = std::function<int(
    const std::vector<float>& obs,
    const std::vector<int>&   legal_actions)>;

struct BCConfig {
    float min_return      = 0.0f;
    int   rollout_horizon = 256;
    float gamma           = 0.99f;
    bool  clip_value      = true;   // clamp value_target to [-1, 1]
};

// Run the heuristic from each snapshot in `starts`, emit a flat vector of
// GenericSituation. Mutates the env (caller is responsible for snapshotting
// any pre-call state they want to preserve).
inline std::vector<learn::GenericSituation>
generate_bc_situations(const mcts::GenericEnv& env,
                       const HeuristicPolicyFn& policy,
                       const std::vector<std::any>& starts,
                       const BCConfig& cfg) {
    std::vector<learn::GenericSituation> out;
    if (env.num_actions <= 0 || !policy) return out;

    struct Step {
        std::vector<float> obs;
        std::vector<int>   legal;
        int                action;
        float              reward;
    };
    std::vector<Step> trace;
    trace.reserve(static_cast<size_t>(cfg.rollout_horizon));

    for (const auto& start : starts) {
        if (!env.restore_fn || !env.step_fn || !env.legal_actions_fn || !env.observe_fn) break;
        env.restore_fn(start);
        trace.clear();

        // Generate one trajectory.
        for (int t = 0; t < cfg.rollout_horizon; ++t) {
            Step s;
            s.obs   = env.observe_fn();
            s.legal = env.legal_actions_fn();
            if (s.legal.empty()) break;
            int a = policy(s.obs, s.legal);
            // Skip silently if the heuristic returns an illegal / OOB
            // action — better to drop the demonstration than poison the
            // buffer with masked-zero policy targets.
            bool legal_ok = false;
            for (int la : s.legal) { if (la == a) { legal_ok = true; break; } }
            if (a < 0 || a >= env.num_actions || !legal_ok) break;
            s.action = a;
            auto sr = env.step_fn(a);
            s.reward = sr.reward;
            trace.push_back(std::move(s));
            if (sr.done) break;
        }

        if (trace.empty()) continue;

        // Compute total discounted return (for the min-return filter) and
        // per-step return-to-go (for value targets).
        std::vector<float> rtg(trace.size(), 0.0f);
        float running = 0.0f;
        for (int i = static_cast<int>(trace.size()) - 1; i >= 0; --i) {
            running = trace[static_cast<size_t>(i)].reward + cfg.gamma * running;
            rtg[static_cast<size_t>(i)] = running;
        }
        float total_return = rtg.front();
        if (total_return < cfg.min_return) continue;

        // Emit one situation per step.
        for (size_t i = 0; i < trace.size(); ++i) {
            const auto& s = trace[i];
            learn::GenericSituation sit;
            sit.obs = s.obs;
            sit.policy_target.assign(static_cast<size_t>(env.num_actions), 0.0f);
            sit.policy_target[static_cast<size_t>(s.action)] = 1.0f;
            sit.action_mask.assign(static_cast<size_t>(env.num_actions), 0.0f);
            for (int la : s.legal) {
                if (la >= 0 && la < env.num_actions)
                    sit.action_mask[static_cast<size_t>(la)] = 1.0f;
            }
            float v = rtg[i];
            if (cfg.clip_value) v = std::clamp(v, -1.0f, 1.0f);
            sit.value_target = v;
            out.push_back(std::move(sit));
        }
    }
    return out;
}

} // namespace brogameagent::grid
