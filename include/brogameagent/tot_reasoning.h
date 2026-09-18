#pragma once

#include "brogameagent/generic_mcts.h"

#include <functional>
#include <string>
#include <vector>

namespace brogameagent::reasoning {

struct ThoughtStep {
    std::string text;
    float reward = 0.0f;
    bool isTerminal = false;
};

struct TotState {
    std::string prompt;
    std::vector<ThoughtStep> trajectory;
    int currentDepth = 0;
    bool done = false;
    float cumulativeReward = 0.0f;
};

struct TotSearchConfig {
    int maxDepth = 5;
    int branchingFactor = 3;
    int mctsIterations = 100;
    float cPuct = 1.4f;
    float gamma = 0.95f;
};

using StepGenerator = std::function<ThoughtStep(
    const std::string& prompt,
    const std::vector<ThoughtStep>& trajectory,
    int action)>;

using StateEvaluator = std::function<float(
    const std::string& prompt,
    const std::vector<ThoughtStep>& trajectory)>;

mcts::GenericEnv makeTotEnv(
    TotState& liveState,
    StepGenerator stepGen,
    int branchingFactor,
    int maxDepth);

std::vector<ThoughtStep> searchTreeOfThoughts(
    const std::string& prompt,
    StepGenerator stepGen,
    StateEvaluator critic = nullptr,
    const TotSearchConfig& cfg = {});

} // namespace brogameagent::reasoning
