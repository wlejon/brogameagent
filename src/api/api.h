#pragma once

#include "embed/embed.h"

namespace brogameagent::api {

/// Mounts `bro.ai.game` (NavGrid, NavMesh, Agent, AgentBinding, Perception, Steering, MCTS)
/// into the current Bronze realm.
void installGameAi();

} // namespace brogameagent::api

using brogameagent::api::installGameAi;
