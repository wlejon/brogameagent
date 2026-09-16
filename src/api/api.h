#pragma once

#include "embed/embed.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brogameagent {
class NavMesh;
struct AABB;
}

namespace brogameagent::api {

struct NavMeshHooks {
    std::function<bool(bronze::Value configObj, std::vector<float>& xyz, std::vector<uint32_t>& indices, std::string& error)> collectGeometry;
    std::function<bool(bronze::Value configObj, float minX, float maxX, float minZ, float maxZ, std::vector<brogameagent::AABB>& boxes, std::string& error)> collectObstacles;
    std::function<void(const std::shared_ptr<brogameagent::NavMesh>&)> registerNavMeshForPump;
};

void setNavMeshHooks(const NavMeshHooks& hooks);
const NavMeshHooks& getNavMeshHooks();

/// Mounts `bro.ai.game` (NavGrid, NavMesh, Agent, AgentBinding, Perception, Steering, MCTS)
/// into the current Bronze realm.
void installGameAi();

} // namespace brogameagent::api

using brogameagent::api::installGameAi;
using brogameagent::api::setNavMeshHooks;
using brogameagent::api::getNavMeshHooks;
using brogameagent::api::NavMeshHooks;
