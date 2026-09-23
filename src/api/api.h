#pragma once

#include "embed/embed.h"
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brogameagent {
class NavMesh;
class Capability;
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

/// The capability JS registered under `name` with
/// bro.ai.game.registerCapability, as a fresh Capability to add to a
/// binding's CapabilitySet — or null when no such name is registered on the
/// calling thread. Its gate/start/advance/cancel call the spec's JS
/// functions, so it must only be driven on the thread that registered it.
/// The instance holds no JS root: it finds the spec by name on each call.
std::unique_ptr<brogameagent::Capability> makeRegisteredCapability(std::string_view name);

/// The id registerCapability assigned to `name`, or -1 when it is not
/// registered on the calling thread. This is the Action::capId a host sets
/// when think() picks the capability (self.useCapability(name, ...)).
int registeredCapabilityId(std::string_view name);

} // namespace brogameagent::api

using brogameagent::api::installGameAi;
using brogameagent::api::setNavMeshHooks;
using brogameagent::api::getNavMeshHooks;
using brogameagent::api::NavMeshHooks;
