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

// ---- Access -----------------------------------------------------------------
//
// Which files a program may touch is the host's decision, not brogameagent's:
// the natives that open a file the app names ask the host's one access gate,
// bronze::embed::checkAccess (embed/embed_access.h), with the path as it will
// be opened (relative to the process's working directory; there is no path
// resolver) made absolute and lexically normal (UTF-8), on the JS thread that
// made the call, before the file is opened. With no gate installed every path
// is allowed and nothing changes.
//
///   AIRecorder.open, GenericGridRecorder.open      the path Write (writeRoster
///                                                  / recordFrame / close write
///                                                  the file open allowed)
///   AIReplayReader.open, GenericReplayReader.open  the path Read
///   createGridTrainer's ckpt.dir                   the directory Write, once
///                                                  (the trainer's thread writes
///                                                  its checkpoints there)
//
// NavMesh save / loadNavMesh and the nets' save / load take and give bytes and
// ask nothing. A refusal is bronze::embed::newAccessError, thrown: an Error
// whose `message` is the gate's sentence (named PermissionRevoked for a grant
// taken back), `code` is "EACCES" and `path` the absolute path asked about.
// Nothing is opened, written or created.

} // namespace brogameagent::api

using brogameagent::api::installGameAi;
using brogameagent::api::setNavMeshHooks;
using brogameagent::api::getNavMeshHooks;
using brogameagent::api::NavMeshHooks;
