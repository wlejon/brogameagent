#pragma once

#include "replay_format.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace brogameagent {

class World;
class Agent;

/// Streaming writer for .bgar replay files.
///
/// Typical loop (matches Simulation's tick order):
///
///     Recorder rec;
///     rec.open("ep0.bgar", episodeId, seed, dt);
///     rec.writeRoster(world.agents());  // after world is populated
///     for (int step = 0; step < N; step++) {
///         sim.step(dt);
///         rec.recordFrame(step, sim.elapsed(), world);
///     }
///     rec.close();
///
/// `recordFrame` auto-captures the damage events emitted since the last
/// recorded frame, so you can call it unconditionally per step without
/// manually slicing `world.events()`. Do NOT call `world.clearEvents()`
/// between `recordFrame` calls or you will lose that window's events.
///
/// On close() the frame index and footer are appended so readers can
/// random-access any frame in O(1).
class Recorder {
public:
    Recorder() = default;
    ~Recorder();
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    /// Open a file for writing. Returns false on I/O error.
    bool open(const std::string& path,
              uint64_t episodeId, uint64_t seed, float dt);

    bool isOpen() const { return file_ != nullptr; }

    /// Write the roster (one AgentStatic per agent). Call once, before any
    /// frames. Agents added after this call will still be recorded per-frame
    /// via AgentState, but they won't appear in the roster.
    void writeRoster(const std::vector<Agent*>& agents);

    /// Record one frame. Captures all live agents, projectiles, and the
    /// slice of `world.events()` that has arrived since the previous call.
    void recordFrame(uint32_t stepIdx, float elapsed, const World& world);

    /// Finalize: write index + footer, close the file. No-op if already
    /// closed. Automatically called by the destructor — but preferred
    /// explicitly so errors surface.
    bool close();

    /// Number of frames written so far.
    size_t frameCount() const { return index_.size(); }

private:
    std::FILE* file_ = nullptr;
    std::vector<replay::IndexEntry> index_;
    size_t lastEventIdx_ = 0;
    bool rosterWritten_ = false;
};

} // namespace brogameagent
