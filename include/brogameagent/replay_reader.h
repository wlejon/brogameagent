#pragma once

#include "replay_format.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brogameagent {

/// Loads a .bgar replay file into memory and provides random-access to
/// frames. Simple and allocation-heavy by design — for real-time playback
/// in `bro` you'll want a streaming / mmap reader; this one is for tests,
/// the CLI query tool, and Python analysis.
class ReplayReader {
public:
    struct Frame {
        replay::FrameHeader          header;
        std::vector<replay::AgentState>      agents;
        std::vector<replay::ProjectileState> projectiles;
        std::vector<replay::DamageEventRec>  events;
    };

    /// Open and parse. Returns false if the file is missing, truncated, or
    /// has a bad magic / version mismatch. On failure, `errorMessage()`
    /// returns a human-readable reason.
    bool open(const std::string& path);

    const std::string& errorMessage() const { return error_; }

    const replay::FileHeader& header() const { return header_; }
    const std::vector<replay::AgentStatic>& roster() const { return roster_; }
    size_t frameCount() const { return index_.size(); }
    const std::vector<replay::IndexEntry>& index() const { return index_; }

    /// Materialize a frame by its position in the index (0-based, not stepIdx).
    Frame frame(size_t i) const;

    /// Find the first frame whose stepIdx matches, or SIZE_MAX if none.
    size_t findByStep(uint32_t stepIdx) const;

    // --- Query helpers (built on top of per-frame iteration) ---

    struct TrajectoryPoint {
        uint32_t stepIdx;
        float    elapsed;
        float    x, z;
        float    hp;
        bool     alive;
    };
    /// Per-step (x, z, hp) for one agent across the whole recording.
    /// Frames where the agent is absent are skipped.
    std::vector<TrajectoryPoint> trajectory(int32_t agentId) const;

    /// Total damage each attacker dealt to each target across the recording.
    struct DamageSummary {
        int32_t attackerId;
        int32_t targetId;
        float   totalDamage;
        uint32_t hits;
        uint32_t kills;
    };
    std::vector<DamageSummary> damageSummary() const;

private:
    std::vector<uint8_t> blob_;
    replay::FileHeader header_{};
    std::vector<replay::AgentStatic> roster_;
    std::vector<replay::IndexEntry>  index_;
    std::string error_;

    bool parse_();
};

} // namespace brogameagent
