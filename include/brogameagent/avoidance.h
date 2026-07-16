#pragma once

#include "types.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace brogameagent {

/// Per-agent parameters for the ORCA local-avoidance solver.
struct AvoidanceAgentParams {
    float radius = 0.4f;          // disc radius (world units)
    float maxSpeed = 6.0f;        // speed cap on the solved velocity
    float neighborDist = 10.0f;   // only agents within this range are considered
    int   maxNeighbors = 10;      // nearest-N cap on the neighbor set
    float timeHorizon = 2.0f;     // seconds of mutual lookahead vs other agents
    float timeHorizonObst = 1.0f; // seconds of lookahead vs static obstacles
};

/// 2D (XZ-plane) Optimal Reciprocal Collision Avoidance — the ORCA algorithm
/// of van den Berg, Guy, Lin & Manocha ("Reciprocal n-Body Collision
/// Avoidance"), implemented from the paper's math in this library's style.
///
/// Each agent is a moving disc with a preferred velocity (what its planner /
/// path follower wants this tick). computeNewVelocities() builds, per agent,
/// one ORCA half-plane per neighbor (from the truncated velocity obstacle,
/// each side taking half the avoidance effort) and per nearby obstacle
/// segment, then solves a small 2D linear program for the admissible velocity
/// closest to the preferred one — falling back to a 3D LP that minimizes the
/// worst constraint violation when the agent-agent constraints are jointly
/// infeasible (dense crowds). Obstacle constraints are never relaxed.
///
/// Static obstacles are counterclockwise polygons (agents are kept outside);
/// a 2-vertex polygon is a wall segment blocked from both sides, and
/// addObstacleBox() bridges an axis-aligned box (e.g. a NavGrid obstacle)
/// into the CCW 4-corner form.
///
/// Deterministic: identical inputs (same agents in the same insertion order,
/// same obstacles, same dt) produce bit-identical velocities. Neighbor sets
/// are gathered from a uniform grid and ordered by (distance, index).
class AvoidanceSim {
public:
    // --- Agents ------------------------------------------------------------

    /// Add an agent slot; returns its index. Indices are dense and stable
    /// until clearAgents().
    int addAgent(bromath::Vec2 position, const AvoidanceAgentParams& params = {});
    void clearAgents();
    int agentCount() const { return (int)agents_.size(); }

    void setPosition(int i, bromath::Vec2 p)     { agents_[(size_t)i].position = p; }
    void setVelocity(int i, bromath::Vec2 v)     { agents_[(size_t)i].velocity = v; }
    void setPrefVelocity(int i, bromath::Vec2 v) { agents_[(size_t)i].prefVelocity = v; }
    void setParams(int i, const AvoidanceAgentParams& p) { agents_[(size_t)i].params = p; }

    /// Non-responsive agents are still avoided by everyone else (at full
    /// rather than shared effort, since they won't take their half) but never
    /// have their own velocity solved — velocity(i) stays whatever was set.
    /// Used for agents that keep legacy/scripted movement inside a world that
    /// otherwise runs avoidance.
    void setResponsive(int i, bool responsive) { agents_[(size_t)i].responsive = responsive; }

    bromath::Vec2 position(int i) const { return agents_[(size_t)i].position; }

    /// After computeNewVelocities(): the solved velocity. Before: the value
    /// last set via setVelocity().
    bromath::Vec2 velocity(int i) const { return agents_[(size_t)i].velocity; }

    // --- Static obstacles ---------------------------------------------------

    /// Add a closed polygon obstacle, vertices in counterclockwise order
    /// (agents are kept on the outside). Two vertices make a wall segment
    /// solid from both sides. Vertices closer than ~1e-5 are rejected (the
    /// whole polygon is dropped, returns false).
    bool addObstacle(const std::vector<bromath::Vec2>& vertices);

    /// Wall segment (both sides solid). Adapter over addObstacle().
    bool addObstacleSegment(bromath::Vec2 a, bromath::Vec2 b);

    /// Axis-aligned box (center + half extents) as a CCW 4-corner polygon —
    /// the natural bridge from NavGrid / World AABB obstacles.
    bool addObstacleBox(const AABB& box);

    void clearObstacles();
    int obstacleVertexCount() const { return (int)obstacles_.size(); }

    // --- Step ---------------------------------------------------------------

    /// ORCA solve: for every responsive agent, compute the velocity closest
    /// to its prefVelocity that is (reciprocally) collision-free against
    /// neighbors for timeHorizon seconds and obstacles for timeHorizonObst
    /// seconds, capped at maxSpeed. Results land in velocity(i). dt is the
    /// integration step the caller will use (it bounds the push-apart rate
    /// for already-overlapping agents).
    void computeNewVelocities(float dt);

    /// computeNewVelocities() + advance every agent's position by
    /// velocity * dt. Standalone use; embedders that own agent positions
    /// (e.g. World) call computeNewVelocities() and integrate themselves.
    void step(float dt);

private:
    struct Slot {
        bromath::Vec2 position;
        bromath::Vec2 velocity;
        bromath::Vec2 prefVelocity;
        AvoidanceAgentParams params;
        bool responsive = true;
    };

    /// One vertex of an obstacle polygon; edge i runs point → vertex[next].
    struct ObstacleVertex {
        bromath::Vec2 point;
        bromath::Vec2 unitDir;  // toward the next vertex
        bool convex = true;
        int next = -1;
        int prev = -1;
    };

    void rebuildGrid_();
    void gatherNeighbors_(int i, std::vector<std::pair<float, int>>& out) const;
    void gatherObstacleEdges_(int i, std::vector<std::pair<float, int>>& out) const;
    bromath::Vec2 solveAgent_(int i, float dt,
                              std::vector<std::pair<float, int>>& scratch) const;

    std::vector<Slot> agents_;
    std::vector<ObstacleVertex> obstacles_;

    // Uniform grid over agent positions, rebuilt each computeNewVelocities().
    // Cell size = max neighborDist so any neighbor is in the 3x3 block.
    float gridCell_ = 1.0f;
    std::unordered_map<uint64_t, std::vector<int>> grid_;
};

} // namespace brogameagent
