#include "brogameagent/avoidance.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace brogameagent {

using bromath::Vec2;
using bromath::vcross;
using bromath::vdot;
using bromath::vlen;
using bromath::vlen2;
using bromath::vnorm;
using bromath::vnormOr;

namespace {

constexpr float kEps = 1e-5f;
constexpr float kInf = std::numeric_limits<float>::infinity();

/// Left-perpendicular of a unit vector: rotate +90 degrees CCW.
inline Vec2 perpLeft(Vec2 v) { return {-v.y, v.x}; }
/// Right-perpendicular: rotate -90 degrees (CW).
inline Vec2 perpRight(Vec2 v) { return {v.y, -v.x}; }

/// Squared distance from point p to segment [a, b].
float distSqPointSegment(Vec2 a, Vec2 b, Vec2 p) {
    const Vec2 ab = b - a;
    const float abLen2 = vlen2(ab);
    if (abLen2 <= kEps * kEps) return vlen2(p - a);
    float t = vdot(p - a, ab) / abLen2;
    t = std::clamp(t, 0.0f, 1.0f);
    return vlen2(p - (a + t * ab));
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Agent / obstacle management
// ═══════════════════════════════════════════════════════════════════════════

int AvoidanceSim::addAgent(Vec2 position, const AvoidanceAgentParams& params) {
    Slot s;
    s.position = position;
    s.params = params;
    agents_.push_back(s);
    return (int)agents_.size() - 1;
}

void AvoidanceSim::clearAgents() { agents_.clear(); }

bool AvoidanceSim::addObstacle(const std::vector<Vec2>& vertices) {
    const int n = (int)vertices.size();
    if (n < 2) return false;
    // Reject degenerate edges up front — a zero-length unitDir would poison
    // every downstream projection.
    for (int i = 0; i < n; i++) {
        const Vec2 d = vertices[(size_t)((i + 1) % n)] - vertices[(size_t)i];
        if (vlen2(d) <= kEps * kEps) return false;
    }

    const int base = (int)obstacles_.size();
    for (int i = 0; i < n; i++) {
        ObstacleVertex v;
        v.point = vertices[(size_t)i];
        v.next = base + (i + 1) % n;
        v.prev = base + (i + n - 1) % n;
        v.unitDir = vnorm(vertices[(size_t)((i + 1) % n)] - vertices[(size_t)i]);
        if (n == 2) {
            // An isolated wall segment: both endpoints act as convex corners.
            v.convex = true;
        } else {
            // CCW polygon: a vertex is convex iff the boundary turns left
            // (or goes straight) through it.
            const Vec2& prevPt = vertices[(size_t)((i + n - 1) % n)];
            const Vec2& nextPt = vertices[(size_t)((i + 1) % n)];
            v.convex = vcross(v.point - prevPt, nextPt - v.point) >= 0.0f;
        }
        obstacles_.push_back(v);
    }
    return true;
}

bool AvoidanceSim::addObstacleSegment(Vec2 a, Vec2 b) {
    return addObstacle({a, b});
}

bool AvoidanceSim::addObstacleBox(const AABB& box) {
    // CCW in the (x, z) plane.
    return addObstacle({
        {box.cx - box.hw, box.cz - box.hd},
        {box.cx + box.hw, box.cz - box.hd},
        {box.cx + box.hw, box.cz + box.hd},
        {box.cx - box.hw, box.cz + box.hd},
    });
}

void AvoidanceSim::clearObstacles() { obstacles_.clear(); }

// ═══════════════════════════════════════════════════════════════════════════
// Neighbor queries
// ═══════════════════════════════════════════════════════════════════════════

namespace {
inline uint64_t cellKey(int cx, int cz) {
    return (uint64_t)(uint32_t)cx << 32 | (uint32_t)cz;
}
} // namespace

void AvoidanceSim::rebuildGrid_() {
    grid_.clear();
    float maxRange = 1.0f;
    for (const Slot& s : agents_)
        maxRange = std::max(maxRange, s.params.neighborDist);
    gridCell_ = maxRange;
    const float inv = 1.0f / gridCell_;
    for (int i = 0; i < (int)agents_.size(); i++) {
        const Vec2 p = agents_[(size_t)i].position;
        const int cx = (int)std::floor(p.x * inv);
        const int cz = (int)std::floor(p.y * inv);
        grid_[cellKey(cx, cz)].push_back(i);
    }
}

void AvoidanceSim::gatherNeighbors_(int i, std::vector<std::pair<float, int>>& out) const {
    out.clear();
    const Slot& self = agents_[(size_t)i];
    const float range = self.params.neighborDist;
    const float rangeSq = range * range;
    const float inv = 1.0f / gridCell_;
    const int cx = (int)std::floor(self.position.x * inv);
    const int cz = (int)std::floor(self.position.y * inv);
    // gridCell_ >= any neighborDist, so the 3x3 block covers the query disc.
    for (int dz = -1; dz <= 1; dz++) {
        for (int dx = -1; dx <= 1; dx++) {
            auto it = grid_.find(cellKey(cx + dx, cz + dz));
            if (it == grid_.end()) continue;
            for (int j : it->second) {
                if (j == i) continue;
                const Slot& other = agents_[(size_t)j];
                // Layer filter: self only avoids neighbors whose layers
                // intersect its mask. One-sided by design — the neighbor
                // still avoids self when ITS mask matches self's layers.
                if ((self.params.mask & other.params.layers) == 0) continue;
                // Elevation filter: agents whose vertical spans don't overlap
                // are on different levels (bridge over tunnel, stacked
                // floors) and must not steer around each other. Default
                // elevations are all 0, so single-level embedders are
                // unaffected.
                const float dy = std::fabs(other.elevation - self.elevation);
                if (dy > 0.5f * (self.params.height + other.params.height))
                    continue;
                const float d2 = vlen2(other.position - self.position);
                if (d2 <= rangeSq) out.push_back({d2, j});
            }
        }
    }
    // Deterministic nearest-first order; cap at maxNeighbors.
    std::sort(out.begin(), out.end());
    if ((int)out.size() > self.params.maxNeighbors)
        out.resize((size_t)self.params.maxNeighbors);
}

void AvoidanceSim::gatherObstacleEdges_(int i, std::vector<std::pair<float, int>>& out) const {
    out.clear();
    const Slot& self = agents_[(size_t)i];
    const float range = self.params.timeHorizonObst * self.params.maxSpeed + self.params.radius;
    const float rangeSq = range * range;
    for (int v = 0; v < (int)obstacles_.size(); v++) {
        const ObstacleVertex& o = obstacles_[(size_t)v];
        const float d2 = distSqPointSegment(o.point, obstacles_[(size_t)o.next].point,
                                            self.position);
        if (d2 <= rangeSq) out.push_back({d2, v});
    }
    std::sort(out.begin(), out.end());
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear programming (2D LP over half-planes, 3D fallback)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

struct LpLine {
    Vec2 point;
    Vec2 direction;
};

/// Constrain `result` to line `lineNo` (clipped by the speed circle and all
/// lines before it). Returns false if infeasible.
bool linearProgram1(const std::vector<LpLine>& lines, size_t lineNo, float radius,
                    Vec2 optVelocity, bool directionOpt, Vec2& result) {
    const float dotProduct = vdot(lines[lineNo].point, lines[lineNo].direction);
    const float discriminant =
        dotProduct * dotProduct + radius * radius - vlen2(lines[lineNo].point);
    if (discriminant < 0.0f) return false;  // speed circle misses this line

    const float sqrtDiscriminant = std::sqrt(discriminant);
    float tLeft = -dotProduct - sqrtDiscriminant;
    float tRight = -dotProduct + sqrtDiscriminant;

    for (size_t i = 0; i < lineNo; i++) {
        const float denominator = vcross(lines[lineNo].direction, lines[i].direction);
        const float numerator =
            vcross(lines[i].direction, lines[lineNo].point - lines[i].point);
        if (std::fabs(denominator) <= kEps) {
            // Parallel lines: feasible only if lineNo lies on i's good side.
            if (numerator < 0.0f) return false;
            continue;
        }
        const float t = numerator / denominator;
        if (denominator >= 0.0f) tRight = std::min(tRight, t);
        else                     tLeft = std::max(tLeft, t);
        if (tLeft > tRight) return false;
    }

    if (directionOpt) {
        // Optimize direction: take the extreme point along optVelocity.
        if (vdot(optVelocity, lines[lineNo].direction) > 0.0f)
            result = lines[lineNo].point + tRight * lines[lineNo].direction;
        else
            result = lines[lineNo].point + tLeft * lines[lineNo].direction;
    } else {
        // Closest point on the feasible interval to optVelocity.
        const float t = vdot(lines[lineNo].direction, optVelocity - lines[lineNo].point);
        if (t < tLeft)       result = lines[lineNo].point + tLeft * lines[lineNo].direction;
        else if (t > tRight) result = lines[lineNo].point + tRight * lines[lineNo].direction;
        else                 result = lines[lineNo].point + t * lines[lineNo].direction;
    }
    return true;
}

/// Incremental 2D LP: velocity closest to optVelocity inside the speed
/// circle and all half-planes. Returns lines.size() on success, else the
/// index of the first infeasible line (result then holds the best velocity
/// under the lines before it).
size_t linearProgram2(const std::vector<LpLine>& lines, float radius, Vec2 optVelocity,
                      bool directionOpt, Vec2& result) {
    if (directionOpt) {
        result = optVelocity * radius;  // optVelocity is a unit direction
    } else if (vlen2(optVelocity) > radius * radius) {
        result = vnorm(optVelocity) * radius;
    } else {
        result = optVelocity;
    }

    for (size_t i = 0; i < lines.size(); i++) {
        if (vcross(lines[i].direction, lines[i].point - result) > 0.0f) {
            // result violates line i — reoptimize on that line.
            const Vec2 tempResult = result;
            if (!linearProgram1(lines, i, radius, optVelocity, directionOpt, result)) {
                result = tempResult;
                return i;
            }
        }
    }
    return lines.size();
}

/// 3D fallback when the agent-agent half-planes are jointly infeasible:
/// minimize the worst violation (project half-planes outward equally),
/// keeping the first numObstLines obstacle constraints hard.
void linearProgram3(const std::vector<LpLine>& lines, size_t numObstLines, size_t beginLine,
                    float radius, Vec2& result) {
    float distance = 0.0f;
    for (size_t i = beginLine; i < lines.size(); i++) {
        if (vcross(lines[i].direction, lines[i].point - result) <= distance) continue;

        // result violates line i by more than the current worst violation.
        std::vector<LpLine> projLines(lines.begin(),
                                      lines.begin() + (std::ptrdiff_t)numObstLines);
        for (size_t j = numObstLines; j < i; j++) {
            LpLine line;
            const float determinant = vcross(lines[i].direction, lines[j].direction);
            if (std::fabs(determinant) <= kEps) {
                // Parallel.
                if (vdot(lines[i].direction, lines[j].direction) > 0.0f) continue;
                line.point = 0.5f * (lines[i].point + lines[j].point);
            } else {
                line.point = lines[i].point +
                             (vcross(lines[j].direction, lines[i].point - lines[j].point) /
                              determinant) * lines[i].direction;
            }
            line.direction = vnorm(lines[j].direction - lines[i].direction);
            projLines.push_back(line);
        }

        const Vec2 tempResult = result;
        if (linearProgram2(projLines, radius, perpLeft(lines[i].direction), true, result) <
            projLines.size()) {
            // This should in principle not happen: the result is by
            // definition already in the feasible region of this LP. Keep the
            // previous best on numerical failure.
            result = tempResult;
        }
        distance = vcross(lines[i].direction, lines[i].point - result);
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// ORCA line construction + solve
// ═══════════════════════════════════════════════════════════════════════════

bromath::Vec2 AvoidanceSim::solveAgent_(int i, float dt,
                                        std::vector<std::pair<float, int>>& scratch) const {
    const Slot& self = agents_[(size_t)i];
    const float radius = self.params.radius;
    const float radiusSq = radius * radius;
    const Vec2 position = self.position;
    const Vec2 velocity = self.velocity;

    std::vector<LpLine> orcaLines;

    // --- Obstacle ORCA lines -----------------------------------------------
    //
    // Each nearby obstacle edge induces a truncated velocity obstacle: a
    // cut-off segment (the edge scaled by 1/timeHorizonObst, inflated by
    // radius/timeHorizonObst) with two "legs" tangent from the agent past
    // the edge endpoints. The current velocity is projected onto the nearest
    // boundary feature of that region and the ORCA line is placed there.
    // Obstacles take no reciprocal share — the agent does all the avoiding.
    const float invTimeHorizonObst = 1.0f / self.params.timeHorizonObst;

    gatherObstacleEdges_(i, scratch);
    for (const auto& [obstDistSq, edgeIdx] : scratch) {
        (void)obstDistSq;
        int idx1 = edgeIdx;
        int idx2 = obstacles_[(size_t)idx1].next;
        const ObstacleVertex* obstacle1 = &obstacles_[(size_t)idx1];
        const ObstacleVertex* obstacle2 = &obstacles_[(size_t)idx2];

        const Vec2 relativePosition1 = obstacle1->point - position;
        const Vec2 relativePosition2 = obstacle2->point - position;

        // Skip if this edge's velocity obstacle is already fully covered by
        // previously added obstacle lines.
        bool alreadyCovered = false;
        for (const LpLine& l : orcaLines) {
            if (vcross(invTimeHorizonObst * relativePosition1 - l.point, l.direction) -
                        invTimeHorizonObst * radius >= -kEps &&
                vcross(invTimeHorizonObst * relativePosition2 - l.point, l.direction) -
                        invTimeHorizonObst * radius >= -kEps) {
                alreadyCovered = true;
                break;
            }
        }
        if (alreadyCovered) continue;

        const float distSq1 = vlen2(relativePosition1);
        const float distSq2 = vlen2(relativePosition2);
        const Vec2 obstacleVector = obstacle2->point - obstacle1->point;
        const float s = vdot(-relativePosition1, obstacleVector) / vlen2(obstacleVector);
        const float distSqLine = vlen2(-relativePosition1 - s * obstacleVector);

        LpLine line;
        if (s < 0.0f && distSq1 <= radiusSq) {
            // Already overlapping the left endpoint: push straight away.
            if (obstacle1->convex) {
                line.point = {0.0f, 0.0f};
                line.direction = vnorm(perpLeft(relativePosition1));
                orcaLines.push_back(line);
            }
            continue;
        }
        if (s > 1.0f && distSq2 <= radiusSq) {
            // Overlapping the right endpoint — only handle it here if no
            // neighboring edge will (it does when the endpoint is ahead of
            // that edge's direction).
            if (obstacle2->convex && vcross(relativePosition2, obstacle2->unitDir) >= 0.0f) {
                line.point = {0.0f, 0.0f};
                line.direction = vnorm(perpLeft(relativePosition2));
                orcaLines.push_back(line);
            }
            continue;
        }
        if (s >= 0.0f && s <= 1.0f && distSqLine <= radiusSq) {
            // Overlapping the edge interior: forbid all motion into it — but
            // only from the edge's own (right) side. A wall segment is two
            // directed edges; if the far edge also added its line here, an
            // agent grazing the wall would be pinned between "move away from
            // the front" and "move away from the back" (velocity component
            // normal to the wall forced to exactly zero) and could never
            // regain clearance. The twin edge covers its own side.
            if (vcross(relativePosition1, obstacleVector) < 0.0f) {
                line.point = {0.0f, 0.0f};
                line.direction = -obstacle1->unitDir;
                orcaLines.push_back(line);
            }
            continue;
        }

        // No current collision — build the VO legs.
        Vec2 leftLegDirection, rightLegDirection;
        if (s < 0.0f && distSqLine <= radiusSq) {
            // Obliquely viewed past the left endpoint: the VO collapses to
            // that endpoint's cone.
            if (!obstacle1->convex) continue;
            idx2 = idx1;
            obstacle2 = obstacle1;
            const float leg1 = std::sqrt(distSq1 - radiusSq);
            leftLegDirection = Vec2{relativePosition1.x * leg1 - relativePosition1.y * radius,
                                    relativePosition1.x * radius + relativePosition1.y * leg1} /
                               distSq1;
            rightLegDirection = Vec2{relativePosition1.x * leg1 + relativePosition1.y * radius,
                                     -relativePosition1.x * radius + relativePosition1.y * leg1} /
                                distSq1;
        } else if (s > 1.0f && distSqLine <= radiusSq) {
            // Obliquely viewed past the right endpoint.
            if (!obstacle2->convex) continue;
            idx1 = idx2;
            obstacle1 = obstacle2;
            const float leg2 = std::sqrt(distSq2 - radiusSq);
            leftLegDirection = Vec2{relativePosition2.x * leg2 - relativePosition2.y * radius,
                                    relativePosition2.x * radius + relativePosition2.y * leg2} /
                               distSq2;
            rightLegDirection = Vec2{relativePosition2.x * leg2 + relativePosition2.y * radius,
                                     -relativePosition2.x * radius + relativePosition2.y * leg2} /
                                distSq2;
        } else {
            // Usual situation: tangent legs from both endpoints. Non-convex
            // endpoints inherit the edge direction (the corner belongs to
            // the neighboring edge).
            if (obstacle1->convex) {
                const float leg1 = std::sqrt(distSq1 - radiusSq);
                leftLegDirection =
                    Vec2{relativePosition1.x * leg1 - relativePosition1.y * radius,
                         relativePosition1.x * radius + relativePosition1.y * leg1} /
                    distSq1;
            } else {
                leftLegDirection = -obstacle1->unitDir;
            }
            if (obstacle2->convex) {
                const float leg2 = std::sqrt(distSq2 - radiusSq);
                rightLegDirection =
                    Vec2{relativePosition2.x * leg2 + relativePosition2.y * radius,
                         -relativePosition2.x * radius + relativePosition2.y * leg2} /
                    distSq2;
            } else {
                rightLegDirection = obstacle1->unitDir;
            }
        }

        // Legs that would cut into a neighboring edge are replaced by that
        // edge's direction and marked foreign — projections onto a foreign
        // leg are dropped (the neighboring edge adds its own line).
        const ObstacleVertex& leftNeighbor = obstacles_[(size_t)obstacle1->prev];
        bool isLeftLegForeign = false;
        bool isRightLegForeign = false;
        if (obstacle1->convex && vcross(leftLegDirection, -leftNeighbor.unitDir) >= 0.0f) {
            leftLegDirection = -leftNeighbor.unitDir;
            isLeftLegForeign = true;
        }
        if (obstacle2->convex && vcross(rightLegDirection, obstacle2->unitDir) <= 0.0f) {
            rightLegDirection = obstacle2->unitDir;
            isRightLegForeign = true;
        }

        // Project the current velocity on the VO boundary: cut-off centers…
        const Vec2 leftCutoff = invTimeHorizonObst * (obstacle1->point - position);
        const Vec2 rightCutoff = invTimeHorizonObst * (obstacle2->point - position);
        const Vec2 cutoffVec = rightCutoff - leftCutoff;

        const bool singlePoint = (obstacle1 == obstacle2);
        const float t = singlePoint
                            ? 0.5f
                            : vdot(velocity - leftCutoff, cutoffVec) / vlen2(cutoffVec);
        const float tLeft = vdot(velocity - leftCutoff, leftLegDirection);
        const float tRight = vdot(velocity - rightCutoff, rightLegDirection);

        if ((t < 0.0f && tLeft < 0.0f) || (singlePoint && tLeft < 0.0f && tRight < 0.0f)) {
            // Closest to the left cut-off circle.
            const Vec2 unitW = vnormOr(velocity - leftCutoff, {1.0f, 0.0f});
            line.direction = perpRight(unitW);
            line.point = leftCutoff + radius * invTimeHorizonObst * unitW;
            orcaLines.push_back(line);
            continue;
        }
        if (t > 1.0f && tRight < 0.0f) {
            // Closest to the right cut-off circle.
            const Vec2 unitW = vnormOr(velocity - rightCutoff, {1.0f, 0.0f});
            line.direction = perpRight(unitW);
            line.point = rightCutoff + radius * invTimeHorizonObst * unitW;
            orcaLines.push_back(line);
            continue;
        }

        // …then the nearest of cut-off segment, left leg, right leg.
        const float distSqCutoff =
            (t < 0.0f || t > 1.0f || singlePoint)
                ? kInf
                : vlen2(velocity - (leftCutoff + t * cutoffVec));
        const float distSqLeft =
            (tLeft < 0.0f) ? kInf
                           : vlen2(velocity - (leftCutoff + tLeft * leftLegDirection));
        const float distSqRight =
            (tRight < 0.0f) ? kInf
                            : vlen2(velocity - (rightCutoff + tRight * rightLegDirection));

        if (distSqCutoff <= distSqLeft && distSqCutoff <= distSqRight) {
            line.direction = -obstacle1->unitDir;
            line.point = leftCutoff + radius * invTimeHorizonObst * perpLeft(line.direction);
            orcaLines.push_back(line);
            continue;
        }
        if (distSqLeft <= distSqRight) {
            if (isLeftLegForeign) continue;
            line.direction = leftLegDirection;
            line.point = leftCutoff + radius * invTimeHorizonObst * perpLeft(line.direction);
            orcaLines.push_back(line);
            continue;
        }
        if (isRightLegForeign) continue;
        line.direction = -rightLegDirection;
        line.point = rightCutoff + radius * invTimeHorizonObst * perpLeft(line.direction);
        orcaLines.push_back(line);
    }

    const size_t numObstLines = orcaLines.size();

    // --- Agent ORCA lines ----------------------------------------------------
    //
    // One half-plane per neighbor, from the velocity obstacle truncated at
    // timeHorizon. u is the smallest change to the relative velocity that
    // exits the VO; a responsive neighbor takes half of it (reciprocity), a
    // non-responsive one takes none so we take it all.
    const float invTimeHorizon = 1.0f / self.params.timeHorizon;

    gatherNeighbors_(i, scratch);
    for (const auto& [neighborDistSq, j] : scratch) {
        (void)neighborDistSq;
        const Slot& other = agents_[(size_t)j];

        const Vec2 relativePosition = other.position - position;
        const Vec2 relativeVelocity = velocity - other.velocity;
        const float distSq = vlen2(relativePosition);
        const float combinedRadius = radius + other.params.radius;
        const float combinedRadiusSq = combinedRadius * combinedRadius;

        LpLine line;
        Vec2 u;

        if (distSq > combinedRadiusSq) {
            // No current overlap. w = vector from the cut-off disc center to
            // the relative velocity.
            const Vec2 w = relativeVelocity - invTimeHorizon * relativePosition;
            const float wLengthSq = vlen2(w);
            const float dotProduct1 = vdot(w, relativePosition);

            if (dotProduct1 < 0.0f && dotProduct1 * dotProduct1 > combinedRadiusSq * wLengthSq) {
                // Project on the cut-off circle.
                const float wLength = std::sqrt(wLengthSq);
                const Vec2 unitW = w / wLength;
                line.direction = perpRight(unitW);
                u = (combinedRadius * invTimeHorizon - wLength) * unitW;
            } else {
                // Project on the nearer leg of the VO cone.
                const float leg = std::sqrt(distSq - combinedRadiusSq);
                if (vcross(relativePosition, w) > 0.0f) {
                    line.direction =
                        Vec2{relativePosition.x * leg - relativePosition.y * combinedRadius,
                             relativePosition.x * combinedRadius + relativePosition.y * leg} /
                        distSq;
                } else {
                    line.direction =
                        -Vec2{relativePosition.x * leg + relativePosition.y * combinedRadius,
                              -relativePosition.x * combinedRadius + relativePosition.y * leg} /
                        distSq;
                }
                const float dotProduct2 = vdot(relativeVelocity, line.direction);
                u = dotProduct2 * line.direction - relativeVelocity;
            }
        } else {
            // Already overlapping: get out within one time step.
            const float invTimeStep = 1.0f / dt;
            Vec2 w = relativeVelocity - invTimeStep * relativePosition;
            if (vlen2(w) <= kEps * kEps) {
                // Coincident positions and velocities: pick a deterministic
                // separation axis from the pair's indices so the two agents
                // push in opposite directions.
                w = (i < j) ? Vec2{1.0f, 0.0f} : Vec2{-1.0f, 0.0f};
            }
            const float wLength = vlen(w);
            const Vec2 unitW = w / wLength;
            line.direction = perpRight(unitW);
            u = (combinedRadius * invTimeStep - wLength) * unitW;
        }

        // Responsibility share (see AvoidanceAgentParams::priority): a
        // reciprocating pair splits the effort by priority — shares sum to
        // 1, so the pair remains jointly collision-free. A neighbor that
        // will NOT avoid self back (non-responsive, or its mask doesn't
        // match self's layers) takes no share, leaving self the full effort
        // — without this, one-sided layer visibility would under-avoid.
        //
        // The share's meaning depends on which side of the VO the current
        // relative velocity is on. On a collision course (relVel inside the
        // VO; u points out) the share is how much of the CORRECTION self
        // performs — the low-priority agent must correct more. Not (yet) on
        // a collision course (relVel outside; u points toward the VO) the
        // share is how much of the remaining SLACK self may consume — there
        // the high-priority agent gets the slack, so the split flips sign.
        // Without the flip a share-0 agent with low current speed is pinned
        // (it may never accelerate toward a distant neighbor at all).
        // u ⊥ line.direction in every branch, with perpLeft(direction)
        // pointing out of the VO — so the sign of u · perpLeft(direction)
        // distinguishes the two cases. Equal priorities give 0.5 on both
        // sides: bit-identical to the classic solver.
        const bool reciprocates =
            other.responsive && (other.params.mask & self.params.layers) != 0;
        float share;
        if (reciprocates) {
            const float pSelf = std::clamp(self.params.priority, 0.0f, 1.0f);
            const float pOther = std::clamp(other.params.priority, 0.0f, 1.0f);
            const bool onCollisionCourse =
                vdot(u, perpLeft(line.direction)) > 0.0f;
            const float delta = 0.5f * (pOther - pSelf);
            share = std::clamp(0.5f + (onCollisionCourse ? delta : -delta),
                               0.0f, 1.0f);
        } else {
            share = 1.0f;
        }
        line.point = velocity + share * u;
        orcaLines.push_back(line);
    }

    // --- Solve ---------------------------------------------------------------
    //
    // Two deterministic tie-breakers rotate the preferred velocity before the
    // LP; both share one chirality (always CCW) so blocked crowds resolve
    // into the classic ORCA vortex instead of cancelling each other out.
    //
    // 1. Symmetry dither — a tiny index-keyed angle (< 0.3 degrees).
    //    Perfectly mirrored inputs (an exact head-on approach, agents on a
    //    regular ring) put the preferred velocity's projection exactly on
    //    the constraint boundary's axis of symmetry, and mirrored agents
    //    then pick mirrored velocities forever — they stall nose-to-nose
    //    instead of passing. The dither breaks the tie without visibly
    //    bending anyone's path.
    //
    // 2. Congestion bias — textbook ORCA has a second, deeper stall: agents
    //    that converge on one point brake each other to a crawl (the
    //    cut-off-circle projection is a pure "slow down" with no side
    //    preference), freezing into a mutually-blocked cluster. When an
    //    agent is constrained AND its actual progress along the preferred
    //    direction has collapsed, rotate the preferred velocity increasingly
    //    to the side (up to kCongestionBias when fully stalled). The LP then
    //    finds the tangential escape, the cluster starts to rotate, and the
    //    bias fades automatically as soon as the agent moves again. Purely a
    //    function of current state — determinism is unaffected.
    uint32_t h = (uint32_t)i * 2654435761u;
    h ^= h >> 16;
    float angle = (0.25f + 0.75f * (float)(h & 1023u) / 1023.0f) * 0.005f;

    constexpr float kCongestionBias = 0.8f;  // radians at full stall (~46 deg)
    const float prefSpeed = vlen(self.prefVelocity);
    if (!orcaLines.empty() && prefSpeed > 1e-4f) {
        const float speedAlong = vdot(velocity, self.prefVelocity) / prefSpeed;
        float blockage = 1.0f - speedAlong / prefSpeed;
        blockage = std::clamp(blockage, 0.0f, 1.0f);
        angle += blockage * blockage * kCongestionBias;
    }

    const float ca = std::cos(angle);
    const float sa = std::sin(angle);
    const Vec2 prefVelocity{self.prefVelocity.x * ca - self.prefVelocity.y * sa,
                            self.prefVelocity.x * sa + self.prefVelocity.y * ca};

    Vec2 newVelocity;
    const size_t lineFail = linearProgram2(orcaLines, self.params.maxSpeed,
                                           prefVelocity, false, newVelocity);
    if (lineFail < orcaLines.size()) {
        linearProgram3(orcaLines, numObstLines, lineFail, self.params.maxSpeed, newVelocity);
    }
    return newVelocity;
}

void AvoidanceSim::computeNewVelocities(float dt) {
    if (dt <= 0.0f || agents_.empty()) return;
    rebuildGrid_();

    std::vector<std::pair<float, int>> scratch;
    std::vector<Vec2> newVelocities(agents_.size());

    // Solve all agents against the same pre-step snapshot…
    for (int i = 0; i < (int)agents_.size(); i++) {
        newVelocities[(size_t)i] =
            agents_[(size_t)i].responsive ? solveAgent_(i, dt, scratch)
                                          : agents_[(size_t)i].velocity;
    }
    // …then commit, so solve order can't leak into the results.
    for (size_t i = 0; i < agents_.size(); i++) {
        agents_[i].velocity = newVelocities[i];
    }
}

void AvoidanceSim::step(float dt) {
    computeNewVelocities(dt);
    for (Slot& s : agents_) s.position += s.velocity * dt;
}

} // namespace brogameagent
