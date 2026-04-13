#include "brogameagent/world.h"
#include "brogameagent/agent.h"

#include <algorithm>
#include <cmath>

namespace brogameagent {

void World::addAgent(Agent* a) {
    agents_.push_back(a);
}

void World::removeAgent(const Agent* a) {
    agents_.erase(std::remove(agents_.begin(), agents_.end(), a), agents_.end());
}

void World::addObstacle(const AABB& box) {
    obstacles_.push_back(box);
}

void World::tick(float dt) {
    for (Agent* a : agents_) {
        if (a->unit().alive()) a->update(dt);
        a->unit().tickCooldowns(dt);
    }
}

std::vector<Agent*> World::enemiesOf(const Agent& self) const {
    std::vector<Agent*> out;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId != self.unit().teamId) out.push_back(a);
    }
    return out;
}

std::vector<Agent*> World::alliesOf(const Agent& self) const {
    std::vector<Agent*> out;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) out.push_back(a);
    }
    return out;
}

static float distSq2D(const Agent& a, const Agent& b) {
    float dx = a.x() - b.x();
    float dz = a.z() - b.z();
    return dx * dx + dz * dz;
}

Agent* World::nearestEnemy(const Agent& self) const {
    Agent* best = nullptr;
    float bestD2 = 1e30f;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        float d2 = distSq2D(self, *a);
        if (d2 < bestD2) { bestD2 = d2; best = a; }
    }
    return best;
}

std::vector<Agent*> World::enemiesInRange(const Agent& self, float range) const {
    float r2 = range * range;
    std::vector<std::pair<float, Agent*>> scored;
    for (Agent* a : agents_) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        float d2 = distSq2D(self, *a);
        if (d2 <= r2) scored.push_back({d2, a});
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& p, const auto& q) { return p.first < q.first; });
    std::vector<Agent*> out;
    out.reserve(scored.size());
    for (auto& p : scored) out.push_back(p.second);
    return out;
}

Agent* World::findById(int id) const {
    for (Agent* a : agents_) {
        if (a->unit().id == id) return a;
    }
    return nullptr;
}

} // namespace brogameagent
