#include "brogameagent/action_mask.h"
#include "brogameagent/agent.h"
#include "brogameagent/world.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace brogameagent::action_mask {

void build(const Agent& self, const World& world,
           float* outMask, int* outEnemyIds)
{
    std::memset(outMask, 0, sizeof(float) * TOTAL);
    for (int i = 0; i < N_ENEMY_SLOTS; i++) outEnemyIds[i] = -1;

    // --- Enemy slots: same ordering as observation::build (nearest-first) ---
    struct Scored { float d2; Agent* a; };
    std::vector<Scored> enemies;
    enemies.reserve(world.agents().size());
    for (Agent* a : world.agents()) {
        if (a == &self) continue;
        if (!a->unit().alive()) continue;
        if (a->unit().teamId == self.unit().teamId) continue;
        float dx = a->x() - self.x();
        float dz = a->z() - self.z();
        enemies.push_back({dx * dx + dz * dz, a});
    }
    std::sort(enemies.begin(), enemies.end(),
              [](const Scored& p, const Scored& q) { return p.d2 < q.d2; });

    float r = self.unit().attackRange;
    float r2 = r * r;
    bool attackReady = self.unit().alive() && self.unit().attackCooldown <= 0.0f;

    int n = std::min(static_cast<int>(enemies.size()), N_ENEMY_SLOTS);
    for (int k = 0; k < n; k++) {
        outEnemyIds[k] = enemies[k].a->unit().id;
        outMask[k] = (attackReady && enemies[k].d2 <= r2) ? 1.0f : 0.0f;
    }

    // --- Ability slots ---
    bool alive = self.unit().alive();
    for (int s = 0; s < N_ABILITY_SLOTS; s++) {
        int base = N_ENEMY_SLOTS + s;
        if (!alive) continue;
        int aid = self.unit().abilitySlot[s];
        if (aid < 0) continue;
        const AbilitySpec* spec = world.abilitySpec(aid);
        if (!spec) continue;
        if (self.unit().abilityCooldowns[s] > 0.0f) continue;
        if (self.unit().mana < spec->manaCost) continue;
        outMask[base] = 1.0f;
    }
}

} // namespace brogameagent::action_mask
