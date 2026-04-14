// 03_projectiles — skillshot and homing projectiles.
//
// Demonstrates:
//   - World::spawnProjectile
//   - Projectile modes: Single / Pierce / AoE and targetId != -1 for homing.
//   - World::stepProjectiles + event log attribution.

#include "brogameagent/agent.h"
#include "brogameagent/world.h"

#include <cstdio>

using namespace brogameagent;

namespace {
int count_damage_events(const World& w, int attackerId) {
    int n = 0;
    for (const auto& ev : w.events()) if (ev.attackerId == attackerId) n++;
    return n;
}
} // namespace

int main() {
    World world;

    Agent hero; hero.unit().id = 1; hero.unit().teamId = 0;
    hero.setPosition(-2.0f, 0.0f);
    world.addAgent(&hero);

    Agent enemy; enemy.unit().id = 2; enemy.unit().teamId = 1;
    enemy.unit().hp = 100.0f; enemy.setPosition(2.0f, 0.0f);
    world.addAgent(&enemy);

    // --- Skillshot: fires along a velocity vector, dies on first hit. ---
    Projectile shot{};
    shot.ownerId = hero.unit().id;
    shot.teamId  = hero.unit().teamId;
    shot.x = hero.x(); shot.z = hero.z();
    shot.vx = 20.0f;  shot.vz = 0.0f;   // flying east at 20 u/s
    shot.speed = 20.0f;
    shot.radius = 0.5f;
    shot.damage = 30.0f;
    shot.kind = DamageKind::Physical;
    shot.remainingLife = 1.0f;
    shot.mode = ProjectileMode::Single;
    world.spawnProjectile(shot);

    // Step until it hits.
    const float dt = 0.02f;
    for (int i = 0; i < 50 && !world.projectiles().empty(); i++) {
        world.stepProjectiles(dt);
        world.cullProjectiles();
    }
    std::printf("after skillshot: enemy hp=%.1f, events=%d\n",
        enemy.unit().hp, count_damage_events(world, hero.unit().id));

    // --- Homing shot: targetId set, projectile tracks the target. ---
    enemy.setPosition(0.0f, 5.0f);         // move the target
    Projectile missile{};
    missile.ownerId = hero.unit().id;
    missile.teamId  = hero.unit().teamId;
    missile.x = hero.x(); missile.z = hero.z();
    missile.targetId = enemy.unit().id;     // <-- homing
    missile.speed = 10.0f;
    missile.radius = 0.5f;
    missile.damage = 20.0f;
    missile.remainingLife = 2.0f;
    world.spawnProjectile(missile);

    for (int i = 0; i < 200 && !world.projectiles().empty(); i++) {
        world.stepProjectiles(dt);
        world.cullProjectiles();
    }
    std::printf("after homing shot: enemy hp=%.1f, events=%d\n",
        enemy.unit().hp, count_damage_events(world, hero.unit().id));
    return 0;
}
