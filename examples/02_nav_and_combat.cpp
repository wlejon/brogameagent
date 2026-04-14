// 02_nav_and_combat — move an agent around an obstacle, then attack.
//
// Demonstrates:
//   - NavGrid: blocking cells, A* pathing.
//   - Agent scripted mode: setTarget + update(dt) walks along the path.
//   - World::resolveAttack + event log.

#include "brogameagent/agent.h"
#include "brogameagent/nav_grid.h"
#include "brogameagent/world.h"

#include <cstdio>

using namespace brogameagent;

int main() {
    // 10x10 world with a vertical wall of obstacles between (-1..1, -5..5).
    NavGrid grid(-5, -5, 5, 5, /*cellSize*/ 0.5f);
    AABB wall{ -0.5f, -3.0f, 0.5f, 3.0f };
    grid.addObstacle(wall, /*padding*/ 0.5f);

    World world;
    world.addObstacle(wall);

    Agent hero;
    hero.unit().id = 1;
    hero.unit().teamId = 0;
    hero.unit().attackRange = 2.0f;
    hero.unit().damage = 20.0f;
    hero.setPosition(-4.0f, 0.0f);
    hero.setNavGrid(&grid);
    world.addAgent(&hero);

    Agent enemy;
    enemy.unit().id = 2;
    enemy.unit().teamId = 1;
    enemy.unit().hp = 50.0f;
    enemy.setPosition(4.0f, 0.0f);
    world.addAgent(&enemy);

    // Path from hero to enemy. The A* will route around the wall.
    auto path = grid.findPath({ hero.x(), hero.z() }, { enemy.x(), enemy.z() });
    std::printf("path length: %zu waypoints\n", path.size());
    for (const auto& wp : path) std::printf("  -> (%.1f, %.1f)\n", wp.x, wp.z);

    // Drive the hero to enemy.x-1 (stop just in attack range).
    hero.setTarget(enemy.x() - 1.0f, enemy.z());
    const float dt = 0.05f;
    int ticks = 0;
    while (!hero.atTarget() && ticks < 400) {
        world.tick(dt);
        ticks++;
    }
    std::printf("hero reached target in %d ticks at (%.2f, %.2f)\n",
        ticks, hero.x(), hero.z());

    // Fire a single auto-attack.
    bool landed = world.resolveAttack(hero, enemy.unit().id);
    std::printf("attack landed: %s; enemy hp now %.1f\n",
        landed ? "yes" : "no", enemy.unit().hp);

    // The event log records what happened.
    for (const auto& ev : world.events()) {
        std::printf("event: %d -> %d  %.1f dmg  killed=%d\n",
            ev.attackerId, ev.targetId, ev.amount, ev.killed);
    }
    return 0;
}
