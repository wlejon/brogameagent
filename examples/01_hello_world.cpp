// 01_hello_world — the smallest possible brogameagent program.
//
// Build a World. Put two Agents on opposing teams. Print the initial state.
// This is the substrate for everything else in the library.

#include "brogameagent/agent.h"
#include "brogameagent/world.h"

#include <cstdio>

using namespace brogameagent;

int main() {
    World world;

    Agent hero;
    hero.unit().id = 1;
    hero.unit().teamId = 0;
    hero.setPosition(-2.0f, 0.0f);
    world.addAgent(&hero);

    Agent enemy;
    enemy.unit().id = 2;
    enemy.unit().teamId = 1;
    enemy.setPosition(2.0f, 0.0f);
    world.addAgent(&enemy);

    std::printf("world has %zu agents\n", world.agents().size());
    for (Agent* a : world.agents()) {
        std::printf("  id=%d team=%d pos=(%.1f, %.1f) hp=%.0f/%.0f\n",
            a->unit().id, a->unit().teamId, a->x(), a->z(),
            a->unit().hp, a->unit().maxHp);
    }

    // Basic spatial queries: find the nearest enemy and allies.
    if (Agent* ne = world.nearestEnemy(hero)) {
        std::printf("nearest enemy of hero: id=%d\n", ne->unit().id);
    }
    std::printf("hero has %zu allies, %zu enemies\n",
        world.alliesOf(hero).size(), world.enemiesOf(hero).size());
    return 0;
}
