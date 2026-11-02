#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <brogameagent/brogameagent.h>
#include <brogameagent/grid/obs_window.h>
#include <brogameagent/grid/frame_stack.h>
#include <brogameagent/grid/failure_tape.h>
#include <brogameagent/grid/best_crop.h>
#include <brogameagent/grid/shaping.h>
#include <brogameagent/grid/bc_ingest.h>
#include <brogameagent/grid/generic_recorder.h>
#include <brogameagent/grid/harness.h>
#include <brogameagent/nn/factored.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace brogameagent;
using bromath::Vec2;
using bromath::wrapAngle;
using bromath::angleDelta;

struct TestEntry {
    const char* name;
    void (*fn)();
};

std::vector<TestEntry>& registry();

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { Register_##name() { registry().push_back({#name, test_##name}); } } reg_##name; \
    static void test_##name()

inline void check(bool cond, const char* msg, int line) {
    if (!cond) {
        printf("    assertion failed at line %d: %s\n", line, msg);
        throw 0;
    }
}

#define CHECK(cond) check(cond, #cond, __LINE__)
#define CHECK_NEAR(a, b, eps) check(std::abs((a) - (b)) < (eps), #a " ~= " #b, __LINE__)

struct McstScene {
    World world;
    Agent hero;
    Agent enemy;
};

inline std::unique_ptr<McstScene> make_duel_scene(float heroX, float heroZ,
                                                  float enemyX, float enemyZ,
                                                  float attackRange = 2.0f) {
    auto s = std::make_unique<McstScene>();
    s->hero.unit().id      = 1;
    s->hero.unit().teamId  = 0;
    s->hero.unit().hp      = 100.0f;
    s->hero.unit().maxHp   = 100.0f;
    s->hero.unit().damage  = 10.0f;
    s->hero.unit().attackRange  = attackRange;
    s->hero.unit().attacksPerSec = 2.0f;
    s->hero.setPosition(heroX, heroZ);
    s->hero.setMaxAccel(30.0f);
    s->hero.setMaxTurnRate(10.0f);

    s->enemy.unit().id     = 2;
    s->enemy.unit().teamId = 1;
    s->enemy.unit().hp     = 50.0f;
    s->enemy.unit().maxHp  = 50.0f;
    s->enemy.unit().damage = 5.0f;
    s->enemy.unit().attackRange  = attackRange;
    s->enemy.unit().attacksPerSec = 1.0f;
    s->enemy.setPosition(enemyX, enemyZ);
    s->enemy.setMaxAccel(30.0f);
    s->enemy.setMaxTurnRate(10.0f);

    s->world.addAgent(&s->hero);
    s->world.addAgent(&s->enemy);
    s->world.seed(42);
    return s;
}

struct TeamScene {
    World world;
    std::vector<std::unique_ptr<Agent>> heroes;
    std::vector<std::unique_ptr<Agent>> enemies;
};

inline std::unique_ptr<TeamScene> make_team_scene(int num_heroes, int num_enemies,
                                                  float attackRange = 3.0f) {
    auto s = std::make_unique<TeamScene>();
    int next_id = 1;
    for (int i = 0; i < num_heroes; i++) {
        auto a = std::make_unique<Agent>();
        a->unit().id = next_id++;
        a->unit().teamId = 0;
        a->unit().hp = 100.0f;
        a->unit().maxHp = 100.0f;
        a->unit().damage = 10.0f;
        a->unit().attackRange = attackRange;
        a->unit().attacksPerSec = 2.0f;
        a->setPosition(-1.0f + 0.4f * i, 0.3f * i);
        a->setMaxAccel(30.0f);
        a->setMaxTurnRate(10.0f);
        s->world.addAgent(a.get());
        s->heroes.push_back(std::move(a));
    }
    for (int i = 0; i < num_enemies; i++) {
        auto a = std::make_unique<Agent>();
        a->unit().id = next_id++;
        a->unit().teamId = 1;
        a->unit().hp = 40.0f;
        a->unit().maxHp = 40.0f;
        a->unit().damage = 5.0f;
        a->unit().attackRange = attackRange;
        a->unit().attacksPerSec = 1.0f;
        a->setPosition(1.0f + 0.4f * i, 0.3f * i);
        a->setMaxAccel(30.0f);
        a->setMaxTurnRate(10.0f);
        s->world.addAgent(a.get());
        s->enemies.push_back(std::move(a));
    }
    s->world.seed(99);
    return s;
}

inline std::vector<Agent*> raw(const std::vector<std::unique_ptr<Agent>>& v) {
    std::vector<Agent*> out;
    out.reserve(v.size());
    for (const auto& u : v) out.push_back(u.get());
    return out;
}
