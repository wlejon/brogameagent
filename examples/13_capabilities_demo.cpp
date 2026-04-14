// 13_capabilities_demo — tower + minion + hero driven by capability sets.
//
// Demonstrates:
//   - CapabilitySet as the "tools" menu for an object.
//   - ScriptedMinionPolicy picking between basic_attack and lane_walk.
//   - A tower with only {basic_attack, hold} — no movement.
//   - A trivial hand-written hero policy that kites (flee below 40% HP).
//
// Run with a think rate of 10 Hz; simulate 20 seconds at 60 Hz.

#include "brogameagent/brogameagent.h"

#include <cstdio>
#include <memory>

using namespace brogameagent;

namespace {

struct Binding {
    Agent* agent = nullptr;
    CapabilitySet set;
    std::unique_ptr<Policy> policy;         // optional — hero uses a lambda instead
    Action current;                         // action in flight
    float thinkHz  = 10.0f;
    float accum    = 0.0f;
    // Hero-style inline policy, called if policy == nullptr.
    std::function<bool(const CapContext&, const CapabilitySet&, Action&)> thinkFn;
};

void stepBinding(Binding& b, World& w, float dt, float nowSec) {
    if (!b.agent || !b.agent->unit().alive()) return;

    CapContext ctx;
    ctx.self = b.agent;
    ctx.unit = &b.agent->unit();
    ctx.world = &w;
    ctx.caps = &b.set;
    ctx.now = nowSec;

    // Advance the in-flight action.
    if (!b.current.done && b.current.capId != kCapNone) {
        if (auto* cap = b.set.get(b.current.capId)) {
            cap->advance(ctx, b.current, dt);
        } else {
            b.current.done = true;
        }
    }

    // Think tick.
    b.accum += dt;
    const float gap = 1.0f / b.thinkHz;
    if (b.current.done && b.accum >= gap) {
        b.accum -= gap;
        Action next{};
        bool chosen = false;
        if (b.thinkFn)      chosen = b.thinkFn(ctx, b.set, next);
        else if (b.policy)  chosen = b.policy->decide(ctx, b.set, next);
        if (!chosen || next.capId == kCapNone) next.capId = kCapHold;
        if (auto* cap = b.set.get(next.capId)) {
            cap->start(ctx, next);
            b.current = next;
        }
    }
}

} // namespace

int main() {
    NavGrid grid(-30, -30, 30, 30, 0.5f);

    // Two red minions marching right; a blue tower at (20, 0); a blue hero.
    Agent minionA, minionB, tower, hero;
    minionA.setNavGrid(&grid); minionA.setPosition(-20, -1); minionA.setSpeed(4);
    minionA.unit().id = 10; minionA.unit().teamId = 0;
    minionA.unit().hp = 60; minionA.unit().maxHp = 60;
    minionA.unit().damage = 8; minionA.unit().attackRange = 2.5f;
    minionA.unit().attacksPerSec = 1.0f;

    minionB = minionA;
    minionB.setPosition(-20, 1);
    minionB.unit().id = 11;

    tower.setNavGrid(&grid); tower.setPosition(20, 0); tower.setSpeed(0);
    tower.unit().id = 20; tower.unit().teamId = 1;
    tower.unit().hp = 500; tower.unit().maxHp = 500;
    tower.unit().damage = 40; tower.unit().attackRange = 7;
    tower.unit().attacksPerSec = 0.7f;

    hero.setNavGrid(&grid); hero.setPosition(10, 0); hero.setSpeed(6);
    hero.unit().id = 30; hero.unit().teamId = 1;
    hero.unit().hp = 200; hero.unit().maxHp = 200;
    hero.unit().damage = 25; hero.unit().attackRange = 6;
    hero.unit().attacksPerSec = 1.2f;

    World w;
    w.addAgent(&minionA); w.addAgent(&minionB);
    w.addAgent(&tower);   w.addAgent(&hero);

    Binding bA, bB, bT, bH;
    bA.agent = &minionA; addAllBuiltinCapabilities(bA.set);
    bA.set.setLaneWaypoints({{ -5, -1 }, { 5, -1 }, { 18, -1 }});
    bA.policy = makeScriptedMinionPolicy();
    bA.thinkHz = 10;

    bB.agent = &minionB; addAllBuiltinCapabilities(bB.set);
    bB.set.setLaneWaypoints({{ -5, 1 }, { 5, 1 }, { 18, 1 }});
    bB.policy = makeScriptedMinionPolicy();
    bB.thinkHz = 10;

    // Tower: only basic_attack + hold. No movement options.
    bT.agent = &tower;
    bT.set.add(makeBasicAttackCapability());
    bT.set.add(makeHoldCapability());
    bT.policy = makeScriptedMinionPolicy(); // picks attack-or-hold via same logic
    bT.thinkHz = 5;

    // Hero: all built-ins, kite behaviour via a lambda.
    bH.agent = &hero; addAllBuiltinCapabilities(bH.set);
    bH.thinkHz = 15;
    bH.thinkFn = [&](const CapContext& ctx, const CapabilitySet& caps, Action& out) -> bool {
        // Low HP -> flee.
        if (ctx.unit->hp < 0.4f * ctx.unit->maxHp && caps.has(kCapFlee)) {
            out.capId = kCapFlee; return true;
        }
        // In-range enemy -> attack nearest.
        if (caps.has(kCapBasicAttack) && ctx.unit->attackCooldown <= 0) {
            auto in = ctx.world->enemiesInRange(*ctx.self, ctx.unit->attackRange);
            if (!in.empty()) {
                out.capId = kCapBasicAttack; out.i0 = in.front()->unit().id;
                return true;
            }
        }
        // Otherwise step toward the nearest enemy.
        if (caps.has(kCapMoveTo)) {
            if (Agent* e = ctx.world->nearestEnemy(*ctx.self)) {
                out.capId = kCapMoveTo; out.fx = e->x(); out.fz = e->z();
                return true;
            }
        }
        out.capId = kCapHold; return true;
    };

    const float dt = 1.0f / 60.0f;
    const int   frames = 60 * 20; // 20 seconds
    for (int f = 0; f < frames; f++) {
        const float now = f * dt;
        stepBinding(bA, w, dt, now);
        stepBinding(bB, w, dt, now);
        stepBinding(bT, w, dt, now);
        stepBinding(bH, w, dt, now);
        w.tick(dt);

        if (f % 60 == 0) {
            std::printf("t=%4.1fs  A hp=%.0f x=%.1f  B hp=%.0f x=%.1f  T hp=%.0f  H hp=%.0f x=%.1f\n",
                now,
                minionA.unit().hp, minionA.x(),
                minionB.unit().hp, minionB.x(),
                tower.unit().hp,
                hero.unit().hp, hero.x());
        }
    }
    return 0;
}
