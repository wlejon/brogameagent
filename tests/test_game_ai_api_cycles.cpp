// Reference cycles through the brogameagent_api binding must be collectable.
//
// A native half that held a JS callback or object in a host root (an
// ev::Persistent) pinned it for as long as the native half lived — and when
// that callback or object refers back to the handle owning the native half,
// the everyday shape `this.mcts = createMcts({ evaluator: w => this.score(w) })`
// is a cycle through a root that no collection can free. Each case here builds
// such a cycle twice: the "dropped" copy must be collected, and the "kept"
// copy must still run its callbacks after the collection.
//
// Every case failed on the binding as it was before the callbacks moved onto
// their owners' JS objects.

#include "api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace ev = bronze::embed;
using Value = bronze::Value;

namespace {

int g_step = 0;

void runJs(const char* label, const char* script) {
    std::cout << "[" << ++g_step << "] " << label << "..." << std::endl;
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::string msg = ev::isObject(res.value)
            ? ev::toUtf8(ev::getProperty(res.value, "message"))
            : ev::toUtf8(res.value);
        std::cerr << label << " threw: " << msg << std::endl;
        std::exit(1);
    }
    if (ev::toUtf8(res.value) != "SUCCESS") {
        std::cerr << label << " returned: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
}

// A full collection with no script frame on the stack: drain the microtask
// checkpoint so WeakRef targets held for the current job are released, and
// run the deferred finalizers (a World's) before collecting again.
void collectNow() {
    ev::drainMicrotasks();
    ev::collectGarbage();
    ev::drainFinalizers();
    ev::collectGarbage();
}

// Each case is a class whose constructor builds the cycle and whose run()
// drives it, counting callback calls in this.calls; `handle` is the native
// handle the cycle goes through.
const char* kSetup = R"JS(
    (function() {
        const G = bro.ai.game;
        const NOOP = { moveDir: 0, attackSlot: -1, abilitySlot: -1 };

        function arena(o) {
            o.world = G.createWorld();
            o.hero = G.createAgent({ id: 1, teamId: 0, x: 0, z: 0, hp: 100, attackRange: 3 });
            o.opp = G.createAgent({ id: 2, teamId: 1, x: 2, z: 0, hp: 100, attackRange: 3 });
            o.world.addAgent(o.hero);
            o.world.addAgent(o.opp);
            o.calls = 0;
        }
        const cfg = { iterations: 8, rolloutHorizon: 2 };

        const cases = {
            classicMcts: class {
                constructor() {
                    arena(this);
                    this.handle = G.createMcts(Object.assign({}, cfg, {
                        rolloutPolicy: () => { this.calls++; return NOOP; },
                        prior: (s, w, acts) => { this.calls++; return acts.map(() => 1); },
                        evaluator: () => { this.calls++; return 0; },
                    }));
                }
                run() { this.handle.search(this.world, this.hero); }
            },
            decoupledMcts: class {
                constructor() {
                    arena(this);
                    this.handle = G.createDecoupledMcts(Object.assign({}, cfg, {
                        rolloutPolicy: () => { this.calls++; return NOOP; },
                        evaluator: () => { this.calls++; return 0; },
                    }));
                }
                run() { this.handle.search(this.world, this.hero, this.opp); }
            },
            teamMcts: class {
                constructor() {
                    arena(this);
                    this.handle = G.createTeamMcts(Object.assign({}, cfg, {
                        evaluator: () => { this.calls++; return 0; },
                    }));
                }
                run() { this.handle.search(this.world, [this.hero]); }
            },
            optionMcts: class {
                constructor() {
                    arena(this);
                    this.opt = G.createOption({
                        name: "hold",
                        canInitiate: () => { this.calls++; return true; },
                        step: () => { this.calls++; return NOOP; },
                        shouldTerminate: (s, w, t) => t >= 1,
                    });
                    this.handle = G.createOptionMcts(Object.assign({}, cfg, {
                        options: [this.opt],
                        evaluator: () => { this.calls++; return 0; },
                    }));
                }
                run() { this.handle.search(this.world, this.hero); }
            },
            tacticMcts: class {
                constructor() {
                    arena(this);
                    this.handle = G.createTacticMcts(Object.assign({}, cfg, {
                        evaluator: () => { this.calls++; return 0; },
                    }));
                }
                run() { this.handle.search(this.world, [this.hero]); }
            },
            layeredPlanner: class {
                constructor() {
                    arena(this);
                    this.handle = G.createLayeredPlanner({
                        tactic: cfg, fine: cfg,
                        rolloutPolicy: () => { this.calls++; return NOOP; },
                        evaluator: () => { this.calls++; return 0; },
                    });
                }
                run() { this.handle.decide(this.world, [this.hero]); }
            },
            teamOptionMcts: class {
                constructor() {
                    arena(this);
                    this.opt = G.createTeamOption({
                        name: "push",
                        canInitiate: () => { this.calls++; return true; },
                        step: (h) => { this.calls++; return h.map(() => NOOP); },
                        shouldTerminate: (h, w, t) => t >= 1,
                    });
                    this.handle = G.createTeamOptionMcts(Object.assign({}, cfg, {
                        options: [this.opt],
                        evaluator: () => { this.calls++; return 0; },
                    }));
                }
                run() { this.handle.search(this.world, [this.hero]); }
            },
            commander: class {
                constructor() {
                    arena(this);
                    const opt = G.createOption({
                        name: "hold",
                        canInitiate: () => { this.calls++; return true; },
                        step: () => { this.calls++; return NOOP; },
                        shouldTerminate: (s, w, t) => t >= 1,
                    });
                    this.handle = G.createCommander({
                        roleCfg: cfg,
                        roles: [{ name: "tank", options: [opt],
                                  evaluator: () => { this.calls++; return 0; } }],
                        assign: (heroes) => { this.calls++; return heroes.map(() => 0); },
                    });
                }
                run() { this.handle.decide(this.world, [this.hero]); }
            },
            simulationPolicy: class {
                constructor() {
                    arena(this);
                    this.handle = G.createSimulation(this.world);
                    this.handle.addPolicy(1, (agent, world) => {
                        if (agent !== this.hero || world !== this.world) return NOOP;
                        this.calls++;
                        return { moveX: 0, moveZ: 0 };
                    });
                }
                run() { this.handle.step(1 / 60); }
            },
            worldAbility: class {
                constructor() {
                    arena(this);
                    this.handle = this.world;
                    this.world.registerAbility(7, {
                        cooldown: 0, manaCost: 0, range: 100,
                        fn: (caster, world) => { if (caster === this.hero && world === this.world) this.calls++; },
                    });
                    this.hero.unit.setAbilitySlot(0, 7);
                }
                run() { this.world.resolveAbility(this.hero, 0, 2); }
            },
            worldRoster: class {
                constructor() {
                    arena(this);
                    this.handle = this.world;
                    this.hero.home = this.world;  // agent -> world -> roster -> agent
                }
                run() { if (this.world.findById(1) === this.hero) this.calls++; }
            },
            unitProxy: class {
                constructor() {
                    this.calls = 0;
                    this.handle = G.createAgent({ id: 5, hp: 40 });
                    this.handle.u = this.handle.unit;  // agent -> unit -> agent
                }
                run() { if (this.handle.u.hp === 40) this.calls++; }
            },
            agentBinding: class {
                constructor() {
                    this.calls = 0;
                    this.handle = G.createAgent({ id: 6 });
                    this.handle.b = this.handle.bind();  // agent -> binding -> agent
                }
                run() { if (this.handle.b.agent === this.handle) this.calls++; }
            },
            genericMctsSnapshot: class {
                constructor() {
                    this.s = 0;
                    this.calls = 0;
                    // The snapshot refers back to the env, which owns the search.
                    this.handle = G.createGenericMcts({ env: this, iterations: 16 });
                }
                get numActions() { return 2; }
                snapshot() { this.calls++; return { owner: this, s: this.s }; }
                restore(v) { if (v && v.owner === this) this.s = v.s; }
                step(a) { this.s += a; return { reward: a, done: this.s >= 3 }; }
                legalActions() { return this.s >= 3 ? [] : [0, 1]; }
                observe() { return new Float32Array([this.s]); }
                run() {
                    this.s = 0;
                    const pick = this.handle.search();
                    if (pick !== 0 && pick !== 1) throw new Error("search picked " + pick);
                    this.handle.advanceRoot(pick);  // keep a subtree, and its snapshots
                }
            },
        };

        globalThis.__dropped = [];
        globalThis.__kept = {};
        for (const name of Object.keys(cases)) {
            const dropped = new cases[name]();
            dropped.run();
            if (dropped.calls === 0) return name + ": no callback ran";
            globalThis.__dropped.push({ name, owner: new WeakRef(dropped),
                                        handle: new WeakRef(dropped.handle) });
            const kept = new cases[name]();
            kept.run();
            globalThis.__kept[name] = kept;
        }
        return "SUCCESS";
    })()
)JS";

const char* kCheck = R"JS(
    (function() {
        for (const d of globalThis.__dropped) {
            if (d.owner.deref() !== undefined) return d.name + ": owner leaked";
            if (d.handle.deref() !== undefined) return d.name + ": handle leaked";
        }
        for (const name of Object.keys(globalThis.__kept)) {
            const kept = globalThis.__kept[name];
            kept.calls = 0;
            kept.run();
            if (kept.calls === 0) return name + ": callbacks lost after a collection";
        }
        return "SUCCESS";
    })()
)JS";

} // namespace

int main() {
    std::cout << "Running brogameagent API reference-cycle test..." << std::endl;

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
        brogameagent::api::installGameAi();

        runJs("build cycles", kSetup);
        collectNow();
        runJs("dropped cycles collected, kept ones still run", kCheck);
    }
    ev::destroyRealm(realm);

    std::cout << "All brogameagent API reference-cycle tests passed!" << std::endl;
    return 0;
}
