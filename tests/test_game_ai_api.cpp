// Standalone test for brogameagent_api, the bronze-runtime JavaScript binding.
// No bro engine, no window: a fresh bronze realm, installGameAi(), then the
// mount points and a NavGrid round trip checked from both the embed API and
// a compiled script.

#include "api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <brogameagent/capability.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace ev = bronze::embed;
using Value = bronze::Value;

#define TEST_CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "CHECK FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

static std::string errorName(Value thrown) {
    return ev::isObject(thrown) ? ev::toUtf8(ev::getProperty(thrown, "name")) : std::string();
}

// Every Value the test keeps across an allocating embed call lives in a
// Persistent (embed.h's GC contract), so the test is itself clean under
// BRONZE_GC_STRESS=1 and a crash there points at the binding.
static ev::Persistent gameNamespace() {
    ev::Persistent bro(ev::globalValue("bro").value);
    ev::Persistent ai(ev::getProperty(bro.get(), "ai"));
    return ev::Persistent(ev::getProperty(ai.get(), "game"));
}

static void test_mounts() {
    std::cout << "[1/3] mount points..." << std::endl;

    ev::GlobalValue broG = ev::globalValue("bro");
    TEST_CHECK(broG.found);
    TEST_CHECK(ev::isObject(broG.value));
    ev::Persistent broP(broG.value);

    ev::Persistent aiP(ev::getProperty(broP.get(), "ai"));
    TEST_CHECK(ev::isObject(aiP.get()));

    ev::Persistent gameP(ev::getProperty(aiP.get(), "game"));
    TEST_CHECK(ev::isObject(gameP.get()));

    // The `AI` alias is the same namespace object.
    ev::GlobalValue aliasG = ev::globalValue("AI");
    TEST_CHECK(aliasG.found);
    TEST_CHECK(ev::isObject(aliasG.value));

    const char* fns[] = {
        "createNavGrid", "createHexNav", "bakeNavMesh", "loadNavMesh",
        "createAgent", "createWorld", "hasLineOfSight", "canSee",
        "computeAim", "computeLeadAim",
    };
    for (const char* name : fns) {
        Value fn = ev::getProperty(gameP.get(), name);
        if (!ev::isFunction(fn)) {
            std::cerr << "missing bro.ai.game." << name << std::endl;
            std::exit(1);
        }
    }

    Value navMeshAvail = ev::getProperty(gameP.get(), "navMeshAvailable");
    TEST_CHECK(ev::isBool(navMeshAvail));

    // Class constructors the namespace installs.
    const char* classes[] = { "AINavGrid", "AIAgent", "AIHexNav", "AIWorld" };
    for (const char* name : classes) {
        ev::GlobalValue c = ev::globalValue(name);
        if (!c.found || !ev::isFunction(c.value)) {
            std::cerr << "missing global constructor " << name << std::endl;
            std::exit(1);
        }
    }
}

static void test_bad_args() {
    std::cout << "[2/3] bad arguments throw TypeError..." << std::endl;

    ev::Persistent gameP = gameNamespace();

    // createNavGrid() with no options object.
    ev::Persistent createNavGrid(ev::getProperty(gameP.get(), "createNavGrid"));
    ev::CallResult r0 = ev::call(createNavGrid.get(), gameP.get(), {});
    TEST_CHECK(r0.thrown);
    TEST_CHECK(errorName(r0.value) == "TypeError");

    // createNavGrid(42): a non-object argument.
    Value num = ev::fromDouble(42.0);
    ev::CallResult r1 = ev::call(createNavGrid.get(), gameP.get(), std::span<const Value>(&num, 1));
    TEST_CHECK(r1.thrown);
    TEST_CHECK(errorName(r1.value) == "TypeError");

    // createHexNav() with no options object.
    ev::Persistent createHexNav(ev::getProperty(gameP.get(), "createHexNav"));
    ev::CallResult r2 = ev::call(createHexNav.get(), gameP.get(), {});
    TEST_CHECK(r2.thrown);
    TEST_CHECK(errorName(r2.value) == "TypeError");

    // loadNavMesh() with no buffer: TypeError when the feature is compiled in,
    // a plain Error explaining the missing feature otherwise. Never a silent
    // null.
    bool navMeshAvail = ev::toBool(ev::getProperty(gameP.get(), "navMeshAvailable"));
    ev::Persistent loadNavMesh(ev::getProperty(gameP.get(), "loadNavMesh"));
    ev::CallResult r3 = ev::call(loadNavMesh.get(), gameP.get(), {});
    TEST_CHECK(r3.thrown);
    TEST_CHECK(errorName(r3.value) == (navMeshAvail ? "TypeError" : "Error"));
}

static void test_navgrid_path() {
    std::cout << "[3/3] NavGrid findPath via bronze eval..." << std::endl;

    const char* script = R"JS(
        (function() {
            if (AI !== bro.ai.game) throw new Error("AI alias is not bro.ai.game");

            const nav = bro.ai.game.createNavGrid({
                minX: -10, minZ: -10, maxX: 10, maxZ: 10,
                cellSize: 0.5,
                obstacles: [{ x: 0, z: 0, hw: 1, hd: 3 }],
                padding: 0.25,
            });
            if (!(nav instanceof AINavGrid)) throw new Error("createNavGrid did not return an AINavGrid");
            if (nav.cellSize !== 0.5) throw new Error("cellSize mismatch: " + nav.cellSize);
            if (nav.width !== 40 || nav.height !== 40) throw new Error("grid dims: " + nav.width + "x" + nav.height);
            if (nav.isWalkable(0, 0)) throw new Error("obstacle cell reported walkable");
            if (!nav.isWalkable(5, 5)) throw new Error("open cell reported blocked");

            // Straight line crosses the obstacle: the path must exist, start
            // and end where asked, and route around the box.
            const path = nav.findPath(-5, 0, 5, 0);
            if (!Array.isArray(path)) throw new Error("findPath did not return an array");
            if (path.length < 2) throw new Error("findPath returned " + path.length + " points");
            if (path.partial !== false) throw new Error("path.partial should be false, got " + path.partial);
            const first = path[0], last = path[path.length - 1];
            if (Math.abs(first.x + 5) > 0.5 || Math.abs(first.z) > 0.5) throw new Error("path start off: " + JSON.stringify(first));
            if (Math.abs(last.x - 5) > 0.5 || Math.abs(last.z) > 0.5) throw new Error("path end off: " + JSON.stringify(last));
            for (const p of path) {
                if (typeof p.x !== "number" || typeof p.z !== "number") throw new Error("waypoint shape: " + JSON.stringify(p));
                if (!nav.isWalkable(p.x, p.z)) throw new Error("waypoint on blocked cell: " + JSON.stringify(p));
            }

            // Goal inside the obstacle clamps to the nearest reachable cell.
            const clamped = nav.findPath(-5, 0, 0, 0);
            if (clamped.partial !== true) throw new Error("blocked goal should give a partial path");
            if (clamped.length < 1) throw new Error("partial path should not be empty");

            // ...unless the caller asked for hard-fail semantics.
            const strict = nav.findPath(-5, 0, 0, 0, { requireFullPath: true });
            if (strict.length !== 0) throw new Error("requireFullPath should return an empty path");

            // A dynamic obstacle blocks a cell that was open.
            nav.addObstacle({ x: 5, z: 5, hw: 0.5, hd: 0.5 });
            if (nav.isWalkable(5, 5)) throw new Error("addObstacle did not block the cell");

            let threw = false;
            try { bro.ai.game.createNavGrid("nope"); } catch (e) { threw = e instanceof TypeError; }
            if (!threw) throw new Error("createNavGrid(string) should throw TypeError");

            return "SUCCESS";
        })()
    )JS";

    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "eval threw: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    TEST_CHECK(ev::toUtf8(res.value) == "SUCCESS");
}

static std::string evalString(const char* script) {
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "eval threw: " << ev::toUtf8(ev::getProperty(res.value, "message")) << std::endl;
        std::exit(1);
    }
    return ev::toUtf8(res.value);
}

// registerCapability stores the spec, and the host that builds bindings gets
// a working Capability for the name through makeRegisteredCapability.
static void test_register_capability() {
    std::cout << "[4/4] registerCapability drives gate/start/advance/cancel..." << std::endl;

    TEST_CHECK(evalString(R"JS(
        (function() {
            const G = bro.ai.game;
            globalThis.capLog = [];
            const spec = {
                ready: false,
                gate() { return this.ready; },
                start(a0, a1) { capLog.push("start " + a0 + " " + a1); },
                advance(dt, elapsed) { capLog.push("adv " + elapsed.toFixed(2)); return elapsed >= 0.2; },
                cancel() { capLog.push("cancel"); },
            };
            globalThis.kiteSpec = spec;
            const id = G.registerCapability("kite", spec);
            if (id < 100) throw new Error("auto id below 100: " + id);
            const again = G.registerCapability("kite", spec);
            if (again !== id) throw new Error("re-registration changed the id");
            if (G.registerCapability("dash", { id: 150 }) !== 150) throw new Error("explicit id");
            const bad = (fn, name, what) => {
                try { fn(); } catch (e) { if (e.name === name) return; throw new Error(what + " threw " + e.name); }
                throw new Error(what + " did not throw");
            };
            bad(() => G.registerCapability("x", { id: 5 }), "RangeError", "id in the built-in range");
            bad(() => G.registerCapability("y", { id: 150 }), "RangeError", "duplicate id");
            bad(() => G.registerCapability("z", { gate: 3 }), "TypeError", "non-function gate");
            bad(() => G.registerCapability("", {}), "TypeError", "empty name");
            return String(id);
        })();
    )JS") != "");

    TEST_CHECK(brogameagent::api::makeRegisteredCapability("nope") == nullptr);
    TEST_CHECK(brogameagent::api::registeredCapabilityId("nope") == -1);
    TEST_CHECK(brogameagent::api::registeredCapabilityId("dash") == 150);

    auto cap = brogameagent::api::makeRegisteredCapability("kite");
    TEST_CHECK(cap != nullptr);
    TEST_CHECK(cap->id() == brogameagent::api::registeredCapabilityId("kite"));
    TEST_CHECK(std::string(cap->name()) == "kite");

    brogameagent::CapContext ctx;
    TEST_CHECK(!cap->gate(ctx));
    evalString("kiteSpec.ready = true; 'ok'");
    TEST_CHECK(cap->gate(ctx));

    brogameagent::Action act;
    act.capId = cap->id();
    act.i0 = 7;
    act.i1 = 9;
    cap->start(ctx, act);
    TEST_CHECK(!act.done);  // an advance keeps it in flight
    cap->advance(ctx, act, 0.1f);
    TEST_CHECK(!act.done);
    cap->advance(ctx, act, 0.1f);
    TEST_CHECK(act.done);
    cap->cancel(ctx, act);
    TEST_CHECK(evalString("capLog.join('|')") == "start 7 9|adv 0.10|adv 0.20|cancel");

    // No advance: the action lasts its duration. A throwing advance ends it.
    evalString(R"JS(
        bro.ai.game.registerCapability("blink", {});
        bro.ai.game.registerCapability("broken", { advance() { throw new Error("boom"); } });
        'ok'
    )JS");
    auto blink = brogameagent::api::makeRegisteredCapability("blink");
    brogameagent::Action b;
    blink->start(ctx, b);
    TEST_CHECK(b.done);
    b = brogameagent::Action{};
    b.dur = 0.15f;
    blink->start(ctx, b);
    TEST_CHECK(!b.done);
    blink->advance(ctx, b, 0.1f);
    TEST_CHECK(!b.done);
    blink->advance(ctx, b, 0.1f);
    TEST_CHECK(b.done);

    auto broken = brogameagent::api::makeRegisteredCapability("broken");
    brogameagent::Action c;
    broken->start(ctx, c);
    TEST_CHECK(!c.done);
    broken->advance(ctx, c, 0.1f);
    TEST_CHECK(c.done);

    // A spec replaced by re-registration takes over existing instances.
    evalString(R"JS(
        bro.ai.game.registerCapability("kite", { gate() { return false; } });
        'ok'
    )JS");
    TEST_CHECK(!cap->gate(ctx));
}

// A policy that adds, replaces or removes policies (its own included) while
// step() runs: the changes land when the step ends, the running function
// keeps running, and nothing it closes over is lost.
static void test_policy_mutation_during_step() {
    std::cout << "[5] Simulation policies mutated from inside a step..." << std::endl;

    const std::string got = evalString(R"JS(
        (function() {
            const G = bro.ai.game;
            const w = G.createWorld();
            w.addAgent(G.createAgent({ id: 1, x: 0, z: 0 }));
            w.addAgent(G.createAgent({ id: 2, x: 3, z: 0 }));
            const sim = G.createSimulation(w);
            const log = [];
            const hold = { moveX: 0, moveZ: 0 };

            sim.addPolicy(1, function first(agent) {
                log.push("first:" + agent.unit.id);
                // Replace myself and add a policy for agent 2 (later in the
                // roster): neither runs until the next step.
                sim.addPolicy(1, function second(a) {
                    log.push("second:" + a.unit.id);
                    sim.removePolicy(1);          // remove myself mid-call
                    sim.removePolicy(2);
                    log.push("second still running");
                    return hold;
                });
                sim.addPolicy(2, function (a) { log.push("two:" + a.unit.id); return hold; });
                return hold;
            });

            sim.step(1 / 60);
            if (log.join("|") !== "first:1") return "step 1: " + log.join("|");
            sim.step(1 / 60);
            if (log.join("|") !== "first:1|second:1|second still running|two:2")
                return "step 2: " + log.join("|");
            sim.step(1 / 60);
            if (log.length !== 4) return "step 3 ran a removed policy: " + log.join("|");
            if (sim.steps !== 3) return "steps: " + sim.steps;

            // Adding back after removal works outside a step.
            let again = 0;
            sim.addPolicy(2, function () { again++; return hold; });
            sim.runSteps(1 / 60, 2);
            if (again !== 2) return "re-added policy ran " + again + " times";
            return "SUCCESS";
        })()
    )JS");
    if (got != "SUCCESS") std::cerr << "policy mutation: " << got << std::endl;
    TEST_CHECK(got == "SUCCESS");
}

// Integer options and arguments go through checkedInt: NaN, ±Infinity and
// values outside the target range throw a RangeError naming the key instead
// of an undefined cast; in-range fractions truncate; absent keys keep their
// defaults.
static void test_integer_range_errors() {
    std::cout << "[6] integer options reject NaN / out-of-range with RangeError..." << std::endl;

    const std::string got = evalString(R"JS(
        (function() {
            const G = bro.ai.game;
            // Returns "" when fn throws a RangeError whose message names
            // `key`, else a description of what happened.
            function rangeErr(label, key, fn) {
                try { fn(); } catch (e) {
                    if (!(e instanceof RangeError)) return label + ": threw " + e.name + ": " + e.message;
                    if (e.message.indexOf(key) < 0) return label + ": message lacks '" + key + "': " + e.message;
                    return "";
                }
                return label + ": did not throw";
            }
            const fails = [];
            function expect(label, key, fn) { const r = rangeErr(label, key, fn); if (r) fails.push(r); }

            expect("agent id NaN", "id", () => G.createAgent({ id: NaN }));
            expect("agent id 2^40", "id", () => G.createAgent({ id: 2 ** 40 }));
            expect("agent teamId -Infinity", "teamId", () => G.createAgent({ teamId: -Infinity }));
            expect("mcts iterations NaN", "iterations", () => G.createMcts({ iterations: NaN }));
            expect("mcts iterations -1", "iterations", () => G.createMcts({ iterations: -1 }));
            expect("mcts budgetMs 1e12", "budgetMs", () => G.createMcts({ budgetMs: 1e12 }));
            expect("mcts actionRepeat 0", "actionRepeat", () => G.createMcts({ actionRepeat: 0 }));
            expect("mcts seed -1", "seed", () => G.createMcts({ seed: -1 }));
            expect("generic numActions", "numActions", () => G.createGenericMcts({
                numActions: 1e12,
                env: { snapshot() {}, restore() {}, step() { return {}; },
                       legalActions() { return []; }, observe() { return []; } } }));
            expect("vecsim numEnvs NaN", "numEnvs", () => G.createVecSimulation({ numEnvs: NaN }));
            expect("hexnav size 1e10", "size", () => G.createHexNav({ size: 1e10 }));
            expect("belief numParticles", "numParticles", () => G.createTeamBelief({ numParticles: 1e9 }));

            const w = G.createWorld();
            const hero = G.createAgent({ id: 3.7, teamId: 0 });
            if (hero.unit.id !== 3) fails.push("fraction should truncate: id " + hero.unit.id);
            w.addAgent(hero);
            expect("findById NaN", "findById", () => w.findById(NaN));
            if (w.findById(3) !== hero) fails.push("findById(3) lost the agent");
            expect("world.seed -1", "seed", () => w.seed(-1));
            w.seed(12345);
            expect("unit.id = Infinity", "id", () => { hero.unit.id = Infinity; });
            if (hero.unit.id !== 3) fails.push("a rejected setter changed the id");
            expect("avoidance maxNeighbors -1", "maxNeighbors",
                   () => hero.setAvoidance({ maxNeighbors: -1 }));
            hero.setAvoidance({ maxNeighbors: 4, layers: -1, mask: 0xFFFFFFFF });

            const sim = G.createSimulation(w);
            expect("runSteps n -5", "runSteps", () => sim.runSteps(1 / 60, -5));
            expect("addPolicy agentId NaN", "addPolicy", () => sim.addPolicy(NaN, () => ({})));

            if (G.grid) {
                expect("obs window colsBehind", "colsBehind",
                       () => G.grid.createObsWindow({ colsBehind: 1e6 }));
                expect("frame stack k NaN", "k",
                       () => G.grid.createFrameStack({ innerDim: 4, k: NaN }));
            }
            if (G.nn) {
                expect("linear dim -1", "createLinear", () => G.nn.createLinear(-1, 4));
                expect("pvn inDim NaN", "inDim",
                       () => G.nn.createPolicyValueNet({ inDim: NaN, numActions: 2, hidden: [4], valueHidden: 4 }));
            }
            return fails.length ? fails.join("\n") : "SUCCESS";
        })()
    )JS");
    if (got != "SUCCESS") std::cerr << "range errors:\n" << got << std::endl;
    TEST_CHECK(got == "SUCCESS");
}

int main() {
    std::cout << "Running brogameagent API test..." << std::endl;

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
        brogameagent::api::installGameAi();
        test_mounts();
        test_bad_args();
        test_navgrid_path();
        test_register_capability();
        test_policy_mutation_during_step();
        test_integer_range_errors();
    }
    ev::destroyRealm(realm);

    std::cout << "All brogameagent API tests passed!" << std::endl;
    return 0;
}
