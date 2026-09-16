// Standalone test for brogameagent_api, the bronze-runtime JavaScript binding.
// No bro engine, no window: a fresh bronze realm, installGameAi(), then the
// mount points and a NavGrid round trip checked from both the embed API and
// a compiled script.

#include "api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

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

static void test_mounts() {
    std::cout << "[1/3] mount points..." << std::endl;

    ev::GlobalValue broG = ev::globalValue("bro");
    TEST_CHECK(broG.found);
    TEST_CHECK(ev::isObject(broG.value));

    Value aiV = ev::getProperty(broG.value, "ai");
    TEST_CHECK(ev::isObject(aiV));

    Value gameV = ev::getProperty(aiV, "game");
    TEST_CHECK(ev::isObject(gameV));

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
        Value fn = ev::getProperty(gameV, name);
        if (!ev::isFunction(fn)) {
            std::cerr << "missing bro.ai.game." << name << std::endl;
            std::exit(1);
        }
    }

    Value navMeshAvail = ev::getProperty(gameV, "navMeshAvailable");
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

    Value gameV = ev::getProperty(ev::getProperty(ev::globalValue("bro").value, "ai"), "game");

    // createNavGrid() with no options object.
    Value createNavGrid = ev::getProperty(gameV, "createNavGrid");
    ev::CallResult r0 = ev::call(createNavGrid, gameV, {});
    TEST_CHECK(r0.thrown);
    TEST_CHECK(errorName(r0.value) == "TypeError");

    // createNavGrid(42): a non-object argument.
    Value num = ev::fromDouble(42.0);
    ev::CallResult r1 = ev::call(createNavGrid, gameV, std::span<const Value>(&num, 1));
    TEST_CHECK(r1.thrown);
    TEST_CHECK(errorName(r1.value) == "TypeError");

    // createHexNav() with no options object.
    Value createHexNav = ev::getProperty(gameV, "createHexNav");
    ev::CallResult r2 = ev::call(createHexNav, gameV, {});
    TEST_CHECK(r2.thrown);
    TEST_CHECK(errorName(r2.value) == "TypeError");

    // loadNavMesh() with no buffer: TypeError when the feature is compiled in,
    // a plain Error explaining the missing feature otherwise. Never a silent
    // null.
    Value loadNavMesh = ev::getProperty(gameV, "loadNavMesh");
    ev::CallResult r3 = ev::call(loadNavMesh, gameV, {});
    TEST_CHECK(r3.thrown);
    bool navMeshAvail = ev::toBool(ev::getProperty(gameV, "navMeshAvailable"));
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

int main() {
    std::cout << "Running brogameagent API test..." << std::endl;

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
        brogameagent::api::installGameAi();
        test_mounts();
        test_bad_args();
        test_navgrid_path();
    }
    ev::destroyRealm(realm);

    std::cout << "All brogameagent API tests passed!" << std::endl;
    return 0;
}
