// Coverage for the parts of brogameagent_api that were restored after the
// bronze port dropped them: the `field` / `applyAction` / `findById` /
// `registerAbility` / `seed` tail of the classic surface, and the whole
// bro.ai.game.nn / .learn / .grid namespaces.
//
// Every check runs as real JavaScript through bronze so the test exercises
// the same path an app does.

#include "api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#ifdef BROGAMEAGENT_HAS_NN
#include "brotensor/api.h"
#endif

#include <cstdlib>
#include <iostream>
#include <string>

namespace ev = bronze::embed;
using Value = bronze::Value;

namespace {

int g_step = 0;

/// Runs a script that must evaluate to the string "SUCCESS"; anything else
/// (a throw, a wrong value) fails the test with the message the script gave.
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

// ---------------------------------------------------------------------------
// The classic surface: World.findById / registerAbility / seed,
// Agent.applyAction, HexNav.field
// ---------------------------------------------------------------------------

const char* kWorldExtras = R"JS(
    (function() {
        const G = bro.ai.game;
        const w = G.createWorld();
        const a1 = G.createAgent({ id: 1, teamId: 0, x: 0, z: 0, mana: 100 });
        const a2 = G.createAgent({ id: 2, teamId: 1, x: 2, z: 0, hp: 100 });
        w.addAgent(a1);
        w.addAgent(a2);

        // seed() must exist and be callable (it only reseeds the RNG).
        if (typeof w.seed !== "function") return "world.seed is missing";
        w.seed(1234);

        // findById returns the very wrapper that was added, and null for
        // anything not on the roster.
        if (w.findById(1) !== a1) return "findById(1) did not return the first agent";
        if (w.findById(2) !== a2) return "findById(2) did not return the second agent";
        if (w.findById(99) !== null) return "findById(99) should be null";

        // registerAbility: fn(caster, world, targetId), fired from the native
        // cast path via resolveAbility(agent, slot, targetId).
        let seen = null;
        w.registerAbility(7, {
            cooldown: 1, manaCost: 5, range: 10,
            fn: function (caster, world, targetId) {
                seen = { caster: caster, world: world, targetId: targetId };
                world.dealDamage(caster, world.findById(targetId), 25, "magical");
            },
        });
        a1.unit.setAbilitySlot(0, 7);
        const cast = w.resolveAbility(a1, 0, 2);
        if (!cast) return "resolveAbility returned false";
        if (seen === null) return "ability fn never fired";
        if (seen.caster !== a1) return "ability fn caster is not the casting agent";
        if (seen.world !== w) return "ability fn world is not the world";
        if (seen.targetId !== 2) return "ability fn targetId: " + seen.targetId;
        if (a2.unit.hp >= 100) return "dealDamage from the ability fn did nothing";

        // Re-registering the same id replaces the callback.
        let second = 0;
        w.registerAbility(7, { cooldown: 0, manaCost: 0, range: 10,
                               fn: function () { second++; } });
        a1.unit.tickCooldowns(5);
        w.resolveAbility(a1, 0, 2);
        if (second !== 1) return "re-registered ability fn did not replace the old one";

        // A spec that is not an object throws, as it did before.
        let threw = false;
        try { w.registerAbility(8, 42); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) return "registerAbility(id, 42) should throw TypeError";

        return "SUCCESS";
    })()
)JS";

const char* kAgentApplyAction = R"JS(
    (function() {
        const G = bro.ai.game;
        const ag = G.createAgent({ id: 1, x: 0, z: 0, speed: 5, maxAccel: 100 });

        // Continuous control: +X move for a few ticks must move the agent.
        for (let i = 0; i < 20; i++) ag.applyAction({ moveX: 1, moveZ: 0 }, 1 / 30);
        if (!(ag.x > 0.01)) return "applyAction did not move the agent: x=" + ag.x;

        // Aim is written straight through.
        ag.applyAction({ moveX: 0, moveZ: 0, aimYaw: 1.25 }, 1 / 30);
        if (Math.abs(ag.aimYaw - 1.25) > 1e-4) return "aimYaw not applied: " + ag.aimYaw;

        let threw = 0;
        try { ag.applyAction({ moveX: 1 }); } catch (e) { if (e instanceof TypeError) threw++; }
        try { ag.applyAction(5, 0.1); } catch (e) { if (e instanceof TypeError) threw++; }
        if (threw !== 2) return "applyAction bad-argument TypeErrors: " + threw;

        return "SUCCESS";
    })()
)JS";

// Paths the GC-stress audit touched: each one used to hold a raw Value across
// an allocating call, or carried a behavioural bug found along the way.
const char* kBindingFixes = R"JS(
    (function() {
        const G = bro.ai.game;
        const w = G.createWorld();
        const a1 = G.createAgent({ id: 1, teamId: 0, x: 0, z: 0 });
        const a2 = G.createAgent({ id: 2, teamId: 1, x: 3, z: 0 });
        const a3 = G.createAgent({ id: 3, teamId: 0, x: 1, z: 0 });
        w.addAgent(a1); w.addAgent(a2); w.addAgent(a3);

        const en = w.enemiesInRange(a1, 10);
        if (en.length !== 1 || en[0] !== a2) return "enemiesInRange: " + en.length;
        const al = w.alliesInRange(a1, 10);
        if (al.length !== 1 || al[0] !== a3) return "alliesInRange: " + al.length;

        // agent.unit answers a live view each read, not a cached one.
        const u1 = a1.unit, u2 = a1.unit;
        if (u1.id !== 1 || u2.id !== 1) return "agent.unit id";

        // snapshot / restore round-trips positions.
        const snap = w.snapshot();
        a1.x = 7;
        w.restore(snap);
        if (Math.abs(a1.x) > 1e-6) return "restore did not put a1 back: " + a1.x;

        // Re-adding a policy replaces the old one.
        const sim = G.createSimulation(w);
        let first = 0, second = 0;
        sim.addPolicy(1, function () { first++; return { moveX: 0, moveZ: 0 }; });
        sim.step(1 / 60);
        sim.addPolicy(1, function () { second++; return { moveX: 0, moveZ: 0 }; });
        sim.step(1 / 60);
        if (first !== 1 || second !== 1) return "addPolicy replace: " + first + "/" + second;
        sim.removePolicy(1);
        sim.step(1 / 60);
        if (second !== 1) return "removePolicy still ran the policy";

        const opt = G.createOption({
            name: "hold",
            canInitiate: () => true,
            step: () => ({ moveDir: 0, attackSlot: -1, abilitySlot: -1 }),
            shouldTerminate: (_s, _w, t) => t >= 1,
        });
        if (opt.name !== "hold") return "createOption name: " + opt.name;
        const topt = G.createTeamOption({
            name: "push",
            canInitiate: () => true,
            step: (h) => h.map(() => ({ moveDir: 0, attackSlot: -1, abilitySlot: -1 })),
            shouldTerminate: (_h, _w, t) => t >= 1,
        });
        if (topt.name !== "push") return "createTeamOption name: " + topt.name;

        let threw = false;
        if (G.navMeshAvailable) {
            const positions = [-10,0,-10,  10,0,-10,  10,0,10,  -10,0,10];
            const indices = [0,2,1, 0,3,2];
            const m = G.bakeNavMesh({ positions, indices, regionMinSize: 2,
                                      regionMergeSize: 10, edgeMaxLen: 8,
                                      edgeMaxError: 1.1, detailSampleDist: 4,
                                      detailSampleMaxError: 0.5 });
            if (!m.findPath({x:-5,y:0,z:-5}, {x:5,y:0,z:5})) return "static bake: no path";

            threw = false;
            try { G.bakeNavMesh({ positions, indices, offMeshLinks: 5 }); }
            catch (e) { threw = e instanceof TypeError; }
            if (!threw) return "offMeshLinks: 5 should throw TypeError";

            const linked = G.bakeNavMesh({ positions, indices, offMeshLinks: [
                { start: {x:-5,y:0,z:0}, end: {x:5,y:0,z:0}, radius: 0.5 } ] });
            if (!linked.findPath({x:-5,y:0,z:-5}, {x:5,y:0,z:5})) return "linked bake: no path";

            const dyn = G.bakeNavMesh({ positions, indices, dynamicObstacles: true,
                                        tileSize: 8, maxObstacles: 4 });
            if (!dyn.supportsObstacles) return "dynamicObstacles ignored";
            const h1 = dyn.addObstacle({ type: "box", min: {x:-1,y:0,z:-1}, max: {x:1,y:2,z:1} });
            const h2 = dyn.addObstacle({ type: "box", center: {x:4,y:1,z:4},
                                         halfExtents: {x:1,y:1,z:1}, yaw: 0.3 });
            while (!dyn.update()) {}
            if (dyn.obstacleCount !== 2) return "obstacleCount: " + dyn.obstacleCount;
            if (!dyn.removeObstacle(h1) || !dyn.removeObstacle(h2)) return "removeObstacle";
            while (!dyn.update()) {}
        }
        return "SUCCESS";
    })()
)JS";

const char* kHexNavField = R"JS(
    (function() {
        const G = bro.ai.game;
        const size = 8;
        const cells = size * size;
        const nav = G.createHexNav({ size: size });

        // A uniform cost-1 table over all six hex directions.
        const costs = new Float64Array(cells * 6);
        costs.fill(1);
        nav.setStepCosts("t", costs);

        if (typeof nav.field !== "function") return "hexNav.field is missing";

        const seeds = new Int32Array([0]);
        const res = nav.field("t", { seeds: seeds, quantum: 0.25, ring: 64 });
        if (!res || !(res.dist instanceof Float64Array)) return "field().dist is not a Float64Array";
        if (!(res.parent instanceof Int32Array)) return "field().parent is not an Int32Array";
        if (res.dist.length !== cells) return "dist length: " + res.dist.length;
        if (res.parent.length !== cells) return "parent length: " + res.parent.length;
        if (res.dist[0] !== 0) return "the seed cell should be at distance 0: " + res.dist[0];
        if (res.parent[0] !== -1) return "the seed cell should have parent -1: " + res.parent[0];
        if (typeof res.pops !== "number" || res.pops <= 0) return "pops: " + res.pops;

        // A blocked mask of the wrong length is a TypeError.
        let threw = false;
        try {
            nav.field("t", { seeds: seeds, blocked: new Uint8Array(3) });
        } catch (e) { threw = e instanceof TypeError; }
        if (!threw) return "a short blocked mask should throw TypeError";

        // No seeds is legal: an all-unreached field, not a throw.
        const empty = nav.field("t", {});
        if (empty.dist.length !== cells) return "seedless field length: " + empty.dist.length;

        return "SUCCESS";
    })()
)JS";

#ifdef BROGAMEAGENT_HAS_NN

// ---------------------------------------------------------------------------
// bro.ai.game.nn
// ---------------------------------------------------------------------------

const char* kNnCircuits = R"JS(
    (function() {
        const nn = bro.ai.game.nn;
        if (nn.available === false) return "nn namespace is stubbed out";

        // The old AITensor factory now answers a bro.tensor GpuTensor.
        const t = nn.createTensor(2, 3);
        if (!t || t.rows !== 2 || t.cols !== 3) return "createTensor shape: " + JSON.stringify(t);

        const lin = nn.createLinear(4, 2, 1234);
        if (lin.inDim !== 4 || lin.outDim !== 2) return "linear dims: " + lin.inDim + "x" + lin.outDim;
        if (!lin.W || lin.W.rows !== 2 || lin.W.cols !== 4) return "linear W shape";
        if (!lin.b || lin.b.rows !== 2) return "linear b shape";

        const x = new Float32Array([1, 0, -1, 0.5]);
        const y = new Float32Array(2);
        lin.forward(x, y);
        if (!isFinite(y[0]) || !isFinite(y[1])) return "linear forward produced non-finite output";

        const relu = nn.createRelu();
        const rin = new Float32Array([-2, 3]);
        const rout = new Float32Array(2);
        relu.forward(rin, rout);
        if (rout[0] !== 0 || rout[1] !== 3) return "relu forward: " + rout[0] + "," + rout[1];

        const tanh = nn.createTanh();
        const tout = new Float32Array(2);
        tanh.forward(new Float32Array([0, 10]), tout);
        if (Math.abs(tout[0]) > 1e-6 || tout[1] < 0.99) return "tanh forward: " + tout[0] + "," + tout[1];

        const enc = nn.createDeepSetsEncoder({ hidden: 8, embedDim: 4 }, 7);
        if (!(enc.outDim > 0)) return "deep sets outDim: " + enc.outDim;
        if (!(enc.numParams > 0)) return "deep sets numParams: " + enc.numParams;
        if (enc.name !== "deepsets" && typeof enc.name !== "string") return "deep sets name";
        enc.zeroGrad();

        const vh = nn.createValueHead(4, 8, 11);
        if (!(vh.numParams > 0)) return "value head numParams: " + vh.numParams;
        const vOut = vh.forward(new Float32Array(4));
        if (typeof vOut !== "number") return "value head forward: " + typeof vOut;

        const head = nn.createFactoredPolicyHead(4, 3);
        if (!(head.numParams > 0)) return "factored head numParams: " + head.numParams;
        if (head.totalLogits !== nn.N_MOVE + nn.N_ATTACK + nn.N_ABILITY) {
            return "factored head totalLogits: " + head.totalLogits;
        }

        return "SUCCESS";
    })()
)JS";

const char* kNnOps = R"JS(
    (function() {
        const nn = bro.ai.game.nn;

        // Ops take a GpuTensor or a Float32Array; the flat form is the one
        // the old binding's AITensor stood in for.
        const y = new Float32Array(3);
        nn.reluForward(new Float32Array([-1, 0.5, 2]), y);
        if (y[0] !== 0 || y[1] !== 0.5 || y[2] !== 2) return "reluForward: " + Array.from(y);

        const probs = new Float32Array(3);
        nn.softmaxForward(new Float32Array([1, 1, 1]), probs);
        let sum = probs[0] + probs[1] + probs[2];
        if (Math.abs(sum - 1) > 1e-5) return "softmax does not sum to 1: " + sum;

        // A mask zeroes the masked entry.
        const masked = new Float32Array(3);
        nn.softmaxForward(new Float32Array([1, 1, 1]), masked, new Float32Array([1, 0, 1]));
        if (masked[1] > 1e-6) return "masked softmax leaked probability: " + masked[1];

        // A mask the kernel would read past, or that is not a Float32Array,
        // is refused rather than read out of bounds or silently ignored.
        const refuses = (fn, what) => {
            try { fn(); } catch (e) { if (e instanceof TypeError) return null; return what + " threw " + e; }
            return what + " did not throw";
        };
        let err = refuses(() => nn.softmaxForward(new Float32Array([1, 1, 1]), masked,
                                                  new Float32Array([1, 0])), "short softmax mask");
        if (err) return err;
        err = refuses(() => nn.softmaxForward(new Float32Array([1, 1, 1]), masked,
                                              new Int32Array([1, 0, 1])), "Int32Array softmax mask");
        if (err) return err;
        const fLogits = new Float32Array(nn.N_MOVE + nn.N_ATTACK + nn.N_ABILITY);
        const fProbs = new Float32Array(fLogits.length);
        err = refuses(() => nn.factoredSoftmax(fLogits, fProbs, new Float32Array(1)), "short atkMask");
        if (err) return err;
        nn.factoredSoftmax(fLogits, fProbs, new Float32Array(nn.N_ATTACK - 1).fill(1),
                           new Float32Array(nn.N_ABILITY - 1).fill(1));

        const mse = nn.mseScalar(0.5, 1.0);
        if (!(mse.loss > 0) || typeof mse.dPred !== "number") return "mseScalar: " + JSON.stringify(mse);

        const acc = new Float32Array([1, 2]);
        nn.addInplace(acc, new Float32Array([1, 1]));
        if (acc[0] !== 2 || acc[1] !== 3) return "addInplace: " + Array.from(acc);
        nn.addScalarInplace(acc, 1);
        if (acc[0] !== 3) return "addScalarInplace: " + acc[0];

        // The factored action space the hero policy speaks.
        if (nn.N_MOVE !== 9 || nn.N_ATTACK !== 6 || nn.N_ABILITY !== 9) {
            return "factored head constants: " + nn.N_MOVE + "/" + nn.N_ATTACK + "/" + nn.N_ABILITY;
        }
        const headSizes = [nn.N_MOVE, nn.N_ATTACK, nn.N_ABILITY];
        const flat = nn.encodeFlatAction([1, 2, 0], headSizes);
        const dec = nn.decodeFlatAction(flat, headSizes);
        if (dec[0] !== 1 || dec[1] !== 2 || dec[2] !== 0) {
            return "flat action round trip: " + JSON.stringify(dec);
        }
        if (nn.flatActionCount(headSizes) !== nn.N_MOVE * nn.N_ATTACK * nn.N_ABILITY) {
            return "flatActionCount: " + nn.flatActionCount(headSizes);
        }

        return "SUCCESS";
    })()
)JS";

const char* kNnNets = R"JS(
    (function() {
        const nn = bro.ai.game.nn;

        const net = nn.createSingleHeroNet({ enc: { hidden: 8, embedDim: 4 },
                                             trunkHidden: 8, valueHidden: 8, seed: 5 });
        if (!(net.numParams > 0)) return "SingleHeroNet numParams: " + net.numParams;
        if (!(net.policyLogits > 0)) return "policyLogits: " + net.policyLogits;
        net.zeroGrad();
        const blob = net.save();
        if (!(blob instanceof Uint8Array) || blob.length === 0) return "SingleHeroNet.save gave nothing";
        net.load(blob);

        // load() of a blob that is not a checkpoint is a catchable TypeError,
        // not a crash.
        let threw = false;
        try { net.load(new Uint8Array([1, 2, 3])); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) return "SingleHeroNet.load(garbage) should throw TypeError";

        const pv = nn.createPolicyValueNet({ inDim: 6, numActions: 4, hidden: [8],
                                             valueHidden: 8, seed: 3 });
        if (pv.inDim !== 6 || pv.numActions !== 4) return "PolicyValueNet dims";
        const logits = new Float32Array(4);
        const out = pv.forward(new Float32Array(6), logits);
        if (typeof out !== "number" && typeof out !== "object") return "PolicyValueNet.forward result";
        if (pv.device !== "cpu" && pv.device !== "gpu") return "PolicyValueNet.device: " + pv.device;

        // Missing required config is a TypeError, not a silent default net.
        let bad = false;
        try { nn.createPolicyValueNet({ inDim: 6 }); } catch (e) { bad = e instanceof TypeError; }
        if (!bad) return "createPolicyValueNet without hidden[] should throw";

        // Multi-head (factored) form.
        const mh = nn.createPolicyValueNet({ inDim: 6, headSizes: [3, 2], hidden: [8],
                                             valueHidden: 8, seed: 3 });
        if (mh.numHeads !== 2) return "numHeads: " + mh.numHeads;
        const sizes = mh.headSizes();
        if (!Array.isArray(sizes) || sizes.length !== 2 || sizes[0] !== 3) {
            return "headSizes(): " + JSON.stringify(sizes);
        }
        const offs = mh.headOffsets();
        if (!Array.isArray(offs) || offs.length !== 3) return "headOffsets(): " + JSON.stringify(offs);

        const tx = nn.createSingleHeroNetTX({ dModel: 8, dFf: 16, numHeads: 2, numBlocks: 1,
                                              trunkHidden: 8, valueHidden: 8, seed: 9 });
        if (!(tx.numParams > 0)) return "SingleHeroNetTX numParams";
        tx.zeroGrad();
        tx.adamStep({ lr: 1e-3 });

        // WeightsHandle is the trainer→evaluator hot-swap: publish(blob,
        // version), snapshot() → { blob, version }, version() — the version
        // is a BigInt, as it was before the port.
        const handle = nn.createWeightsHandle();
        if (handle.snapshot() !== null) return "a fresh handle should snapshot null";
        handle.publish(blob, 3);
        const snap = handle.snapshot();
        if (!snap || !(snap.blob instanceof Uint8Array)) return "weights handle snapshot blob";
        if (typeof snap.version !== "bigint") return "snapshot version type: " + typeof snap.version;
        if (snap.version !== 3n) return "snapshot version: " + snap.version;
        if (handle.version() !== 3n) return "handle.version(): " + handle.version();

        return "SUCCESS";
    })()
)JS";

// ---------------------------------------------------------------------------
// bro.ai.game.learn
// ---------------------------------------------------------------------------

const char* kLearn = R"JS(
    (function() {
        const G = bro.ai.game;
        const learn = G.learn;
        if (learn.available === false) return "learn namespace is stubbed out";

        const buf = learn.createReplayBuffer(4);
        if (buf.capacity !== 4) return "buffer capacity: " + buf.capacity;
        for (let i = 0; i < 6; i++) buf.push({ obs: new Float32Array([i]), valueTarget: i });
        if (buf.size !== 4) return "ring buffer did not evict: " + buf.size;
        const all = buf.all();
        if (!Array.isArray(all) || all.length !== 4) return "all(): " + all.length;
        if (!(all[0].obs instanceof Float32Array)) return "situation.obs is not a Float32Array";
        const batch = buf.sample(2);
        if (batch.length !== 2) return "sample(2): " + batch.length;
        buf.clear();
        if (buf.size !== 0) return "clear() left " + buf.size;

        const net = G.nn.createSingleHeroNet({ enc: { hidden: 8, embedDim: 4 },
                                               trunkHidden: 8, valueHidden: 8, seed: 5 });
        const handle = G.nn.createWeightsHandle();

        const evaluator = learn.createNeuralEvaluator(net, handle);
        const w = G.createWorld();
        const hero = G.createAgent({ id: 1, teamId: 0, x: 0, z: 0 });
        w.addAgent(hero);
        const v = evaluator.evaluate(w, 1);
        if (typeof v !== "number") return "evaluator.evaluate: " + typeof v;

        const prior = learn.createNeuralPrior(net, handle);
        prior.setTemperature(1.5);
        prior.setUniformMix(0.25);

        const gumbel = learn.createGumbelNoisePrior(prior, 0.5);
        gumbel.reseed(99);
        gumbel.setScale(0.25);

        // A prior argument that is not a prior throws rather than silently
        // producing a dead object.
        let threw = false;
        try { learn.createNeuralPrior({}); } catch (e) { threw = e instanceof TypeError; }
        if (!threw) return "createNeuralPrior({}) should throw TypeError";

        const trainer = learn.createExItTrainer();
        trainer.setNet(net);
        trainer.setBuffer(buf);
        trainer.setWeightsHandle(handle);
        trainer.setConfig({ batch: 2, lr: 1e-3, publishEvery: 1 });
        if (trainer.totalSteps !== 0) return "fresh trainer totalSteps: " + trainer.totalSteps;
        for (let i = 0; i < 4; i++) buf.push({ obs: new Float32Array([i]), valueTarget: 0.5 });
        trainer.stepN(2);
        if (!(trainer.totalSteps > 0)) return "stepN did not step: " + trainer.totalSteps;

        // Generic (non-hero) side.
        const gbuf = learn.createGenericReplayBuffer(8);
        gbuf.push({ obs: new Float32Array([1, 2]), policyTarget: new Float32Array([0.5, 0.5]),
                    valueTarget: 1 });
        const gall = gbuf.all();
        if (!(gall[0].policyTarget instanceof Float32Array)) return "generic situation policyTarget";
        if (gbuf.size !== 1) return "generic buffer size: " + gbuf.size;
        const gt = learn.createGenericExItTrainer();
        gt.setBuffer(gbuf);

        const pvnet = G.nn.createPolicyValueNet({ inDim: 2, numActions: 2, hidden: [8],
                                                  valueHidden: 8, seed: 1 });
        const backend = learn.createDirectBackend(pvnet);
        if (backend.numActions !== 2 || backend.inDim !== 2) return "direct backend dims";

        const server = learn.createInferenceServer(pvnet, { maxBatchSize: 4 });
        const one = server.evaluate(new Float32Array([0.5, 0.5]));
        if (!one || !(one.logits instanceof Float32Array)) return "server.evaluate logits";
        if (typeof one.value !== "number") return "server.evaluate value";
        const many = server.evaluateBatch([new Float32Array([0, 0]), new Float32Array([1, 1])]);
        if (!Array.isArray(many) || many.length !== 2) return "evaluateBatch length";
        // A row of the wrong width is a catchable Error, not a C++ exception
        // unwinding through compiled code.
        let rowErr = null;
        try { server.evaluateBatch([new Float32Array([0, 0]), new Float32Array([1, 1, 1])]); }
        catch (e) { rowErr = e; }
        if (!(rowErr instanceof Error)) return "evaluateBatch accepted a 3-wide row";
        let oneErr = null;
        try { server.evaluate(new Float32Array([1])); } catch (e) { oneErr = e; }
        if (!(oneErr instanceof Error)) return "evaluate accepted a 1-wide row";
        const sbackend = learn.createServerBackend(server, pvnet);
        if (sbackend.numActions !== 2) return "server backend numActions";
        server.shutdown();

        // A ServerBackend co-owns its server: shutdown() drops the server
        // handle's reference only, so a search still driving the backend
        // afterwards must work rather than call a freed server.
        let s = 0;
        const env = {
            numActions: 2,
            snapshot() { return s; },
            restore(v) { s = v; },
            step(a) { s += 1; return { reward: a === 1 ? 1 : 0, done: s >= 3 }; },
            legalActions() { return s >= 3 ? [] : [0, 1]; },
            observe() { return new Float32Array([s, 1]); },
        };
        const gm = G.createGenericMcts({ env, iterations: 16, backend: sbackend });
        const pick = gm.search();
        if (pick !== 0 && pick !== 1) return "search through a shut-down server's backend: " + pick;

        // An observation the backend cannot take is a catchable Error.
        s = 0;
        const wideEnv = Object.assign({}, env, { observe() { return new Float32Array([s, 1, 2]); } });
        const gmWide = G.createGenericMcts({ env: wideEnv, iterations: 4,
                                             backend: learn.createDirectBackend(pvnet) });
        let wideErr = null;
        try { gmWide.search(); } catch (e) { wideErr = e; }
        if (!(wideErr instanceof Error)) return "search with a 3-wide observation did not throw";

        return "SUCCESS";
    })()
)JS";

// ---------------------------------------------------------------------------
// bro.ai.game.grid
// ---------------------------------------------------------------------------

const char* kGrid = R"JS(
    (function() {
        const grid = bro.ai.game.grid;
        if (grid.available === false) return "grid namespace is stubbed out";

        // ObsWindow: a JS tile sampler feeding a native window.
        const win = grid.createObsWindow({
            colsBehind: 1, colsAhead: 1, rowsUp: 1, rowsDown: 1,
            tile: { channels: 1, sample: function (col, row) { return (col + row) % 2; } },
        });
        if (!(win.outDim > 0)) return "obs window outDim: " + win.outDim;
        const obs = win.build(2, 2);
        if (!(obs instanceof Float32Array) || obs.length !== win.outDim) return "obs window build";

        // FrameStack.
        const fs = grid.createFrameStack({ innerDim: 2, k: 3 });
        if (fs.outDim !== 6 || fs.k !== 3) return "frame stack dims";
        fs.push(new Float32Array([1, 2]));
        if (fs.filled < 1) return "frame stack filled: " + fs.filled;
        const stacked = fs.read();
        if (!(stacked instanceof Float32Array) || stacked.length !== 6) return "frame stack read";
        fs.reset();
        if (fs.filled !== 0) return "frame stack reset";

        // FailureTape.
        const tape = grid.createFailureTape({ tapeDepth: 4, ringCapacity: 8, penalty: 0.5 });
        tape.recordFailure([{ obsKey: 1, action: 0 }, { obsKey: 2, action: 1 }]);
        if (tape.size < 1) return "failure tape size: " + tape.size;
        const mult = tape.multipliers(1, 2);
        if (!(mult instanceof Float32Array)) return "multipliers is not a Float32Array";
        const priors = new Float32Array([0.5, 0.5]);
        tape.applyPriors(1, priors);
        tape.clear();
        if (tape.size !== 0) return "failure tape clear";

        // BestCrop.
        const crop = grid.createBestCrop({ capacity: 4, seedTopK: 2, seed: 7 });
        crop.push({ score: 1, depth: 2, state: new Float32Array([1]) });
        crop.push({ score: 3, depth: 4, state: new Float32Array([2]) });
        if (crop.size !== 2) return "best crop size: " + crop.size;
        crop.seed();
        crop.clear();
        if (crop.size !== 0) return "best crop clear";

        // PotentialShaper + StallDetector.
        const shaper = grid.createPotentialShaper({ gamma: 0.95 });
        if (Math.abs(shaper.gamma - 0.95) > 1e-6) return "shaper gamma: " + shaper.gamma;
        shaper.reset(0);
        const bonus = shaper.step(1);
        if (typeof bonus !== "number") return "shaper.step: " + typeof bonus;

        const stall = grid.createStallDetector({ epsilon: 0.01, patience: 2 });
        stall.reset();
        stall.tick(0);
        const stalled = stall.tick(0);
        if (typeof stalled !== "boolean") return "stall detector tick: " + typeof stalled;

        return "SUCCESS";
    })()
)JS";

const char* kGridRecording = R"JS(
    (function() {
        const grid = bro.ai.game.grid;
        const path = "brogameagent_test_grid_replay.bin";

        // open(path, episodeId, seed, dt, { roster, frame, events }); rows are
        // positional, in schema order, exactly as the old binding read them.
        const rec = grid.createGenericRecorder();
        const schemas = {
            roster: [{ name: "id", type: "i32" }, { name: "team", type: "i32" }],
            frame: [{ name: "x", type: "f32" }, { name: "y", type: "f32" }],
            events: [{ name: "kind", type: "i32" }],
        };
        if (!rec.open(path, 1, 42, 1 / 60, schemas)) return "recorder.open failed";
        if (!rec.isOpen()) return "recorder.isOpen() is false after open";
        rec.writeRoster([[1, 0], [2, 1]]);
        for (let i = 0; i < 3; i++) rec.recordFrame(i, i / 60, [[i, -i], [i + 1, 0]]);
        if (rec.frameCount !== 3) return "recorder frameCount: " + rec.frameCount;
        rec.close();
        if (rec.isOpen()) return "recorder still open after close";

        const rr = grid.createGenericReplayReader();
        if (!rr.open(path)) return "reader.open failed: " + rr.errorMessage;
        if (rr.frameCount !== 3) return "reader frameCount: " + rr.frameCount;
        const f0 = rr.frame(0);
        if (!Array.isArray(f0.rows) || f0.rows.length !== 2) {
            return "reader.frame rows: " + JSON.stringify(f0.rows);
        }
        if (f0.rows[0][0] !== 0) return "reader.frame field: " + JSON.stringify(f0.rows[0]);
        if (typeof f0.stepIdx !== "bigint") return "frame stepIdx type: " + typeof f0.stepIdx;
        const traj = rr.trajectory(0, "x");
        if (!Array.isArray(traj) || traj.length !== 3) return "trajectory length: " + traj.length;
        if (traj[2] !== 2) return "trajectory values: " + JSON.stringify(traj);

        // The trainer: the harness the grid apps drive.
        // The documented nested shape (docs/ai-game-tools.js).
        const trainer = grid.createGridTrainer({
            net: { inDim: 4, numActions: 3, hidden: [8], valueHidden: 8, seed: 1 },
            buffer: { capacity: 16 },
            trainer: { batch: 2, lr: 1e-3 },
        });
        if (trainer.running) return "a fresh trainer should not be running";
        trainer.ingestSituation({ obs: new Float32Array([0, 0, 0, 0]),
                                  policyTarget: new Float32Array([1, 0, 0]), valueTarget: 1 });
        trainer.ingestSituation({ obs: new Float32Array([1, 0, 0, 0]),
                                  policyTarget: new Float32Array([0, 1, 0]), valueTarget: 0 });
        trainer.ingestEpisode({ totalReturn: 1, depth: 2, failed: false, prefix: [0, 1] });
        trainer.stepSync(1);
        const stats = trainer.stats();
        if (!stats || typeof stats.totalSteps !== "number" || stats.totalSteps < 1) {
            return "trainer stats: " + JSON.stringify(stats);
        }
        if (stats.bufferSize < 1) return "trainer bufferSize: " + stats.bufferSize;
        const events = trainer.pollEvents();
        if (!Array.isArray(events)) return "pollEvents is not an array";

        return "SUCCESS";
    })()
)JS";

#endif  // BROGAMEAGENT_HAS_NN

// ---------------------------------------------------------------------------
// GenericMcts: the env and the prior/value callbacks ride on the search's JS
// object, so a game that is its own env and keeps `this.mcts` is an ordinary
// cycle the collector can free, and a kept one still searches after a GC.
// ---------------------------------------------------------------------------

const char* kGenericMctsCycleSetup = R"JS(
    (function() {
        const G = bro.ai.game;
        class Game {
            constructor() {
                this.s = 0;
                this.priorCalls = 0;
                this.valueCalls = 0;
                const self = this;
                this.mcts = G.createGenericMcts({
                    env: this, iterations: 24,
                    priorFn(obs, legal) { self.priorCalls++; return new Float32Array([0.2, 0.8]); },
                    valueFn(obs) { self.valueCalls++; return obs[0] > 1 ? 1 : 0; },
                });
            }
            get numActions() { return 2; }
            snapshot() { return { s: this.s }; }
            restore(v) { this.s = v.s; }
            step(a) { this.s += a; return { reward: a, done: this.s >= 3 }; }
            legalActions() { return this.s >= 3 ? [] : [0, 1]; }
            observe() { return new Float32Array([this.s, 1]); }
        }

        const dropped = new Game();
        const p0 = dropped.mcts.search();
        if (p0 !== 0 && p0 !== 1) return "search: " + p0;
        if (dropped.priorCalls === 0 || dropped.valueCalls === 0) return "callbacks not called";
        globalThis.__droppedGame = new WeakRef(dropped);
        globalThis.__droppedMcts = new WeakRef(dropped.mcts);

        const kept = new Game();
        kept.mcts.search();
        kept.mcts.advanceRoot(1);
        globalThis.__keptGame = kept;
        return "SUCCESS";
    })()
)JS";

const char* kGenericMctsCycleCheck = R"JS(
    (function() {
        if (globalThis.__droppedGame.deref() !== undefined) return "a game that is its own env leaked";
        if (globalThis.__droppedMcts.deref() !== undefined) return "its GenericMcts leaked";

        const kept = globalThis.__keptGame;
        kept.s = 0;
        kept.priorCalls = 0; kept.valueCalls = 0;
        kept.mcts.reset();
        const pick = kept.mcts.search();
        if (pick !== 0 && pick !== 1) return "kept search after GC: " + pick;
        if (kept.priorCalls === 0 || kept.valueCalls === 0) return "kept callbacks lost after GC";

        // Clearing the callbacks falls back to uniform priors / rollouts.
        kept.mcts.setPriorFn(null);
        kept.mcts.setValueFn(null);
        kept.priorCalls = 0; kept.valueCalls = 0;
        kept.s = 0; kept.mcts.reset();
        const p2 = kept.mcts.search();
        if (p2 !== 0 && p2 !== 1) return "search without callbacks: " + p2;
        if (kept.priorCalls !== 0 || kept.valueCalls !== 0) return "cleared callbacks still called";
        return "SUCCESS";
    })()
)JS";

// A full collection with no script frame on the stack: drain the microtask
// checkpoint so WeakRef targets held for the current job are released.
void collectNow() {
    ev::drainMicrotasks();
    ev::collectGarbage();
    ev::drainFinalizers();
    ev::collectGarbage();
}

} // namespace

int main() {
    std::cout << "Running brogameagent restored-API test..." << std::endl;

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
#ifdef BROGAMEAGENT_HAS_NN
        // nn.createTensor answers a bro.tensor GpuTensor, so the realm needs
        // brotensor's own binding — the host installs it exactly once, and a
        // standalone test stands in for the host here.
        brotensor::api::installTensor();
#endif
        brogameagent::api::installGameAi();

        runJs("World.findById / registerAbility / seed", kWorldExtras);
        runJs("Agent.applyAction", kAgentApplyAction);
        runJs("HexNav.field", kHexNavField);
        runJs("binding fixes (GC audit)", kBindingFixes);
        runJs("GenericMcts env cycle: setup", kGenericMctsCycleSetup);
        collectNow();
        runJs("GenericMcts env cycle: collected / kept", kGenericMctsCycleCheck);
#ifdef BROGAMEAGENT_HAS_NN
        runJs("nn circuits", kNnCircuits);
        runJs("nn ops", kNnOps);
        runJs("nn nets", kNnNets);
        runJs("learn", kLearn);
        runJs("grid", kGrid);
        runJs("grid recording", kGridRecording);
#else
        std::cout << "(nn / learn / grid skipped: built without the neural layer)" << std::endl;
#endif
    }
    ev::destroyRealm(realm);

    std::cout << "All brogameagent restored-API tests passed!" << std::endl;
    return 0;
}
