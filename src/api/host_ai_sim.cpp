// Simulation / Recorder / ReplayReader.
//
// The fixed-dt rollout driver and the .bgar replay writer + reader were all
// dropped by the port; createSimulation / createRecorder / createReplayReader
// were missing entirely, taking addPolicy, removePolicy, runSteps,
// resetCounters, open, writeRoster, recordFrame, close, frame, trajectory and
// damageSummary with them.

#include "host_ai_mcts_shared.h"

#include <brogameagent/recorder.h>
#include <brogameagent/replay_reader.h>
#include <brogameagent/simulation.h>

#include <memory>
#include <vector>

namespace brogameagent::api {

HostClass g_simulationClass;
HostClass g_recorderClass;
HostClass g_replayReaderClass;

namespace {

// The world handle rides on the simulation as `_world` and each policy as
// `_policies[i]` (policySlots[i] is the native end, policyAgents[i] its
// agent id), so a policy closing over the game that owns the simulation —
// or a world that keeps its simulation — is an ordinary collectable cycle.
// step/runSteps bind them for the call (StepScope); the native Simulation
// holds a World& whose lifetime `worldLife` guards in place of a root.
struct HostSimulation {
    uint32_t tag = kHostSimulationTag;
    std::unique_ptr<brogameagent::Simulation> sim;
    HostWorld* world = nullptr;
    std::weak_ptr<int> worldLife;
    std::vector<int> policyAgents;
    std::vector<JsSlotPtr> policySlots;
    ev::Persistent* liveWorld = nullptr;  // the world handle, inside a StepScope
};

struct HostRecorder {
    uint32_t tag = kHostRecorderTag;
    brogameagent::Recorder recorder;
};

struct HostReplayReader {
    uint32_t tag = kHostReplayReaderTag;
    brogameagent::ReplayReader reader;
};

HostSimulation* unwrapSimulation(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostSimulation*>(ev::handleData(v));
    return (h && h->tag == kHostSimulationTag) ? h : nullptr;
}

HostRecorder* unwrapRecorder(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostRecorder*>(ev::handleData(v));
    return (h && h->tag == kHostRecorderTag) ? h : nullptr;
}

HostReplayReader* unwrapReplayReader(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<HostReplayReader*>(ev::handleData(v));
    return (h && h->tag == kHostReplayReaderTag) ? h : nullptr;
}

/// self._policies[idx] = fn, creating the array on first use. Both current.
void setPolicyFn(Value self, size_t idx, Value fn) {
    ev::Persistent selfP(self), fnP(fn);
    ev::Persistent arr(ev::getProperty(selfP.get(), "_policies"));
    if (!ev::isObject(arr.get())) {
        arr.set(ev::makeArray(0));
        selfP.set(ev::setProperty(selfP.get(), "_policies", arr.get()));
    }
    ev::setElement(arr.get(), static_cast<uint32_t>(idx), fnP.get());
}

/// Binds the policies and the world for one step()/runSteps() call. False
/// (and nothing bound) once the world the Simulation steps is gone.
class StepScope {
public:
    StepScope(HostSimulation* h, Value self)
        : h_(h), prev_(h->liveWorld), self_(self),
          world_(checkedWorld(h, self_.get())),
          policies_(self_.get(), h->policySlots, "_policies"),
          active_(world_.get()) {
        h_->liveWorld = &world_;
    }
    ~StepScope() { h_->liveWorld = prev_; }
    StepScope(const StepScope&) = delete;
    StepScope& operator=(const StepScope&) = delete;

private:
    // `_world` is reachable from JS; a replaced value is not the world the
    // Simulation steps, so policies then see undefined for it.
    static Value checkedWorld(HostSimulation* h, Value self) {
        Value v = ev::getProperty(self, "_world");
        return unwrapWorld(v) == h->world ? v : ev::undefined();
    }

    HostSimulation* h_;
    ev::Persistent* prev_;
    ev::Persistent self_;
    ev::Persistent world_;
    CallbackScope policies_;
    WorldArgScope active_;
};

} // namespace

void ensureAISimClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    // ── Simulation ─────────────────────────────────────────────────────────
    g_simulationClass.init("AISimulation", [](ObjectBuilder& b) {
        b.def("step", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapSimulation(self);
            if (!h || !h->sim || h->worldLife.expired()) return ev::undefined();
            StepScope scope(h, self);
            h->sim->step(static_cast<float>(numAt(a, 0)));
            return ev::undefined();
        });

        b.def("runSteps", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapSimulation(self);
            if (!h || !h->sim || h->worldLife.expired()) return ev::undefined();
            StepScope scope(h, self);
            h->sim->runSteps(static_cast<float>(numAt(a, 0)), i32At(a, 1));
            return ev::undefined();
        });

        b.accessor("steps", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapSimulation(self);
            return ev::fromDouble((h && h->sim) ? h->sim->steps() : 0);
        }, nullptr);

        b.accessor("elapsed", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapSimulation(self);
            return ev::fromDouble((h && h->sim) ? h->sim->elapsed() : 0.0);
        }, nullptr);

        b.def("resetCounters", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapSimulation(self);
            if (h && h->sim) h->sim->resetCounters();
            return ev::undefined();
        });

        b.def("addPolicy", 2, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapSimulation(self);
            if (!h || !h->sim || a.size() < 2) return ev::undefined();
            int agentId = i32At(a, 0);
            if (!ev::isFunction(a[1])) return ev::throwTypeError("policy must be a function");

            // One `_policies` index per agent id: re-adding replaces the
            // function there and the native policy with a fresh slot.
            size_t idx = 0;
            while (idx < h->policyAgents.size() && h->policyAgents[idx] != agentId) ++idx;
            if (idx == h->policyAgents.size()) {
                h->policyAgents.push_back(agentId);
                h->policySlots.push_back(nullptr);
            }
            setPolicyFn(self, idx, a[1]);
            auto slot = std::make_shared<JsCallbackSlot>();
            h->policySlots[idx] = slot;
            h->sim->addPolicy(agentId, [h, slot](
                    brogameagent::Agent& agent,
                    const brogameagent::World&) -> brogameagent::AgentAction {
                if (!slot->bound() || !h->liveWorld) return {};
                ev::Persistent agentVal(ev::undefined());
                if (ev::isObject(h->liveWorld->get())) {
                    agentVal.set(worldAgentValue(h->liveWorld->get(), &agent));
                }
                Value args[2] = { agentVal.get(), h->liveWorld->get() };
                auto r = slot->call(args);
                if (r.thrown || !ev::isObject(r.value)) return {};
                return parseAgentAction(r.value);
            });
            return ev::undefined();
        });

        b.def("removePolicy", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapSimulation(self);
            if (!h || !h->sim || a.empty()) return ev::undefined();
            int agentId = i32At(a, 0);
            h->sim->removePolicy(agentId);
            for (size_t i = 0; i < h->policyAgents.size(); ++i) {
                if (h->policyAgents[i] != agentId) continue;
                h->policySlots[i] = nullptr;
                setPolicyFn(self, i, ev::undefined());
                break;
            }
            return ev::undefined();
        });
    });

    // ── Recorder ───────────────────────────────────────────────────────────
    g_recorderClass.init("AIRecorder", [](ObjectBuilder& b) {
        b.def("open", 4, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapRecorder(self);
            if (!h || a.empty()) return ev::fromBool(false);
            return ev::fromBool(h->recorder.open(strAt(a, 0), u64At(a, 1), u64At(a, 2),
                                                 static_cast<float>(numAt(a, 3))));
        });

        b.accessor("isOpen", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapRecorder(self);
            return ev::fromBool(h && h->recorder.isOpen());
        }, nullptr);

        b.def("writeRoster", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapRecorder(self);
            if (!h || a.empty()) return ev::undefined();
            auto* w = unwrapWorld(a[0]);
            if (!w) return ev::throwTypeError("writeRoster: expected a World");
            h->recorder.writeRoster(w->world.agents());
            return ev::undefined();
        });

        b.def("recordFrame", 3, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapRecorder(self);
            if (!h || a.size() < 3) return ev::undefined();
            auto* w = unwrapWorld(a[2]);
            if (!w) return ev::throwTypeError("recordFrame: expected a World");
            h->recorder.recordFrame(u32At(a, 0), static_cast<float>(numAt(a, 1)), w->world);
            return ev::undefined();
        });

        b.def("close", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapRecorder(self);
            return ev::fromBool(h && h->recorder.close());
        });

        b.accessor("frameCount", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapRecorder(self);
            return ev::fromDouble(h ? static_cast<double>(h->recorder.frameCount()) : 0.0);
        }, nullptr);
    });

    // ── ReplayReader ───────────────────────────────────────────────────────
    g_replayReaderClass.init("AIReplayReader", [](ObjectBuilder& b) {
        b.def("open", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapReplayReader(self);
            if (!h || a.empty()) return ev::fromBool(false);
            return ev::fromBool(h->reader.open(strAt(a, 0)));
        });

        b.accessor("errorMessage", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapReplayReader(self);
            return h ? ev::fromUtf8(h->reader.errorMessage()) : ev::fromUtf8("");
        }, nullptr);

        b.accessor("frameCount", [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapReplayReader(self);
            return ev::fromDouble(h ? static_cast<double>(h->reader.frameCount()) : 0.0);
        }, nullptr);

        b.def("frame", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapReplayReader(self);
            if (!h || a.empty()) return ev::null();
            int idx = i32At(a, 0);
            if (idx < 0 || idx >= static_cast<int>(h->reader.frameCount())) return ev::null();
            auto f = h->reader.frame(static_cast<size_t>(idx));

            ObjectBuilder o;
            o.set("stepIdx", ev::fromDouble(f.header.stepIdx));
            o.set("elapsed", ev::fromDouble(f.header.elapsed));
            o.set("agents", hostArrayOf(f.agents.size(), [&](size_t i) {
                const auto& ag = f.agents[i];
                ObjectBuilder ao;
                ao.set("id", ev::fromDouble(ag.id));
                ao.set("x", ev::fromDouble(ag.x));
                ao.set("z", ev::fromDouble(ag.z));
                ao.set("hp", ev::fromDouble(ag.hp));
                ao.set("mana", ev::fromDouble(ag.mana));
                ao.set("yaw", ev::fromDouble(ag.yaw));
                ao.set("alive", ev::fromBool(
                    (ag.flags & brogameagent::replay::AGENT_FLAG_ALIVE) != 0));
                return ao.get();
            }));
            o.set("events", hostArrayOf(f.events.size(), [&](size_t i) {
                const auto& e = f.events[i];
                ObjectBuilder eo;
                eo.set("attackerId", ev::fromDouble(e.attackerId));
                eo.set("targetId", ev::fromDouble(e.targetId));
                eo.set("amount", ev::fromDouble(e.amount));
                eo.set("killed", ev::fromBool(e.killed != 0));
                return eo.get();
            }));
            return o.get();
        });

        b.def("trajectory", 1, [](Value self, std::span<const Value> a) -> Value {
            auto* h = unwrapReplayReader(self);
            if (!h || a.empty()) return hostArrayOf(0, [](size_t) { return ev::null(); });
            auto traj = h->reader.trajectory(i32At(a, 0));
            return hostArrayOf(traj.size(), [&](size_t i) {
                ObjectBuilder pt;
                pt.set("stepIdx", ev::fromDouble(traj[i].stepIdx));
                pt.set("elapsed", ev::fromDouble(traj[i].elapsed));
                pt.set("x", ev::fromDouble(traj[i].x));
                pt.set("z", ev::fromDouble(traj[i].z));
                pt.set("hp", ev::fromDouble(traj[i].hp));
                pt.set("alive", ev::fromBool(traj[i].alive));
                return pt.get();
            });
        });

        b.def("damageSummary", 0, [](Value self, std::span<const Value>) -> Value {
            auto* h = unwrapReplayReader(self);
            if (!h) return hostArrayOf(0, [](size_t) { return ev::null(); });
            auto summary = h->reader.damageSummary();
            return hostArrayOf(summary.size(), [&](size_t i) {
                ObjectBuilder o;
                o.set("attackerId", ev::fromDouble(summary[i].attackerId));
                o.set("targetId", ev::fromDouble(summary[i].targetId));
                o.set("totalDamage", ev::fromDouble(summary[i].totalDamage));
                o.set("hits", ev::fromDouble(summary[i].hits));
                o.set("kills", ev::fromDouble(summary[i].kills));
                return o.get();
            });
        });
    });
}

void installAISim(ObjectBuilder& game) {
    ensureAISimClassesInstalled();

    game.def("createSimulation", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("createSimulation(world)");
        auto* w = unwrapWorld(a[0]);
        if (!w) return ev::throwTypeError("createSimulation: invalid world");
        auto cell = std::make_unique<HostSimulation>();
        cell->sim = std::make_unique<brogameagent::Simulation>(w->world);
        cell->world = w;
        cell->worldLife = w->life;
        ev::Persistent sim(g_simulationClass.createInstance(std::move(cell)));
        sim.set(ev::setProperty(sim.get(), "_world", a[0]));
        return sim.get();
    });

    game.def("createRecorder", 0, [](Value, std::span<const Value>) -> Value {
        return g_recorderClass.createInstance(std::make_unique<HostRecorder>());
    });

    game.def("createReplayReader", 0, [](Value, std::span<const Value>) -> Value {
        return g_replayReaderClass.createInstance(std::make_unique<HostReplayReader>());
    });
}

} // namespace brogameagent::api
