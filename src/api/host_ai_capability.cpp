// bro.ai.game.registerCapability(name, spec) — JS-authored capabilities.
//
// A registered capability is a brogameagent::Capability whose gate / start /
// advance / cancel are the spec's JS functions. The host that builds agent
// bindings (bro's scene-node attachAgent) turns a capability name from its
// `capabilities` list into one with makeRegisteredCapability (api.h), adds it
// to the binding's CapabilitySet, and from then on the binding drives it like
// any built-in: gate() when it builds the mask, start() when think() picks it
// through self.useCapability(name, arg0, arg1), advance() each frame until
// it reports done.
//
// Where the JS lives: every spec object is a property of ONE registry object
// per thread (bronze's runtime is per thread, and so is a Persistent). A
// Capability instance holds only the capability's name and looks its spec up
// at call time. No capability, binding or other C++ object roots a callback,
// so a callback that closes over the scene node that owns the binding is an
// ordinary JS cycle the collector can see through, not a leak held open by a
// native root. Re-registering a name replaces the spec for every binding that
// uses it.

#include "api.h"
#include "host_ai_internal.h"

#include <brogameagent/capability.h>

#include <cmath>
#include <unordered_map>

namespace brogameagent::api {

namespace {

struct CapRegistry {
    // Deliberately never destroyed, like HostClass's slots: a Persistent must
    // be released against the runtime that made it, and thread_local
    // destructors run in no order relative to bronze's own teardown.
    ev::Persistent* specs = nullptr;
    std::unordered_map<std::string, int> ids;
    int nextId = kJsCapFirst;
};

CapRegistry& registry() {
    static thread_local CapRegistry reg;
    if (!reg.specs) reg.specs = new ev::Persistent(ev::createObject());
    return reg;
}

bool idInUse(const CapRegistry& reg, int id) {
    for (const auto& [n, i] : reg.ids)
        if (i == id) return true;
    return false;
}

// Call spec[method](args...) with the spec as `this`. `found` is false when
// the capability or the method is gone (not registered on this thread, or
// the spec has no such function); a throw is reported as `thrown`.
struct SpecCall {
    bool found = false;
    ev::CallResult result;
};

SpecCall callSpec(const std::string& name, const char* method, std::span<const Value> args) {
    SpecCall out;
    CapRegistry& reg = registry();
    ev::Persistent spec(ev::getProperty(reg.specs->get(), name));
    if (!ev::isObject(spec.get())) return out;
    ev::Persistent fn(ev::getProperty(spec.get(), method));
    if (!ev::isFunction(fn.get())) return out;
    out.found = true;
    out.result = ev::call(fn.get(), spec.get(), args);
    return out;
}

bool specHas(const std::string& name, const char* method) {
    CapRegistry& reg = registry();
    ev::Persistent spec(ev::getProperty(reg.specs->get(), name));
    return ev::isObject(spec.get()) && ev::isFunction(ev::getProperty(spec.get(), method));
}

class RegisteredCapability final : public Capability {
public:
    RegisteredCapability(std::string name, int id) : name_(std::move(name)), id_(id) {}

    int id() const override { return id_; }
    const char* name() const override { return name_.c_str(); }

    // A missing gate means always available; a throwing one means not.
    bool gate(const CapContext&) const override {
        SpecCall c = callSpec(name_, "gate", {});
        if (!c.found) return true;
        return !c.result.thrown && ev::toBool(c.result.value);
    }

    // start(arg0, arg1): the two integers think() passed to
    // self.useCapability(name, arg0, arg1), -1 when omitted. With an advance
    // the action is in flight until advance says it is done; without one it
    // lasts action.dur seconds (0 = done at once).
    void start(const CapContext&, Action& a) override {
        a.elapsed = 0.0f;
        a.done = a.dur <= 0.0f && !specHas(name_, "advance");
        const Value args[2] = {ev::fromDouble(a.i0), ev::fromDouble(a.i1)};
        SpecCall c = callSpec(name_, "start", std::span<const Value>(args, 2));
        if (c.found && c.result.thrown) a.done = true;
    }

    // advance(dt, elapsed) -> true when done. A non-boolean answer keeps the
    // action running until its duration; a throw ends it rather than wedging
    // the binding on a broken callback.
    void advance(const CapContext&, Action& a, float dt) override {
        a.elapsed += dt;
        const Value args[2] = {ev::fromDouble(dt), ev::fromDouble(a.elapsed)};
        SpecCall c = callSpec(name_, "advance", std::span<const Value>(args, 2));
        if (c.found) {
            if (c.result.thrown) {
                a.done = true;
                return;
            }
            if (ev::isBool(c.result.value)) {
                a.done = ev::toBool(c.result.value);
                return;
            }
        }
        if (a.elapsed >= a.dur) a.done = true;
    }

    void cancel(const CapContext&, Action&) override { callSpec(name_, "cancel", {}); }

private:
    std::string name_;
    int id_;
};

Value jsRegisterCapability(Value, std::span<const Value> a) {
    if (a.size() < 2 || !ev::isString(a[0]) || !ev::isObject(a[1]))
        return ev::throwTypeError("registerCapability(name, spec)");
    const std::string name = ev::toUtf8(a[0]);
    if (name.empty()) return ev::throwTypeError("registerCapability: name must not be empty");

    static const char* const kMethods[] = {"gate", "start", "advance", "cancel"};
    for (const char* m : kMethods) {
        Value v = ev::getProperty(a[1], m);
        if (!ev::isUndefined(v) && !ev::isNull(v) && !ev::isFunction(v))
            return ev::throwTypeError(std::string("registerCapability: spec.") + m +
                                      " must be a function");
    }

    CapRegistry& reg = registry();
    int id = -1;
    Value idVal = ev::getProperty(a[1], "id");
    if (!ev::isUndefined(idVal) && !ev::isNull(idVal)) {
        if (!ev::isNumber(idVal)) return ev::throwTypeError("registerCapability: spec.id must be a number");
        const double d = ev::toDouble(idVal);
        if (!(d >= static_cast<double>(kJsCapFirst) && d <= 1e9 && std::floor(d) == d))
            return ev::throwRangeError("registerCapability: spec.id must be an integer >= " +
                                       std::to_string(kJsCapFirst));
        id = static_cast<int>(d);
        auto self = reg.ids.find(name);
        if ((self == reg.ids.end() || self->second != id) && idInUse(reg, id))
            return ev::throwRangeError("registerCapability: id " + std::to_string(id) +
                                       " is already registered");
    } else if (auto it = reg.ids.find(name); it != reg.ids.end()) {
        id = it->second;  // re-registration keeps the name's id
    } else {
        while (idInUse(reg, reg.nextId)) ++reg.nextId;
        id = reg.nextId++;
    }

    reg.specs->set(ev::setProperty(reg.specs->get(), name, a[1]));
    reg.ids[name] = id;
    return ev::fromDouble(id);
}

} // namespace

std::unique_ptr<Capability> makeRegisteredCapability(std::string_view name) {
    CapRegistry& reg = registry();
    auto it = reg.ids.find(std::string(name));
    if (it == reg.ids.end()) return nullptr;
    return std::make_unique<RegisteredCapability>(it->first, it->second);
}

int registeredCapabilityId(std::string_view name) {
    CapRegistry& reg = registry();
    auto it = reg.ids.find(std::string(name));
    return it == reg.ids.end() ? -1 : it->second;
}

void installRegisterCapability(ObjectBuilder& b) {
    b.def("registerCapability", 2, jsRegisterCapability);
}

} // namespace brogameagent::api
