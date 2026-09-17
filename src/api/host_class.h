#pragma once

#include "embed/embed.h"
#include "object_builder.h"

#include <functional>
#include <memory>
#include <span>
#include <string>

namespace brogameagent::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

// A host class: one constructor function + one prototype, with instances
// born on that prototype by make()/createInstance().
//
// The class objects are process-global (`HostClass g_navMeshClass`), but
// what they hold is PER THREAD: bronze's runtime is per-thread, and a
// Persistent is a slot in its creating thread's registry, so a constructor
// made on the main thread means nothing to a Worker's realm. Every accessor
// reads the CALLING thread's slots and install() fills the calling thread's;
// a realm installs each class once, and installed() answers for the calling
// thread, which is what an install guard has to ask.
class HostClass {
public:
    struct Slots {
        ev::Persistent* proto = nullptr;
        ev::Persistent* ctor = nullptr;
    };

    void install(const char* name, uint32_t arity, ev::NativeFn body,
                 const std::function<void(ObjectBuilder&)>& decorate = nullptr);

    void init(const char* name, const std::function<void(ObjectBuilder&)>& decorate) {
        install(name, 0, nullptr, decorate);
    }

    template <typename T>
    Value createInstance(std::unique_ptr<T> ptr) const {
        return make(ptr.release(), [](void* p) { delete static_cast<T*>(p); });
    }

    void alias(const char* name) const;
    void inherit(const HostClass& base) const;

    Value make(void* data, ev::HandleDestructor dtor,
               ev::Finalize when = ev::Finalize::InSweep) const;

    void setStatic(const char* name, Value v) const;

    void* unwrap(Value val) const { return ev::handleData(val); }

    Value prototype() const;
    Value constructor() const;

    // Whether install() has run on the CALLING thread.
    bool installed() const;

private:
    Slots& slots() const;
    const Slots* slotsIfAny() const;
};

} // namespace brogameagent::api
