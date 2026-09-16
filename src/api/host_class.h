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

class HostClass {
public:
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

private:
    ev::Persistent* proto_ = nullptr;
    ev::Persistent* ctor_ = nullptr;
};

} // namespace brogameagent::api
