#pragma once

#include "embed/embed.h"

#include <exception>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace brogameagent::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

/// A native method body with every C++ exception turned into a JS Error. The
/// caller may be compiled JS, whose frames carry no unwind metadata, so an
/// exception escaping a body (bad_alloc on a corrupt replay's counts, a
/// library's invalid_argument) would otherwise end the process.
inline ev::NativeFn guardNative(ev::NativeFn fn) {
    return [fn = std::move(fn)](Value self, std::span<const Value> args) -> Value {
        try {
            return fn(self, args);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        } catch (...) {
            return ev::throwError("brogameagent: native error");
        }
    };
}

/// Helper to build objects and namespaces property by property using bronze::embed.
/// Handles moving GC by rooting the target in an ev::Persistent.
struct ObjectBuilder {
    ev::Persistent obj;

    ObjectBuilder() : obj(ev::createObject()) {}
    explicit ObjectBuilder(Value existing) : obj(existing) {}

    void set(std::string_view name, Value v) {
        obj.set(ev::setProperty(obj.get(), name, v));
    }

    void set(std::string_view name, double d) {
        set(name, ev::fromDouble(d));
    }

    void set(std::string_view name, bool b) {
        set(name, ev::fromBool(b));
    }

    void set(std::string_view name, const std::string& s) {
        set(name, ev::fromUtf8(s));
    }

    void set(std::string_view name, const char* s) {
        set(name, ev::fromUtf8(s));
    }

    void def(std::string_view name, uint32_t arity, ev::NativeFn fn) {
        Value f = ev::makeFunction(guardNative(std::move(fn)), arity, name);
        obj.set(ev::setProperty(obj.get(), name, f));
    }

    void accessor(std::string_view name, ev::NativeFn getter, ev::NativeFn setter = nullptr) {
        const std::string getName = "get " + std::string(name);
        const std::string setName = "set " + std::string(name);
        ev::Persistent g(ev::makeFunction(guardNative(std::move(getter)), 0, getName));
        Value s = setter ? ev::makeFunction(guardNative(std::move(setter)), 1, setName)
                         : ev::undefined();
        obj.set(ev::defineAccessor(obj.get(), name, g.get(), s, /*enumerable=*/true));
    }

    Value get() const { return obj.get(); }
    Value build() const { return obj.get(); }
};

} // namespace brogameagent::api
