#pragma once

// JS callbacks a native adapter can call without holding a root to them.
//
// A host root (ev::Persistent) is invisible to the collector as an edge, so a
// native search that roots a JS callback which closes over the handle owning
// the search — `this.mcts = createMcts({ evaluator: w => this.score(w) })` —
// pins both forever. Instead each callback lives on the owning handle's own
// JS object (`_callbacks[i]`, an edge the collector traces), and the native
// adapter holds a JsCallbackSlot that is bound to that function only while a
// CallbackScope for the owner is on the stack. Every method that can run the
// native search opens one; outside a scope a slot is empty and the adapter
// answers its neutral default.

#include "embed/embed.h"

#include <memory>
#include <span>
#include <vector>

namespace brogameagent::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

/// The property on an owning handle that holds its callbacks, in slot order.
inline constexpr const char* kCallbacksKey = "_callbacks";

struct JsCallbackSlot {
    ev::Persistent fn;  // undefined outside a CallbackScope

    bool bound() const { return ev::isFunction(fn.get()); }
    /// Calls the bound function (`this` undefined); a thrown result when
    /// nothing is bound. `args` must be slot reads (current addresses).
    ev::CallResult call(std::span<const Value> args) const;
};
using JsSlotPtr = std::shared_ptr<JsCallbackSlot>;

/// Collects the callbacks a factory parses out of its options, then moves
/// them onto the new handle. The functions are rooted here only until
/// attach().
class JsCallbackSet {
public:
    JsCallbackSet() = default;
    JsCallbackSet(const JsCallbackSet&) = delete;
    JsCallbackSet& operator=(const JsCallbackSet&) = delete;

    /// A new slot for `fn` (which must be a function).
    JsSlotPtr add(Value fn);

    /// Takes on another handle's slots (an Option passed in opts.options),
    /// reading their functions from that handle's `_callbacks`.
    void adopt(Value handle, const std::vector<JsSlotPtr>& slots);

    /// Stores the collected functions on `self` as `_callbacks` and answers
    /// self's post-call address. The slots stay with the set (takeSlots).
    Value attach(Value self);

    std::vector<JsSlotPtr> takeSlots() { return std::move(slots_); }

private:
    std::vector<JsSlotPtr> slots_;
    std::vector<ev::Persistent> fns_;
};

/// Binds `slots[i]` to `self[key][i]` for the scope's lifetime and roots
/// `self` meanwhile; the previous bindings come back on exit, so a callback
/// that re-enters its own search leaves the outer call's bindings intact.
/// A null entry in `slots` is skipped.
class CallbackScope {
public:
    CallbackScope(Value self, const std::vector<JsSlotPtr>& slots,
                  const char* key = kCallbacksKey);
    ~CallbackScope();
    CallbackScope(const CallbackScope&) = delete;
    CallbackScope& operator=(const CallbackScope&) = delete;

    /// The rooted owner, current across allocations.
    Value self() const { return self_.get(); }

private:
    ev::Persistent self_;
    std::vector<JsSlotPtr> slots_;
    std::vector<ev::Persistent> prev_;
};

} // namespace brogameagent::api
