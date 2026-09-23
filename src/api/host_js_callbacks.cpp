#include "host_js_callbacks.h"

namespace brogameagent::api {

ev::CallResult JsCallbackSlot::call(std::span<const Value> args) const {
    if (!bound()) {
        ev::CallResult r;
        r.value = ev::undefined();
        r.thrown = true;
        return r;
    }
    return ev::call(fn.get(), ev::undefined(), args);
}

JsSlotPtr JsCallbackSet::add(Value fn) {
    auto slot = std::make_shared<JsCallbackSlot>();
    fns_.emplace_back(fn);
    slots_.push_back(slot);
    return slot;
}

void JsCallbackSet::adopt(Value handle, const std::vector<JsSlotPtr>& slots) {
    if (slots.empty()) return;
    ev::Persistent h(handle);
    ev::Persistent arr(ev::isObject(h.get()) ? ev::getProperty(h.get(), kCallbacksKey)
                                             : ev::undefined());
    for (size_t i = 0; i < slots.size(); ++i) {
        Value f = ev::isObject(arr.get())
            ? ev::getElement(arr.get(), static_cast<uint32_t>(i)) : ev::undefined();
        fns_.emplace_back(ev::isFunction(f) ? f : ev::undefined());
        slots_.push_back(slots[i]);
    }
}

Value JsCallbackSet::attach(Value self) {
    ev::Persistent selfP(self);
    if (fns_.empty()) return selfP.get();
    ev::Persistent arr(ev::makeArray(0));
    for (size_t i = 0; i < fns_.size(); ++i) {
        arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), fns_[i].get()));
    }
    selfP.set(ev::setProperty(selfP.get(), kCallbacksKey, arr.get()));
    fns_.clear();
    return selfP.get();
}

CallbackScope::CallbackScope(Value self, const std::vector<JsSlotPtr>& slots, const char* key)
    : self_(self), slots_(slots) {
    prev_.resize(slots_.size());
    if (slots_.empty()) return;
    ev::Persistent arr(ev::getProperty(self_.get(), key));
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i]) continue;
        prev_[i].set(slots_[i]->fn.get());
        Value f = ev::isObject(arr.get())
            ? ev::getElement(arr.get(), static_cast<uint32_t>(i)) : ev::undefined();
        slots_[i]->fn.set(ev::isFunction(f) ? f : ev::undefined());
    }
}

CallbackScope::~CallbackScope() {
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i]) slots_[i]->fn.set(prev_[i].get());
    }
}

} // namespace brogameagent::api
