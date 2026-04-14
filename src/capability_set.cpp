#include "brogameagent/capability.h"

namespace brogameagent {

void CapabilitySet::add(std::unique_ptr<Capability> cap) {
    if (!cap) return;
    const int id = cap->id();
    for (auto& slot : caps_) {
        if (slot && slot->id() == id) { slot = std::move(cap); return; }
    }
    caps_.push_back(std::move(cap));
}

bool CapabilitySet::remove(int id) {
    for (auto it = caps_.begin(); it != caps_.end(); ++it) {
        if (*it && (*it)->id() == id) { caps_.erase(it); return true; }
    }
    return false;
}

Capability* CapabilitySet::get(int id) {
    for (auto& c : caps_) if (c && c->id() == id) return c.get();
    return nullptr;
}

const Capability* CapabilitySet::get(int id) const {
    for (const auto& c : caps_) if (c && c->id() == id) return c.get();
    return nullptr;
}

uint32_t CapabilitySet::buildBuiltinMask(const CapContext& ctx) const {
    uint32_t mask = 0;
    for (const auto& c : caps_) {
        if (!c) continue;
        const int id = c->id();
        if (id < 0 || id >= 32) continue;
        if (c->gate(ctx)) mask |= (1u << id);
    }
    return mask;
}

void addAllBuiltinCapabilities(CapabilitySet& set) {
    set.add(makeMoveToCapability());
    set.add(makeLaneWalkCapability());
    set.add(makeBasicAttackCapability());
    set.add(makeCastAbilityCapability());
    set.add(makeFleeCapability());
    set.add(makeHoldCapability());
}

} // namespace brogameagent
