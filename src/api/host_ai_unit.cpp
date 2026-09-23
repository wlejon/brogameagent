// AIUnit implementation for brogameagent_api.
// Unit data proxy for AIAgent.

#include "host_ai_internal.h"

namespace brogameagent::api {

HostClass g_unitClass;

void decorateUnitProto(ObjectBuilder& b) {
    b.accessor("id",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().id : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().id = static_cast<int>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("teamId",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().teamId : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().teamId = static_cast<int>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("hp",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().hp : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().hp = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("maxHp",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().maxHp : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().maxHp = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("mana",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().mana : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().mana = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("maxMana",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().maxMana : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().maxMana = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("damage",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().damage : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().damage = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("attackRange",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().attackRange : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().attackRange = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("attacksPerSec",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().attacksPerSec : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().attacksPerSec = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("armor",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().armor : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().armor = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("magicResist",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().magicResist : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().magicResist = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("moveSpeed",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().moveSpeed : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().moveSpeed = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("radius",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().radius : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().radius = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("manaRegenPerSec",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromDouble(u && u->agent() ? u->agent()->unit().manaRegenPerSec : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty()) u->agent()->unit().manaRegenPerSec = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("alive", [](Value self, std::span<const Value>) -> Value {
        auto* u = unwrapUnit(self);
        return ev::fromBool(u && u->agent() && u->agent()->unit().alive());
    }, nullptr);

    b.accessor("effectiveArmor", [](Value self, std::span<const Value>) -> Value {
        auto* u = unwrapUnit(self);
        return ev::fromDouble(u && u->agent() ? u->agent()->unit().effectiveArmor() : 0);
    }, nullptr);

    b.accessor("effectiveDamage", [](Value self, std::span<const Value>) -> Value {
        auto* u = unwrapUnit(self);
        return ev::fromDouble(u && u->agent() ? u->agent()->unit().effectiveDamage() : 0);
    }, nullptr);

    b.accessor("effectiveMoveSpeed", [](Value self, std::span<const Value>) -> Value {
        auto* u = unwrapUnit(self);
        return ev::fromDouble(u && u->agent() ? u->agent()->unit().effectiveMoveSpeed() : 0);
    }, nullptr);

    b.accessor("effectiveMagicResist", [](Value self, std::span<const Value>) -> Value {
        auto* u = unwrapUnit(self);
        return ev::fromDouble(u && u->agent() ? u->agent()->unit().effectiveMagicResist() : 0);
    }, nullptr);

    b.accessor("effectiveAttacksPerSec", [](Value self, std::span<const Value>) -> Value {
        auto* u = unwrapUnit(self);
        return ev::fromDouble(u && u->agent() ? u->agent()->unit().effectiveAttacksPerSec() : 0);
    }, nullptr);

    b.def("tickCooldowns", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* u = unwrapUnit(self);
        if (u && u->agent() && !a.empty()) u->agent()->unit().tickCooldowns(static_cast<float>(numAt(a, 0)));
        return ev::undefined();
    });

    b.def("setAbilitySlot", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* u = unwrapUnit(self);
        if (u && u->agent() && a.size() >= 2) {
            int slot = static_cast<int>(numAt(a, 0)), abilityId = static_cast<int>(numAt(a, 1));
            if (slot >= 0 && slot < brogameagent::Unit::MAX_ABILITIES) {
                u->agent()->unit().abilitySlot[slot] = abilityId;
            }
        }
        return ev::undefined();
    });

    b.def("getAbilitySlot", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* u = unwrapUnit(self);
        if (!u || !u->agent() || a.empty()) return ev::fromDouble(-1);
        int slot = static_cast<int>(numAt(a, 0));
        if (slot < 0 || slot >= brogameagent::Unit::MAX_ABILITIES) return ev::fromDouble(-1);
        return ev::fromDouble(u->agent()->unit().abilitySlot[slot]);
    });

    b.def("getAbilityCooldown", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* u = unwrapUnit(self);
        if (!u || !u->agent() || a.empty()) return ev::fromDouble(0.0);
        int slot = static_cast<int>(numAt(a, 0));
        if (slot < 0 || slot >= brogameagent::Unit::MAX_ABILITIES) return ev::fromDouble(0.0);
        return ev::fromDouble(u->agent()->unit().abilityCooldowns[slot]);
    });

    b.def("takeDamage", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* u = unwrapUnit(self);
        if (!u || !u->agent() || a.empty()) return ev::fromDouble(0.0);
        float amount = static_cast<float>(numAt(a, 0));
        std::string kindStr = (a.size() >= 2 && ev::isString(a[1])) ? ev::toUtf8(a[1]) : "physical";
        float actual = u->agent()->unit().takeDamage(amount, parseDamageKind(kindStr.c_str()));
        return ev::fromDouble(actual);
    });

    // Buffs
    auto bindBuff = [&](const char* name, const char* nameRem, float brogameagent::Unit::* val, float brogameagent::Unit::* rem, float defVal = 0.0f) {
        b.accessor(name,
            [val, defVal](Value s, std::span<const Value>) -> Value {
                auto* u = unwrapUnit(s); return ev::fromDouble(u && u->agent() ? (u->agent()->unit().*val) : defVal);
            },
            [val](Value s, std::span<const Value> a) -> Value {
                auto* u = unwrapUnit(s); if (u && u->agent() && !a.empty()) (u->agent()->unit().*val) = static_cast<float>(numAt(a, 0));
                return ev::undefined();
            });
        b.accessor(nameRem,
            [rem](Value s, std::span<const Value>) -> Value {
                auto* u = unwrapUnit(s); return ev::fromDouble(u && u->agent() ? (u->agent()->unit().*rem) : 0.0);
            },
            [rem](Value s, std::span<const Value> a) -> Value {
                auto* u = unwrapUnit(s); if (u && u->agent() && !a.empty()) (u->agent()->unit().*rem) = static_cast<float>(numAt(a, 0));
                return ev::undefined();
            });
    };

    bindBuff("armorBonus", "armorBonusRemaining", &brogameagent::Unit::armorBonus, &brogameagent::Unit::armorBonusRemaining);
    bindBuff("magicResistBonus", "magicResistBonusRemaining", &brogameagent::Unit::magicResistBonus, &brogameagent::Unit::magicResistBonusRemaining);
    bindBuff("damageMul", "damageMulRemaining", &brogameagent::Unit::damageMul, &brogameagent::Unit::damageMulRemaining, 1.0f);
    bindBuff("attacksMul", "attacksMulRemaining", &brogameagent::Unit::attacksMul, &brogameagent::Unit::attacksMulRemaining, 1.0f);
    bindBuff("moveSpeedMul", "moveSpeedMulRemaining", &brogameagent::Unit::moveSpeedMul, &brogameagent::Unit::moveSpeedMulRemaining, 1.0f);
    bindBuff("stealthChance", "stealthChanceRemaining", &brogameagent::Unit::stealthChance, &brogameagent::Unit::stealthChanceRemaining);
    bindBuff("dotDps", "dotRemaining", &brogameagent::Unit::dotDps, &brogameagent::Unit::dotRemaining);
    bindBuff("hotRate", "hotRemaining", &brogameagent::Unit::hotRate, &brogameagent::Unit::hotRemaining);

    b.accessor("dotSourceId",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self); return ev::fromDouble(u && u->agent() ? u->agent()->unit().dotSourceId : -1);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self); if (u && u->agent() && !a.empty()) u->agent()->unit().dotSourceId = static_cast<int>(numAt(a, 0));
            return ev::undefined();
        });

    b.accessor("dotKind",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromUtf8(u && u->agent() ? damageKindStr(u->agent()->unit().dotKind) : "physical");
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty() && ev::isString(a[0])) {
                u->agent()->unit().dotKind = parseDamageKind(ev::toUtf8(a[0]).c_str());
            }
            return ev::undefined();
        });

    b.accessor("attackKind",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self);
            return ev::fromUtf8(u && u->agent() ? damageKindStr(u->agent()->unit().attackKind) : "physical");
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self);
            if (u && u->agent() && !a.empty() && ev::isString(a[0])) {
                u->agent()->unit().attackKind = parseDamageKind(ev::toUtf8(a[0]).c_str());
            }
            return ev::undefined();
        });

    b.accessor("attackCooldown",
        [](Value self, std::span<const Value>) -> Value {
            auto* u = unwrapUnit(self); return ev::fromDouble(u && u->agent() ? u->agent()->unit().attackCooldown : 0);
        },
        [](Value self, std::span<const Value> a) -> Value {
            auto* u = unwrapUnit(self); if (u && u->agent() && !a.empty()) u->agent()->unit().attackCooldown = static_cast<float>(numAt(a, 0));
            return ev::undefined();
        });
}

Value makeUnitHandle(HostAgent* owner, Value agentVal) {
    if (!owner) return ev::undefined();
    ev::Persistent agentP(agentVal);
    ensureAIClassesInstalled();
    auto* cell = new HostUnit();
    cell->owner = owner;
    cell->ownerLife = owner->life;
    // The proxy keeps its agent alive through `_agent` on its own handle, an
    // edge rather than a root: `agent.u = agent.unit` stays collectable.
    ev::Persistent unit(g_unitClass.make(cell, [](void* p) { delete static_cast<HostUnit*>(p); }));
    if (!ev::isUndefined(agentP.get())) unit.set(ev::setProperty(unit.get(), "_agent", agentP.get()));
    return unit.get();
}

} // namespace brogameagent::api
