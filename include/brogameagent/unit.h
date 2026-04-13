#pragma once

namespace brogameagent {

enum class DamageKind {
    Physical, // reduced by armor
    Magical,  // reduced by magicResist
    True      // unreduced
};

/// A damage event. Appended to World's event log whenever damage is dealt
/// through World::dealDamage, resolveAttack, or projectile impact.
struct DamageEvent {
    int attackerId;  // Unit::id; may be -1 for world/environmental damage
    int targetId;
    float amount;    // actual HP lost (post-reduction)
    DamageKind kind;
    bool killed;     // true if this damage brought the target from alive to dead
};

/// Combat / stat payload attached to every Agent. Plain-old-data so it can be
/// copied, reset, and flattened into observation vectors cheaply.
///
/// Values are in "game units" — pick a consistent scale (meters, HP points,
/// seconds) and stick to it. Observation building normalizes against the
/// per-field max on the unit itself, so ranges are policy-neutral.
///
/// Timed effects (buffs, DoTs, HoTs) live directly on Unit as fixed fields
/// rather than a polymorphic list — keeps the struct trivially copyable for
/// snapshot/restore and avoids per-agent allocations. Magnitudes are
/// overwrite-on-cast (last application wins); duration is always refreshed
/// to the new value.
struct Unit {
    static constexpr int MAX_ABILITIES = 8;

    int id = 0;
    int teamId = 0;

    float hp        = 100.0f;
    float maxHp     = 100.0f;
    float mana      = 0.0f;
    float maxMana   = 0.0f;
    float manaRegenPerSec = 1.0f;

    float damage       = 10.0f;
    float attackRange  = 5.0f;
    float attacksPerSec = 1.0f;
    float armor        = 0.0f;
    float magicResist  = 0.0f;

    float moveSpeed = 6.0f;
    float radius    = 0.4f;

    // Damage kind dealt by auto-attacks.
    DamageKind attackKind = DamageKind::Physical;

    // Time-remaining on auto-attack (0 = ready).
    float attackCooldown = 0.0f;
    // Time-remaining per ability slot (0 = ready).
    float abilityCooldowns[MAX_ABILITIES] = {0, 0, 0, 0, 0, 0, 0, 0};
    // Which registered ability (World ability id) occupies each slot. -1 = empty.
    int abilitySlot[MAX_ABILITIES] = {-1, -1, -1, -1, -1, -1, -1, -1};

    // --- Timed buffs (additive to armor/MR, multiplicative to dmg/aps/move) ---
    float armorBonus          = 0.0f;
    float armorBonusRemaining = 0.0f;
    float magicResistBonus    = 0.0f;
    float magicResistBonusRemaining = 0.0f;

    float damageMul          = 1.0f;
    float damageMulRemaining = 0.0f;
    float attacksMul         = 1.0f;
    float attacksMulRemaining = 0.0f;
    float moveSpeedMul        = 1.0f;
    float moveSpeedMulRemaining = 0.0f;

    // Stealth: probability that attacks against this unit miss (per attack roll).
    float stealthChance          = 0.0f;
    float stealthChanceRemaining = 0.0f;

    // DoT (damage-over-time) applied to this unit.
    float       dotDps       = 0.0f;
    float       dotRemaining = 0.0f;
    DamageKind  dotKind      = DamageKind::Magical;
    int         dotSourceId  = -1;   // attackerId for event attribution

    // HoT (heal-over-time) applied to this unit.
    float hotRate      = 0.0f;
    float hotRemaining = 0.0f;

    bool alive() const { return hp > 0.0f; }

    float effectiveArmor()       const { return armor       + armorBonus; }
    float effectiveMagicResist() const { return magicResist + magicResistBonus; }
    float effectiveDamage()        const { return damage        * damageMul; }
    float effectiveAttacksPerSec() const { return attacksPerSec * attacksMul; }
    float effectiveMoveSpeed()     const { return moveSpeed     * moveSpeedMul; }

    /// Apply damage honoring effective armor / magic resist. Returns actual HP lost.
    /// Standard MOBA reduction: damage * 100 / (100 + stat). Negative final
    /// stats are treated as zero (no amplification).
    float takeDamage(float amount, DamageKind kind) {
        if (!alive() || amount <= 0) return 0.0f;
        float reduced = amount;
        if (kind == DamageKind::Physical) {
            float a = effectiveArmor();
            if (a < 0) a = 0;
            reduced = amount * 100.0f / (100.0f + a);
        } else if (kind == DamageKind::Magical) {
            float mr = effectiveMagicResist();
            if (mr < 0) mr = 0;
            reduced = amount * 100.0f / (100.0f + mr);
        }
        if (reduced < 0) reduced = 0;
        float before = hp;
        hp -= reduced;
        if (hp < 0) hp = 0;
        return before - hp;
    }

    /// Advance cooldowns, buff timers, and mana regen by dt. DoT/HoT do NOT
    /// tick here — they emit damage/heal events and live on World::applyDotHot.
    void tickCooldowns(float dt) {
        attackCooldown = (attackCooldown > dt) ? attackCooldown - dt : 0.0f;
        for (int i = 0; i < MAX_ABILITIES; i++) {
            abilityCooldowns[i] = (abilityCooldowns[i] > dt)
                ? abilityCooldowns[i] - dt
                : 0.0f;
        }

        // Buffs: decrement remaining; when timer elapses, reset magnitude.
        auto decayAdd = [dt](float& remaining, float& bonus) {
            if (remaining <= 0) { bonus = 0; return; }
            remaining -= dt;
            if (remaining <= 0) { remaining = 0; bonus = 0; }
        };
        auto decayMul = [dt](float& remaining, float& mul) {
            if (remaining <= 0) { mul = 1.0f; return; }
            remaining -= dt;
            if (remaining <= 0) { remaining = 0; mul = 1.0f; }
        };
        decayAdd(armorBonusRemaining,        armorBonus);
        decayAdd(magicResistBonusRemaining,  magicResistBonus);
        decayMul(damageMulRemaining,         damageMul);
        decayMul(attacksMulRemaining,        attacksMul);
        decayMul(moveSpeedMulRemaining,      moveSpeedMul);
        decayAdd(stealthChanceRemaining,     stealthChance);

        // Mana regen.
        if (maxMana > 0) {
            mana += manaRegenPerSec * dt;
            if (mana > maxMana) mana = maxMana;
            if (mana < 0)       mana = 0;
        }
    }
};

} // namespace brogameagent
