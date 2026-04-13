#pragma once

namespace brogameagent {

/// Combat / stat payload attached to every Agent. Plain-old-data so it can be
/// copied, reset, and flattened into observation vectors cheaply.
///
/// Values are in "game units" — pick a consistent scale (meters, HP points,
/// seconds) and stick to it. Observation building normalizes against the
/// per-field max on the unit itself, so ranges are policy-neutral.
struct Unit {
    static constexpr int MAX_ABILITIES = 4;

    int id = 0;
    int teamId = 0;

    float hp        = 100.0f;
    float maxHp     = 100.0f;
    float mana      = 0.0f;
    float maxMana   = 0.0f;

    float damage       = 10.0f;
    float attackRange  = 5.0f;
    float attacksPerSec = 1.0f;
    float armor        = 0.0f;
    float magicResist  = 0.0f;

    float moveSpeed = 6.0f;
    float radius    = 0.4f;

    // Time-remaining on auto-attack (0 = ready).
    float attackCooldown = 0.0f;
    // Time-remaining per ability slot (0 = ready).
    float abilityCooldowns[MAX_ABILITIES] = {0, 0, 0, 0};

    bool alive() const { return hp > 0.0f; }

    /// Advance cooldowns by dt; never below zero.
    void tickCooldowns(float dt) {
        attackCooldown = (attackCooldown > dt) ? attackCooldown - dt : 0.0f;
        for (int i = 0; i < MAX_ABILITIES; i++) {
            abilityCooldowns[i] = (abilityCooldowns[i] > dt)
                ? abilityCooldowns[i] - dt
                : 0.0f;
        }
    }
};

} // namespace brogameagent
