// Replay file schema for brogameagent recordings.
//
// This header is the ground truth for the .bgar on-disk format. External
// readers (e.g. the `bro` renderer) should include THIS header at compile
// time so any schema change is a build-time mismatch instead of silent data
// corruption. The header has no dependencies beyond <cstdint>.
//
// File layout (streamed, little-endian — x86 / ARM native):
//
//   FileHeader
//   uint32_t rosterCount
//   AgentStatic[rosterCount]
//   Frame*  (stream, count == Footer::indexCount)
//     FrameHeader
//     AgentState[liveCount]
//     ProjectileState[projCount]
//     DamageEventRec[eventCount]
//   IndexEntry[Footer::indexCount]
//   Footer  (last sizeof(Footer) bytes of the file)
//
// To random-access: seek to EOF - sizeof(Footer), read Footer, seek to
// Footer::indexOffset, read the index table, then seek to the desired
// frame's offset.
#pragma once

#include <cstdint>

namespace brogameagent::replay {

// 'BGAR' packed little-endian: 'B'=0x42 'G'=0x47 'A'=0x41 'R'=0x52.
inline constexpr uint32_t MAGIC   = 0x52414742u;
inline constexpr uint32_t VERSION = 1u;

// Bit flags on AgentState::flags.
inline constexpr uint32_t AGENT_FLAG_ALIVE = 1u << 0;

#pragma pack(push, 1)

struct FileHeader {
    uint32_t magic;        // MAGIC
    uint32_t version;      // VERSION
    uint64_t episodeId;    // caller-assigned (e.g. training episode number)
    uint64_t seed;         // World RNG seed at episode start (informational)
    float    dt;           // fixed step size in seconds
    uint32_t reserved[3];  // zero-filled
};

struct AgentStatic {
    int32_t id;
    int32_t teamId;
    float   maxHp;
    float   maxMana;
    float   radius;
    float   attackRange;
};

struct FrameHeader {
    uint32_t stepIdx;
    float    elapsed;
    uint16_t liveCount;    // number of AgentState records that follow
    uint16_t projCount;    // number of ProjectileState records
    uint16_t eventCount;   // number of DamageEventRec records
    uint16_t reserved;     // zero
};

struct AgentState {
    int32_t  id;
    float    x, z;
    float    vx, vz;
    float    yaw;
    float    aimYaw;
    float    hp;
    float    mana;
    float    attackCooldown;
    uint32_t flags;        // AGENT_FLAG_*
};

struct ProjectileState {
    int32_t  id;
    int32_t  ownerId;
    int32_t  teamId;
    float    x, z;
    float    vx, vz;
    uint8_t  mode;         // matches ProjectileMode enum ordinals
    uint8_t  alive;        // 0 / 1
    uint16_t pad;
};

struct DamageEventRec {
    int32_t  attackerId;
    int32_t  targetId;
    float    amount;
    uint8_t  kind;         // matches DamageKind enum ordinals
    uint8_t  killed;       // 0 / 1
    uint16_t pad;
};

struct IndexEntry {
    uint32_t stepIdx;
    uint32_t reserved;
    uint64_t offset;       // absolute byte offset of the frame's FrameHeader
};

struct Footer {
    uint64_t indexOffset;  // absolute byte offset of the IndexEntry array
    uint32_t indexCount;   // number of frames / IndexEntry records
    uint32_t reserved;
};

#pragma pack(pop)

} // namespace brogameagent::replay
