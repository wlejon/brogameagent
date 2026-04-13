#include "brogameagent/recorder.h"
#include "brogameagent/agent.h"
#include "brogameagent/world.h"

#include <cstring>

namespace brogameagent {

using namespace replay;

namespace {

template <typename T>
bool writeRaw(std::FILE* f, const T& v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
bool writeArray(std::FILE* f, const T* p, size_t n) {
    if (n == 0) return true;
    return std::fwrite(p, sizeof(T), n, f) == n;
}

uint64_t currentOffset(std::FILE* f) {
    // ftell returns long; cast to uint64 for our footer fields. We never seek
    // backwards, so even on 32-bit long platforms this is safe up to 2 GiB
    // (replay files we target are well under that).
    return static_cast<uint64_t>(std::ftell(f));
}

} // namespace

Recorder::~Recorder() {
    if (file_) close();
}

bool Recorder::open(const std::string& path,
                    uint64_t episodeId, uint64_t seed, float dt)
{
    if (file_) return false;
#ifdef _MSC_VER
    fopen_s(&file_, path.c_str(), "wb");
#else
    file_ = std::fopen(path.c_str(), "wb");
#endif
    if (!file_) return false;

    FileHeader hdr{};
    hdr.magic     = MAGIC;
    hdr.version   = VERSION;
    hdr.episodeId = episodeId;
    hdr.seed      = seed;
    hdr.dt        = dt;
    // reserved already zero-initialized
    if (!writeRaw(file_, hdr)) {
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }
    index_.clear();
    lastEventIdx_ = 0;
    rosterWritten_ = false;
    return true;
}

void Recorder::writeRoster(const std::vector<Agent*>& agents) {
    if (!file_ || rosterWritten_) return;
    uint32_t n = static_cast<uint32_t>(agents.size());
    writeRaw(file_, n);
    for (Agent* a : agents) {
        AgentStatic s{};
        s.id          = a->unit().id;
        s.teamId      = a->unit().teamId;
        s.maxHp       = a->unit().maxHp;
        s.maxMana     = a->unit().maxMana;
        s.radius      = a->unit().radius;
        s.attackRange = a->unit().attackRange;
        writeRaw(file_, s);
    }
    rosterWritten_ = true;
}

void Recorder::recordFrame(uint32_t stepIdx, float elapsed, const World& world) {
    if (!file_) return;
    if (!rosterWritten_) {
        // Emit an empty roster so the file is still well-formed.
        uint32_t zero = 0;
        writeRaw(file_, zero);
        rosterWritten_ = true;
    }

    // Collect agents.
    std::vector<AgentState> agentStates;
    agentStates.reserve(world.agents().size());
    for (Agent* a : world.agents()) {
        AgentState st{};
        st.id             = a->unit().id;
        st.x              = a->x();
        st.z              = a->z();
        Vec2 v            = a->velocity();
        st.vx             = v.x;
        st.vz             = v.z;
        st.yaw            = a->yaw();
        st.aimYaw         = a->aimYaw();
        st.hp             = a->unit().hp;
        st.mana           = a->unit().mana;
        st.attackCooldown = a->unit().attackCooldown;
        st.flags          = a->unit().alive() ? AGENT_FLAG_ALIVE : 0u;
        agentStates.push_back(st);
    }

    // Collect projectiles (live only — dead ones are typically culled at
    // tick end anyway, but this makes the recording robust either way).
    std::vector<ProjectileState> projStates;
    projStates.reserve(world.projectiles().size());
    for (const Projectile& p : world.projectiles()) {
        ProjectileState s{};
        s.id       = p.id;
        s.ownerId  = p.ownerId;
        s.teamId   = p.teamId;
        s.x        = p.x;
        s.z        = p.z;
        s.vx       = p.vx;
        s.vz       = p.vz;
        s.mode     = static_cast<uint8_t>(p.mode);
        s.alive    = p.alive ? 1u : 0u;
        s.pad      = 0;
        projStates.push_back(s);
    }

    // Slice damage events since last frame.
    const auto& ev = world.events();
    std::vector<DamageEventRec> eventRecs;
    if (ev.size() > lastEventIdx_) {
        size_t n = ev.size() - lastEventIdx_;
        eventRecs.reserve(n);
        for (size_t i = lastEventIdx_; i < ev.size(); i++) {
            DamageEventRec r{};
            r.attackerId = ev[i].attackerId;
            r.targetId   = ev[i].targetId;
            r.amount     = ev[i].amount;
            r.kind       = static_cast<uint8_t>(ev[i].kind);
            r.killed     = ev[i].killed ? 1u : 0u;
            r.pad        = 0;
            eventRecs.push_back(r);
        }
        lastEventIdx_ = ev.size();
    }

    // Note the offset where this frame starts, for the index.
    uint64_t off = currentOffset(file_);

    FrameHeader fh{};
    fh.stepIdx    = stepIdx;
    fh.elapsed    = elapsed;
    fh.liveCount  = static_cast<uint16_t>(agentStates.size());
    fh.projCount  = static_cast<uint16_t>(projStates.size());
    fh.eventCount = static_cast<uint16_t>(eventRecs.size());
    fh.reserved   = 0;
    writeRaw(file_, fh);
    writeArray(file_, agentStates.data(), agentStates.size());
    writeArray(file_, projStates.data(),  projStates.size());
    writeArray(file_, eventRecs.data(),   eventRecs.size());

    IndexEntry ie{};
    ie.stepIdx  = stepIdx;
    ie.reserved = 0;
    ie.offset   = off;
    index_.push_back(ie);
}

bool Recorder::close() {
    if (!file_) return true;

    // Index table.
    uint64_t indexOff = currentOffset(file_);
    writeArray(file_, index_.data(), index_.size());

    // Footer.
    Footer footer{};
    footer.indexOffset = indexOff;
    footer.indexCount  = static_cast<uint32_t>(index_.size());
    footer.reserved    = 0;
    writeRaw(file_, footer);

    int rc = std::fclose(file_);
    file_ = nullptr;
    index_.clear();
    lastEventIdx_ = 0;
    rosterWritten_ = false;
    return rc == 0;
}

} // namespace brogameagent
