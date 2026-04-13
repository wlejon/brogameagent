#include "brogameagent/replay_reader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>

namespace brogameagent {

using namespace replay;

namespace {

bool readFile(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
    std::FILE* f = nullptr;
#ifdef _MSC_VER
    fopen_s(&f, path.c_str(), "rb");
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) { err = "cannot open file"; return false; }
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); err = "seek failed"; return false; }
    long len = std::ftell(f);
    if (len < 0) { std::fclose(f); err = "tell failed"; return false; }
    std::rewind(f);
    out.resize(static_cast<size_t>(len));
    size_t n = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    if (n != out.size()) { err = "short read"; return false; }
    return true;
}

// Copy a POD from a byte cursor; returns false if the read would overrun.
template <typename T>
bool popPOD(const uint8_t*& cur, const uint8_t* end, T& out) {
    if (static_cast<size_t>(end - cur) < sizeof(T)) return false;
    std::memcpy(&out, cur, sizeof(T));
    cur += sizeof(T);
    return true;
}

} // namespace

bool ReplayReader::open(const std::string& path) {
    blob_.clear();
    roster_.clear();
    index_.clear();
    error_.clear();
    if (!readFile(path, blob_, error_)) return false;
    return parse_();
}

bool ReplayReader::parse_() {
    const uint8_t* base = blob_.data();
    const uint8_t* end  = base + blob_.size();

    if (blob_.size() < sizeof(FileHeader) + sizeof(Footer)) {
        error_ = "file too small"; return false;
    }

    // Header at start.
    const uint8_t* cur = base;
    if (!popPOD(cur, end, header_)) { error_ = "bad header"; return false; }
    if (header_.magic != MAGIC) { error_ = "bad magic"; return false; }
    if (header_.version != VERSION) { error_ = "unsupported version"; return false; }

    // Footer at end.
    Footer footer{};
    std::memcpy(&footer, end - sizeof(Footer), sizeof(Footer));
    if (footer.indexOffset + static_cast<uint64_t>(footer.indexCount) * sizeof(IndexEntry)
        > blob_.size() - sizeof(Footer)) {
        error_ = "index offset out of range"; return false;
    }

    // Roster immediately after header.
    uint32_t rosterCount = 0;
    if (!popPOD(cur, end, rosterCount)) { error_ = "truncated roster count"; return false; }
    roster_.resize(rosterCount);
    for (uint32_t i = 0; i < rosterCount; i++) {
        if (!popPOD(cur, end, roster_[i])) { error_ = "truncated roster"; return false; }
    }

    // Index at footer.indexOffset.
    const uint8_t* idxCur = base + footer.indexOffset;
    index_.resize(footer.indexCount);
    for (uint32_t i = 0; i < footer.indexCount; i++) {
        if (!popPOD(idxCur, end, index_[i])) { error_ = "truncated index"; return false; }
    }

    return true;
}

ReplayReader::Frame ReplayReader::frame(size_t i) const {
    Frame out{};
    if (i >= index_.size()) return out;
    const uint8_t* base = blob_.data();
    const uint8_t* end  = base + blob_.size();
    const uint8_t* cur  = base + index_[i].offset;
    if (!popPOD(cur, end, out.header)) return out;
    out.agents.resize(out.header.liveCount);
    for (auto& a : out.agents) popPOD(cur, end, a);
    out.projectiles.resize(out.header.projCount);
    for (auto& p : out.projectiles) popPOD(cur, end, p);
    out.events.resize(out.header.eventCount);
    for (auto& e : out.events) popPOD(cur, end, e);
    return out;
}

size_t ReplayReader::findByStep(uint32_t stepIdx) const {
    for (size_t i = 0; i < index_.size(); i++) {
        if (index_[i].stepIdx == stepIdx) return i;
    }
    return SIZE_MAX;
}

std::vector<ReplayReader::TrajectoryPoint>
ReplayReader::trajectory(int32_t agentId) const {
    std::vector<TrajectoryPoint> out;
    out.reserve(index_.size());
    for (size_t i = 0; i < index_.size(); i++) {
        Frame f = frame(i);
        for (const AgentState& a : f.agents) {
            if (a.id == agentId) {
                TrajectoryPoint p{};
                p.stepIdx = f.header.stepIdx;
                p.elapsed = f.header.elapsed;
                p.x       = a.x;
                p.z       = a.z;
                p.hp      = a.hp;
                p.alive   = (a.flags & AGENT_FLAG_ALIVE) != 0;
                out.push_back(p);
                break;
            }
        }
    }
    return out;
}

std::vector<ReplayReader::DamageSummary> ReplayReader::damageSummary() const {
    std::map<std::pair<int32_t,int32_t>, DamageSummary> agg;
    for (size_t i = 0; i < index_.size(); i++) {
        Frame f = frame(i);
        for (const DamageEventRec& e : f.events) {
            auto key = std::make_pair(e.attackerId, e.targetId);
            auto& s = agg[key];
            s.attackerId  = e.attackerId;
            s.targetId    = e.targetId;
            s.totalDamage += e.amount;
            s.hits        += 1;
            s.kills       += e.killed ? 1u : 0u;
        }
    }
    std::vector<DamageSummary> out;
    out.reserve(agg.size());
    for (auto& kv : agg) out.push_back(kv.second);
    std::sort(out.begin(), out.end(), [](const DamageSummary& a, const DamageSummary& b) {
        return a.totalDamage > b.totalDamage;
    });
    return out;
}

} // namespace brogameagent
