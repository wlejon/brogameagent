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
    framesEnd_ = 0;
    if (!readFile(path, blob_, error_)) return false;
    if (parse_()) return true;
    // A rejected file leaves the reader empty, not half-populated.
    blob_.clear();
    roster_.clear();
    index_.clear();
    framesEnd_ = 0;
    return false;
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

    // Footer at end. Every count and offset below comes from the file, so
    // each is checked against the bytes actually present before it sizes an
    // allocation or positions a cursor: a corrupt or hostile file must fail
    // to open, not allocate gigabytes or read past the blob.
    Footer footer{};
    std::memcpy(&footer, end - sizeof(Footer), sizeof(Footer));
    const uint64_t bodyEnd = blob_.size() - sizeof(Footer);  // index must end here

    // Roster immediately after header.
    uint32_t rosterCount = 0;
    if (!popPOD(cur, end, rosterCount)) { error_ = "truncated roster count"; return false; }
    const uint64_t rosterAvail = static_cast<uint64_t>(end - cur);
    if (rosterCount > kMaxRoster ||
        static_cast<uint64_t>(rosterCount) * sizeof(AgentStatic) > rosterAvail) {
        error_ = "roster count out of range"; return false;
    }
    roster_.resize(rosterCount);
    for (uint32_t i = 0; i < rosterCount; i++) {
        if (!popPOD(cur, end, roster_[i])) { error_ = "truncated roster"; return false; }
    }
    const uint64_t framesBegin = static_cast<uint64_t>(cur - base);

    // Index at footer.indexOffset: after the roster, and ending exactly
    // where the footer begins (the writer emits nothing between them).
    if (footer.indexOffset < framesBegin || footer.indexOffset > bodyEnd) {
        error_ = "index offset out of range"; return false;
    }
    if (static_cast<uint64_t>(footer.indexCount) * sizeof(IndexEntry)
        != bodyEnd - footer.indexOffset) {
        error_ = "index count out of range"; return false;
    }
    const uint8_t* idxCur = base + footer.indexOffset;
    index_.resize(footer.indexCount);
    for (uint32_t i = 0; i < footer.indexCount; i++) {
        if (!popPOD(idxCur, end, index_[i])) { error_ = "truncated index"; return false; }
        // A frame's header lies wholly in the frame stream (between the
        // roster and the index); frame() checks its body against the same
        // bound.
        const IndexEntry& e = index_[i];
        if (e.offset < framesBegin || e.offset > footer.indexOffset ||
            footer.indexOffset - e.offset < sizeof(FrameHeader)) {
            error_ = "frame offset out of range"; return false;
        }
    }
    framesEnd_ = footer.indexOffset;

    return true;
}

ReplayReader::Frame ReplayReader::frame(size_t i) const {
    Frame out{};
    if (i >= index_.size()) return out;
    // parse_() validated the offset; the frame body must also end before the
    // index, so a count the header overstates yields an empty frame rather
    // than records read out of the index table (or past the blob).
    const uint8_t* base = blob_.data();
    const uint8_t* end  = base + framesEnd_;
    const uint8_t* cur  = base + index_[i].offset;
    FrameHeader fh{};
    if (!popPOD(cur, end, fh)) return out;
    const uint64_t body =
        static_cast<uint64_t>(fh.liveCount)  * sizeof(AgentState) +
        static_cast<uint64_t>(fh.projCount)  * sizeof(ProjectileState) +
        static_cast<uint64_t>(fh.eventCount) * sizeof(DamageEventRec);
    if (body > static_cast<uint64_t>(end - cur)) return out;
    out.header = fh;
    out.agents.resize(fh.liveCount);
    for (auto& a : out.agents) popPOD(cur, end, a);
    out.projectiles.resize(fh.projCount);
    for (auto& p : out.projectiles) popPOD(cur, end, p);
    out.events.resize(fh.eventCount);
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
