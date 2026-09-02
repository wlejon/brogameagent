#include "brogameagent/hex_nav.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace brogameagent {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr float kInfF = std::numeric_limits<float>::infinity();
constexpr char kKeySep = '\x1f';
} // namespace

// odd-r offset deltas, indexed by row parity then direction
// (0=E 1=NE 2=NW 3=W 4=SW 5=SE).
const int HexNav::STEP[2][6][2] = {
    {{1, 0}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}},   // even rows
    {{1, 0}, {1, -1}, {0, -1}, {-1, 0}, {0, 1}, {1, 1}}      // odd rows
};

HexNav::HexNav(int size) : size_(size < 1 ? 1 : size) {
    const size_t n = static_cast<size_t>(size_) * static_cast<size_t>(size_);
    cost_.assign(n, kInfF);
    parent_.assign(n, -1);
    touched_.reserve(n);
    heap_.reserve(1024);
}

// ── Tables ─────────────────────────────────────────────────────────────────

bool HexNav::setStepCosts(const std::string& id, const double* costs, size_t n) {
    if (!costs || n != static_cast<size_t>(cells()) * 6) return false;
    tables_[id].assign(costs, costs + n);
    invalidateComponents(id);
    return true;
}

bool HexNav::updateStepCosts(const std::string& id, const int32_t* cellIdx, size_t nCells,
                             const double* values) {
    auto it = tables_.find(id);
    if (it == tables_.end() || (!cellIdx && nCells) || (!values && nCells)) return false;
    std::vector<double>& t = it->second;
    const int32_t n = cells();
    for (size_t k = 0; k < nCells; k++) {
        const int32_t c = cellIdx[k];
        if (c < 0 || c >= n) continue;
        for (int d = 0; d < 6; d++) t[static_cast<size_t>(c) * 6 + d] = values[k * 6 + d];
    }
    invalidateComponents(id);
    return true;
}

bool HexNav::hasStepCosts(const std::string& id) const {
    return tables_.find(id) != tables_.end();
}

const std::vector<double>* HexNav::stepCosts(const std::string& id) const {
    auto it = tables_.find(id);
    return it == tables_.end() ? nullptr : &it->second;
}

bool HexNav::setClearance(const std::string& id, const uint8_t* table, size_t n) {
    if (!table || n != static_cast<size_t>(cells())) return false;
    clearance_[id].assign(table, table + n);
    // A clearance table is half of a (table, clearance) component key; every
    // cached labelling that used this id is stale.
    for (auto it = components_.begin(); it != components_.end();) {
        const size_t sep = it->first.find(kKeySep);
        if (sep != std::string::npos && it->first.compare(sep + 1, std::string::npos, id) == 0)
            it = components_.erase(it);
        else
            ++it;
    }
    return true;
}

const std::vector<uint8_t>& HexNav::buildClearance(const std::string& id, int radius,
                                                   const uint8_t* passable, const int8_t* elevation,
                                                   const int16_t* floors, int crushFloors) {
    const int size = size_;
    const int n = cells();
    std::vector<uint8_t> out(static_cast<size_t>(n), 3);

    // Cube offsets of the radius-r disk.
    std::vector<std::pair<int, int>> disk;
    if (radius < 0) radius = 0;
    disk.reserve(static_cast<size_t>(3 * radius * (radius + 1) + 1));
    for (int dq = -radius; dq <= radius; dq++) {
        const int lo = std::max(-radius, -dq - radius), hi = std::min(radius, -dq + radius);
        for (int dr = lo; dr <= hi; dr++) disk.emplace_back(dq, dr);
    }

    if (passable && elevation && floors) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                const int i0 = y * size + x;
                const int e0 = elevation[i0];
                const int q0 = x - ((y - (y & 1)) >> 1);
                bool crush = false;
                uint8_t v = 0;
                for (const auto& [dq, dr] : disk) {
                    const int cy = y + dr;
                    const int cx = q0 + dq + ((cy - (cy & 1)) >> 1);
                    if (cx < 0 || cy < 0 || cx >= size || cy >= size) { v = 3; break; }
                    const int i = cy * size + cx;
                    if (!passable[i]) { v = 3; break; }
                    const int f = floors[i];
                    if (f >= 0) {
                        if (crushFloors < 0 || f > crushFloors) { v = 3; break; }
                        crush = true;
                    }
                    const int de = elevation[i] - e0;
                    if (de > 1 || de < -1) { v = 3; break; }
                }
                out[static_cast<size_t>(i0)] = v ? v : (crush ? 2 : 1);
            }
        }
    }

    std::vector<uint8_t>& slot = clearance_[id];
    slot.swap(out);
    for (auto it = components_.begin(); it != components_.end();) {
        const size_t sep = it->first.find(kKeySep);
        if (sep != std::string::npos && it->first.compare(sep + 1, std::string::npos, id) == 0)
            it = components_.erase(it);
        else
            ++it;
    }
    return slot;
}

bool HexNav::hasClearance(const std::string& id) const {
    return clearance_.find(id) != clearance_.end();
}

const std::vector<uint8_t>* HexNav::clearance(const std::string& id) const {
    auto it = clearance_.find(id);
    return it == clearance_.end() ? nullptr : &it->second;
}

// ── Frontier ───────────────────────────────────────────────────────────────
//
// Binary min-heap keyed on (k, h, seq). The sequence number is load-bearing:
// among equal (k, h) the earliest-pushed entry pops first, which is what
// makes the parent assignment — and so the path — a function of the inputs
// alone rather than of the heap's shape.

static inline bool entryLess(double ak, double ah, int32_t as, double bk, double bh, int32_t bs) {
    if (ak != bk) return ak < bk;
    if (ah != bh) return ah < bh;
    return as < bs;
}

void HexNav::heapPush(double k, double h, int32_t v) {
    Entry e{k, h, seq_++, v};
    heap_.push_back(e);
    size_t i = heap_.size() - 1;
    while (i > 0) {
        const size_t p = (i - 1) >> 1;
        const Entry pe = heap_[p];
        if (!entryLess(e.k, e.h, e.seq, pe.k, pe.h, pe.seq)) break;
        heap_[i] = pe;
        i = p;
    }
    heap_[i] = e;
}

HexNav::Entry HexNav::heapPop() {
    const Entry top = heap_[0];
    const Entry last = heap_.back();
    heap_.pop_back();
    const size_t n = heap_.size();
    if (n > 0) {
        size_t i = 0;
        for (;;) {
            const size_t l = 2 * i + 1;
            if (l >= n) break;
            const size_t r = l + 1;
            size_t m = l;
            if (r < n && entryLess(heap_[r].k, heap_[r].h, heap_[r].seq,
                                   heap_[l].k, heap_[l].h, heap_[l].seq)) m = r;
            const Entry me = heap_[m];
            if (!entryLess(me.k, me.h, me.seq, last.k, last.h, last.seq)) break;
            heap_[i] = me;
            i = m;
        }
        heap_[i] = last;
    }
    return top;
}

void HexNav::releaseScratch() {
    for (int32_t i : touched_) {
        cost_[static_cast<size_t>(i)] = kInfF;
        parent_[static_cast<size_t>(i)] = -1;
    }
    touched_.clear();
    heap_.clear();
}

double HexNav::heuristic(int x, int y, int gq, int gr) const {
    const int q = x - ((y - (y & 1)) >> 1);
    const int dq = gq - q, dr = gr - y;
    return static_cast<double>(std::abs(dq) + std::abs(dq + dr) + std::abs(dr)) / 2.0;
}

// ── The one search ─────────────────────────────────────────────────────────

bool HexNav::search(const std::vector<double>& table, const std::vector<uint8_t>* clr,
                    int x0, int y0, int goal, double maxCost) {
    const int size = size_;
    const bool astar = goal >= 0;
    int gq = 0, gr = 0;
    if (astar) {
        const int gx = goal % size, gy = goal / size;
        gq = gx - ((gy - (gy & 1)) >> 1);
        gr = gy;
    }
    const double* t = table.data();
    const uint8_t* c = clr ? clr->data() : nullptr;

    const int32_t start = y0 * size + x0;
    cost_[static_cast<size_t>(start)] = 0.0f;
    touched_.push_back(start);
    heap_.clear();
    seq_ = 0;
    const double h0 = astar ? heuristic(x0, y0, gq, gr) : 0.0;
    heapPush(h0, h0, start);

    while (!heap_.empty()) {
        const Entry e = heapPop();
        const int32_t i = e.v;
        const double g = e.k - e.h;                    // f − h, both exact; 0 for Dijkstra
        if (g > static_cast<double>(cost_[static_cast<size_t>(i)])) continue;  // stale entry
        if (i == goal) return true;
        const int cx = i % size, cy = i / size;
        const int (*step)[2] = STEP[cy & 1];
        for (int d = 0; d < 6; d++) {
            const int nx = cx + step[d][0], ny = cy + step[d][1];
            if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
            const int32_t ni = ny * size + nx;
            // Leaving c in direction d enters n from the opposite side: that slot.
            double sc = t[static_cast<size_t>(ni) * 6 + ((d + 3) % 6)];
            if (sc == kInf) continue;
            if (c) {
                const uint8_t v = c[static_cast<size_t>(ni)];
                if (v != 1 && v != 2) continue;
                sc = sc * static_cast<double>(v);
            }
            const double nc = g + sc;
            if (nc > maxCost) continue;
            float& slot = cost_[static_cast<size_t>(ni)];
            if (nc < static_cast<double>(slot)) {
                if (slot == kInfF) touched_.push_back(ni);
                slot = static_cast<float>(nc);
                parent_[static_cast<size_t>(ni)] = i;
                const double h = astar ? heuristic(nx, ny, gq, gr) : 0.0;
                heapPush(static_cast<double>(slot) + h, h, ni);
            }
        }
    }
    return false;
}

bool HexNav::findPath(const std::string& id, int x0, int y0, int x1, int y1, double maxCost,
                      std::vector<int32_t>& outPath) {
    auto it = tables_.find(id);
    if (it == tables_.end()) return false;
    if (!inBounds(x0, y0) || !inBounds(x1, y1)) return false;
    const int32_t start = y0 * size_ + x0, goal = y1 * size_ + x1;
    if (disconnected(id, std::string(), start, goal)) return false;
    const bool ok = search(it->second, nullptr, x0, y0, goal, maxCost);
    if (ok) {
        outPath.clear();
        for (int32_t i = goal; i != -1; i = parent_[static_cast<size_t>(i)]) outPath.push_back(i);
        std::reverse(outPath.begin(), outPath.end());
    }
    releaseScratch();
    return ok;
}

bool HexNav::findPathRadius(const std::string& id, const std::string& clearanceId,
                            int x0, int y0, int x1, int y1, double maxCost,
                            std::vector<int32_t>& outPath) {
    auto it = tables_.find(id);
    auto ct = clearance_.find(clearanceId);
    if (it == tables_.end() || ct == clearance_.end()) return false;
    if (!inBounds(x0, y0) || !inBounds(x1, y1)) return false;
    const int32_t start = y0 * size_ + x0, goal = y1 * size_ + x1;
    const uint8_t gv = ct->second[static_cast<size_t>(goal)];
    if (gv != 1 && gv != 2) return false;
    if (disconnected(id, clearanceId, start, goal)) return false;
    const bool ok = search(it->second, &ct->second, x0, y0, goal, maxCost);
    if (ok) {
        outPath.clear();
        for (int32_t i = goal; i != -1; i = parent_[static_cast<size_t>(i)]) outPath.push_back(i);
        std::reverse(outPath.begin(), outPath.end());
    }
    releaseScratch();
    return ok;
}

bool HexNav::movementField(const std::string& id, int x0, int y0, double maxCost,
                           std::vector<float>& cost, std::vector<int32_t>& parent) {
    auto it = tables_.find(id);
    if (it == tables_.end() || !inBounds(x0, y0)) return false;
    search(it->second, nullptr, x0, y0, -1, maxCost);
    cost.assign(cost_.begin(), cost_.end());
    parent.assign(parent_.begin(), parent_.end());
    releaseScratch();
    return true;
}

// ── Reversed field ─────────────────────────────────────────────────────────

namespace {

// The fallback frontier: a plain (dist, idx) heap with the same sift rules
// as the reference, so even the pop count of a build the ring could not
// order matches.
struct DistHeap {
    std::vector<double> hd;
    std::vector<int32_t> hi;
    size_t n = 0;
    void push(double d, int32_t i) {
        size_t c = n;
        hd[c] = d; hi[c] = i;
        while (c > 0) {
            const size_t p = (c - 1) >> 1;
            if (hd[p] <= hd[c]) break;
            std::swap(hd[p], hd[c]);
            std::swap(hi[p], hi[c]);
            c = p;
        }
        n++;
    }
    void pop(double& d, int32_t& i) {
        d = hd[0]; i = hi[0];
        n -= 1;
        hd[0] = hd[n]; hi[0] = hi[n];
        size_t c = 0;
        for (;;) {
            const size_t l = 2 * c + 1, r = l + 1;
            size_t m = c;
            if (l < n && hd[l] < hd[m]) m = l;
            if (r < n && hd[r] < hd[m]) m = r;
            if (m == c) break;
            std::swap(hd[m], hd[c]);
            std::swap(hi[m], hi[c]);
            c = m;
        }
    }
};

} // namespace

size_t HexNav::targetField(const std::string& id, const int32_t* seeds, size_t nSeeds,
                           const uint8_t* blocked, const uint8_t* aura, double auraMult,
                           double quantum, int ring,
                           std::vector<double>& dist, std::vector<int32_t>& parent) {
    const int size = size_;
    const size_t n = static_cast<size_t>(cells());
    dist.assign(n, kInf);
    parent.assign(n, -1);
    auto it = tables_.find(id);
    if (it == tables_.end()) return 0;
    const double* t = it->second.data();

    std::vector<int32_t> seedList;
    seedList.reserve(nSeeds);
    for (size_t k = 0; k < nSeeds; k++) {
        const int32_t s = seeds ? seeds[k] : -1;
        if (s < 0 || static_cast<size_t>(s) >= n) continue;
        if (dist[static_cast<size_t>(s)] == 0.0) continue;
        dist[static_cast<size_t>(s)] = 0.0;
        seedList.push_back(s);
    }

    size_t pops = 0;
    bool ok = ring > 0 && quantum > 0.0;

    if (ok) {
        // The bucket queue: a ring of `ring` FIFO buckets by distance in
        // quanta; an entry is a cell in a singly linked list per bucket.
        const size_t cap = n * 6 + seedList.size() + 8;
        std::vector<int32_t> ecell(cap), link(cap);
        std::vector<int32_t> head(static_cast<size_t>(ring), -1), tail(static_cast<size_t>(ring), -1);
        size_t count = 0, pending = 0;
        long long cur = 0;
        for (int32_t s : seedList) {
            const int32_t k = static_cast<int32_t>(count++);
            ecell[static_cast<size_t>(k)] = s; link[static_cast<size_t>(k)] = -1;
            if (tail[0] < 0) head[0] = k; else link[static_cast<size_t>(tail[0])] = k;
            tail[0] = k;
            pending++;
        }
        while (pending > 0) {
            const size_t bk = static_cast<size_t>(cur % ring);
            const int32_t k = head[bk];
            if (k < 0) { cur++; continue; }
            pops++;
            head[bk] = link[static_cast<size_t>(k)];
            if (head[bk] < 0) tail[bk] = -1;
            pending--;
            const int32_t idx = ecell[static_cast<size_t>(k)];
            const double d0 = static_cast<double>(cur) * quantum;
            if (d0 > dist[static_cast<size_t>(idx)]) continue;   // stale entry
            const int cx = idx % size, cy = idx / size;
            const double enterMult = (aura && aura[static_cast<size_t>(idx)]) ? auraMult : 1.0;
            const int (*step)[2] = STEP[cy & 1];
            const size_t base = static_cast<size_t>(idx) * 6;
            for (int d = 0; d < 6; d++) {
                const int nx = cx + step[d][0], ny = cy + step[d][1];
                if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                const int32_t nIdx = ny * size + nx;
                if (blocked && blocked[static_cast<size_t>(nIdx)]) continue;
                const double sc = t[base + static_cast<size_t>(d)];
                if (sc == kInf) continue;
                const double cand = d0 + sc * enterMult;
                if (cand < dist[static_cast<size_t>(nIdx)] - 1e-9) {
                    dist[static_cast<size_t>(nIdx)] = cand;
                    parent[static_cast<size_t>(nIdx)] = idx;
                    const double u = cand / quantum;
                    if (u != std::floor(u) || u - static_cast<double>(cur) >= ring || count >= cap) {
                        ok = false;
                        break;
                    }
                    const size_t nb = static_cast<size_t>(static_cast<long long>(u) % ring);
                    const int32_t nk = static_cast<int32_t>(count++);
                    ecell[static_cast<size_t>(nk)] = nIdx; link[static_cast<size_t>(nk)] = -1;
                    if (tail[nb] < 0) head[nb] = nk; else link[static_cast<size_t>(tail[nb])] = nk;
                    tail[nb] = nk;
                    pending++;
                }
            }
            if (!ok) break;
        }
        if (ok) return pops;
        std::fill(dist.begin(), dist.end(), kInf);
        std::fill(parent.begin(), parent.end(), -1);
        for (int32_t s : seedList) dist[static_cast<size_t>(s)] = 0.0;
    }

    // The heap: a cost the ring cannot order — same distances.
    DistHeap heap;
    heap.hd.resize(n + 8);
    heap.hi.resize(n + 8);
    for (int32_t s : seedList) heap.push(0.0, s);
    while (heap.n > 0) {
        double d0; int32_t idx;
        heap.pop(d0, idx);
        pops++;
        if (d0 > dist[static_cast<size_t>(idx)]) continue;
        const int cx = idx % size, cy = idx / size;
        const double enterMult = (aura && aura[static_cast<size_t>(idx)]) ? auraMult : 1.0;
        const int (*step)[2] = STEP[cy & 1];
        const size_t base = static_cast<size_t>(idx) * 6;
        for (int d = 0; d < 6; d++) {
            const int nx = cx + step[d][0], ny = cy + step[d][1];
            if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
            const int32_t nIdx = ny * size + nx;
            if (blocked && blocked[static_cast<size_t>(nIdx)]) continue;
            const double sc = t[base + static_cast<size_t>(d)];
            if (sc == kInf) continue;
            const double cand = d0 + sc * enterMult;
            if (cand < dist[static_cast<size_t>(nIdx)] - 1e-9) {
                dist[static_cast<size_t>(nIdx)] = cand;
                parent[static_cast<size_t>(nIdx)] = idx;
                if (heap.n < heap.hd.size()) heap.push(cand, nIdx);
                else {
                    heap.hd.resize(heap.hd.size() * 2);
                    heap.hi.resize(heap.hi.size() * 2);
                    heap.push(cand, nIdx);
                }
            }
        }
    }
    return pops;
}

// ── Components ─────────────────────────────────────────────────────────────

void HexNav::invalidateComponents(const std::string& tableId) {
    const std::string prefix = tableId + kKeySep;
    for (auto it = components_.begin(); it != components_.end();) {
        if (it->first.compare(0, prefix.size(), prefix) == 0) it = components_.erase(it);
        else ++it;
    }
}

const std::vector<int32_t>& HexNav::components(const std::string& id,
                                               const std::string& clearanceId) {
    const std::string key = id + kKeySep + clearanceId;
    auto cached = components_.find(key);
    if (cached != components_.end()) return cached->second;

    const int size = size_;
    const size_t n = static_cast<size_t>(cells());
    std::vector<int32_t> labels(n, -1);
    auto it = tables_.find(id);
    if (it != tables_.end()) {
        const double* t = it->second.data();
        const uint8_t* c = nullptr;
        if (!clearanceId.empty()) {
            auto ct = clearance_.find(clearanceId);
            if (ct != clearance_.end()) c = ct->second.data();
        }
        auto standable = [c](int32_t i) {
            if (!c) return true;
            const uint8_t v = c[static_cast<size_t>(i)];
            return v == 1 || v == 2;
        };
        std::vector<int32_t> stack;
        stack.reserve(n);
        int32_t label = 0;
        for (size_t s = 0; s < n; s++) {
            if (labels[s] != -1) continue;
            labels[s] = label;
            stack.push_back(static_cast<int32_t>(s));
            while (!stack.empty()) {
                const int32_t i = stack.back();
                stack.pop_back();
                const int cx = i % size, cy = i / size;
                const int (*step)[2] = STEP[cy & 1];
                for (int d = 0; d < 6; d++) {
                    const int nx = cx + step[d][0], ny = cy + step[d][1];
                    if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
                    const int32_t ni = ny * size + nx;
                    if (labels[static_cast<size_t>(ni)] != -1) continue;
                    // i → n enters n from OPP(d); n → i enters i from d.
                    const bool out = t[static_cast<size_t>(ni) * 6 + ((d + 3) % 6)] != kInf && standable(ni);
                    const bool in = t[static_cast<size_t>(i) * 6 + d] != kInf && standable(i);
                    if (!out && !in) continue;
                    labels[static_cast<size_t>(ni)] = label;
                    stack.push_back(ni);
                }
            }
            label++;
        }
    }
    return components_.emplace(key, std::move(labels)).first->second;
}

bool HexNav::disconnected(const std::string& id, const std::string& clearanceId, int a, int b) {
    const std::vector<int32_t>& comp = components(id, clearanceId);
    return comp[static_cast<size_t>(a)] != comp[static_cast<size_t>(b)];
}

} // namespace brogameagent
