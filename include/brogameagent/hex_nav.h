#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace brogameagent {

/// Weighted A* / Dijkstra over a pointy-top, odd-r offset hex grid.
///
/// The grid is `size × size` cells stored row-major (`idx = y * size + x`).
/// Directions are 0=E 1=NE 2=NW 3=W 4=SW 5=SE; odd rows are shoved right.
/// The navigator owns no rule of its own: every cost comes from a **step
/// table** the embedder authors, `size*size*6` doubles where slot
/// `c * 6 + d` is the cost of ENTERING cell `c` from its neighbour in
/// direction `d` (infinity = impassable). A radius-r unit adds a **clearance
/// table**, one byte per cell: 1 clear, 2 crushing (the step is doubled),
/// anything else cannot be stood on.
///
/// The searches reproduce a reference implementation's expansion order
/// exactly, so that an embedder migrating from its own A* gets the same path
/// for every query, not merely an equal-cost one:
///   - frontier ordered by (f, h, insertion sequence) — ties in f break by
///     the heuristic, then by the order the entry was pushed;
///   - `g` is stored in single precision (the reference kept its cost field
///     in a Float32Array) while keys and sums are double;
///   - a `maxCost` cutoff refuses any relaxation whose cost exceeds it;
///   - the goal is answered the moment it is *settled* (popped);
///   - the path is the parent chain from the goal, start first.
/// Deterministic: no threads, no allocation order dependence, no float
/// summation reordering.
class HexNav {
public:
    explicit HexNav(int size);

    int size() const { return size_; }
    int cells() const { return size_ * size_; }

    // ── Step-cost tables ─────────────────────────────────────────────────
    /// Install (copy) a table under `id`; `n` must be `cells()*6`.
    /// Returns false (and installs nothing) on a size mismatch.
    bool setStepCosts(const std::string& id, const double* costs, size_t n);
    /// Rewrite the six entry slots of `nCells` cells: `values` holds
    /// `nCells*6` doubles in the same order. Unknown table → false.
    bool updateStepCosts(const std::string& id, const int32_t* cellIdx, size_t nCells,
                         const double* values);
    bool hasStepCosts(const std::string& id) const;
    const std::vector<double>* stepCosts(const std::string& id) const;

    // ── Clearance tables ─────────────────────────────────────────────────
    /// Install (copy) a clearance table under `id`; `n` must be `cells()`.
    bool setClearance(const std::string& id, const uint8_t* table, size_t n);
    /// Compute a clearance table natively and install it under `id`:
    /// a radius-`radius` footprint can stand centred on a cell when every
    /// footprint cell is in bounds, `passable[cell] != 0`, within ±1 of the
    /// centre's `elevation`, and either has no structure (`floors[cell] < 0`)
    /// or is crushable — `crushFloors >= 0` and `floors[cell] <= crushFloors`
    /// — in which case the answer is 2 (crushing) rather than 1. Any failure
    /// writes 3. Returns the installed table.
    const std::vector<uint8_t>& buildClearance(const std::string& id, int radius,
                                               const uint8_t* passable, const int8_t* elevation,
                                               const int16_t* floors, int crushFloors);
    bool hasClearance(const std::string& id) const;
    const std::vector<uint8_t>* clearance(const std::string& id) const;

    // ── Searches ─────────────────────────────────────────────────────────
    /// A* from (x0,y0) to (x1,y1) over table `id`. On success `outPath`
    /// holds the cell indices start..goal inclusive and true is returned;
    /// otherwise false (no path within `maxCost`, out of bounds, unknown
    /// table). A goal weakly disconnected from the start is answered without
    /// a search (see components()).
    bool findPath(const std::string& id, int x0, int y0, int x1, int y1, double maxCost,
                  std::vector<int32_t>& outPath);

    /// The clearance search: every destination's step is `table × clearance`
    /// (1 or 2), a destination that cannot be stood on is impassable. Same
    /// ordering as findPath. Returns false when the goal cannot be stood on.
    bool findPathRadius(const std::string& id, const std::string& clearanceId,
                        int x0, int y0, int x1, int y1, double maxCost,
                        std::vector<int32_t>& outPath);

    /// Dijkstra from (x0,y0) out to `maxCost`: `cost[cell]` (single precision,
    /// infinity = unreached) and `parent[cell]` (−1 = none) over the whole
    /// grid. Returns false on an unknown table or out-of-bounds start.
    bool movementField(const std::string& id, int x0, int y0, double maxCost,
                       std::vector<float>& cost, std::vector<int32_t>& parent);

    /// Reversed Dijkstra to a set of target cells: `dist[cell]` = cost of
    /// walking from that cell to the nearest seed, walking n → c paying c's
    /// entry cost from table `id`, multiplied by `auraMult` when
    /// `aura[c] != 0`. `blocked` cells are never relaxed (walls). Frontier is
    /// a bucket queue (Dial's) in units of `quantum` with a ring of `ring`
    /// buckets and FIFO order within a bucket; any edge the ring cannot order
    /// (not a whole number of quanta, or longer than the ring) restarts the
    /// build on a plain heap — same distances. `parent[cell]` is the
    /// neighbour the cell's distance came through (−1 for seeds / unreached).
    /// Returns the number of frontier pops, stale entries included — an
    /// embedder that used to slice the build under a per-tick pop budget can
    /// land the finished field on the same tick it would have.
    size_t targetField(const std::string& id, const int32_t* seeds, size_t nSeeds,
                       const uint8_t* blocked, const uint8_t* aura, double auraMult,
                       double quantum, int ring,
                       std::vector<double>& dist, std::vector<int32_t>& parent);

    /// Weakly connected components of the table's finite edges (with a
    /// clearance table: of the edges whose destination can be stood on), one
    /// label per cell, labels numbered in first-visit order. Two cells with
    /// different labels have no path between them in either direction, so a
    /// search can answer "unreachable" in O(1). Cached per (table,
    /// clearance) and dropped when the table changes.
    const std::vector<int32_t>& components(const std::string& id,
                                           const std::string& clearanceId = std::string());

    /// The step vector of direction `d` on a row of parity `parity` (0 even,
    /// 1 odd): {dx, dy}.
    static const int STEP[2][6][2];

private:
    struct Entry { double k; double h; int32_t seq; int32_t v; };

    bool inBounds(int x, int y) const { return x >= 0 && y >= 0 && x < size_ && y < size_; }
    double heuristic(int x, int y, int gq, int gr) const;
    // The one search; leaves its answer in the scratch (cost_/parent_).
    bool search(const std::vector<double>& table, const std::vector<uint8_t>* clr,
                int x0, int y0, int goal, double maxCost);
    void releaseScratch();
    void heapPush(double k, double h, int32_t v);
    Entry heapPop();
    void invalidateComponents(const std::string& tableId);
    bool disconnected(const std::string& id, const std::string& clearanceId, int a, int b);

    int size_;
    std::unordered_map<std::string, std::vector<double>> tables_;
    std::unordered_map<std::string, std::vector<uint8_t>> clearance_;
    std::unordered_map<std::string, std::vector<int32_t>> components_;

    // Search scratch, kept clean between calls (Infinity / −1) by releaseScratch().
    std::vector<float> cost_;
    std::vector<int32_t> parent_;
    std::vector<int32_t> touched_;
    std::vector<Entry> heap_;
    int32_t seq_ = 0;
};

} // namespace brogameagent
