#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace brogameagent::grid {

// ─── ObsWindow ─────────────────────────────────────────────────────────────
//
// Egocentric multi-channel rasterizer for 2D grid / side-scrolling envs.
//
// The window is a fixed (cols × rows) rectangle around an ego cell, with
// `cols_behind` columns to the ego's left and `cols_ahead` to its right,
// and `rows_up` rows above + `rows_down` rows below. The ego cell sits at
// (cols_behind, rows_up) inside the window.
//
// The output is a flat Float32 vector laid out as:
//
//     [ tile block         ]   cols * rows * tile_channels
//     [ entity layer 0     ]   cols * rows * layers[0].channels
//     [ entity layer 1     ]   cols * rows * layers[1].channels
//     [ ... more layers ...]
//     [ self block         ]   self_block_size
//
// Within each block, indexing is row-major over the window grid:
//
//     idx = ((row * cols) + col) * channels + channel
//
// where (col, row) are window-local coordinates: col=0 is the leftmost
// (cols_behind cells behind ego), col=cols-1 is the rightmost. row=0 is
// the topmost (rows_up cells above ego), row=rows-1 is the bottommost.
//
// Tile sampling: tile_fn is called once per cell in the window with the
// world-space (col, row) and writes `tile_channels` floats. Cells that
// fall outside the world are filled with `oob_tile` (defaults to all
// zeros). No top-K cap.
//
// Entity rasterization: for each layer, the caller hands back the live
// entity list. Each entity is sampled to (col, row, channel_values[]).
// Entities outside the window are skipped. Multiple entities mapping to
// the same cell accumulate additively (configurable via overwrite mode
// per layer). After rasterization, each channel is multiplied by its
// configured normalizer.
//
// Self block: appended verbatim from the caller-supplied span. Size is
// fixed at construction time. The caller is responsible for any
// normalization on these values.
//
// Snapshots / replays: the rasterizer is stateless. Call build() with
// any (ego_col, ego_row, self_block) tuple in any order.

struct EntityCell {
    int col = 0;
    int row = 0;
    // Per-channel values; size must equal layer.channels at build time.
    // Out-of-window entities are skipped before this is read, so the
    // sampler can return arbitrary col/row without bounds-checking.
    std::vector<float> values;
};

struct EntityLayerSpec {
    int  channels = 1;
    bool overwrite = false;   // false = additive accumulate; true = last write wins
    // Normalizer per channel. Multiplied into the rasterized cell value
    // after accumulation. Empty means all 1.0.
    std::vector<float> normalize;
    // Hot path: each build() call invokes enumerate_fn once, then
    // sample_fn per returned entity. The kit treats opaque entity ids
    // as size_t — the caller picks the encoding (vector index, pointer
    // hash, EntityID). Returning a count of 0 is fine.
    std::function<size_t()>                    enumerate_fn;
    std::function<EntityCell(size_t entity_idx)> sample_fn;
};

struct ObsWindowSpec {
    int cols_behind = 0;
    int cols_ahead  = 0;
    int rows_up     = 0;
    int rows_down   = 0;

    // Tile channels per cell.
    int tile_channels = 1;

    // Per-channel multiplier applied to tile values after sampling.
    // Empty == all 1.0.
    std::vector<float> tile_normalize;

    // Out-of-bounds tile fill. Empty == zeros (size = tile_channels).
    std::vector<float> oob_tile;

    int self_block_size = 0;
};

// Tile sampler: (col, row) -> tile_channels floats written to `out`.
// Return false if the cell is out-of-world; build() will substitute
// oob_tile in that case. Return true on a valid sample.
using TileSampleFn = std::function<bool(int col, int row, float* out)>;

// Documented layout of one build() output. Returned by ObsWindow::layout()
// for callers who need to read back specific blocks (test invariants,
// UI overlays, debug dumps).
struct ObsWindowLayout {
    int cols = 0;
    int rows = 0;

    int tile_offset   = 0;    // float index of the tile block
    int tile_channels = 0;
    int tile_size     = 0;    // cols * rows * tile_channels

    struct LayerInfo {
        int offset   = 0;
        int channels = 0;
        int size     = 0;     // cols * rows * channels
    };
    std::vector<LayerInfo> layers;

    int self_offset = 0;
    int self_size   = 0;

    int total = 0;            // = sum of all blocks; matches out_dim()
};

class ObsWindow {
public:
    ObsWindow(ObsWindowSpec spec,
              TileSampleFn tile_fn,
              std::vector<EntityLayerSpec> layers);

    int out_dim() const { return layout_.total; }
    const ObsWindowLayout& layout() const { return layout_; }

    // Build one observation into `out` (size must be >= out_dim()).
    // self_block is copied verbatim into the self block; if its size is
    // less than self_block_size, the remainder is zero-filled.
    void build(int ego_col,
               int ego_row,
               const float* self_block, size_t self_len,
               float* out) const;

    // Convenience overload allocating a fresh vector.
    std::vector<float> build(int ego_col, int ego_row,
                             const std::vector<float>& self_block) const;

private:
    ObsWindowSpec                 spec_;
    TileSampleFn                  tile_fn_;
    std::vector<EntityLayerSpec>  layers_;
    ObsWindowLayout               layout_{};
};

} // namespace brogameagent::grid
