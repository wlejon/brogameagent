#include "brogameagent/grid/obs_window.h"

#include <algorithm>
#include <cstring>

namespace brogameagent::grid {

namespace {

void compute_layout(const ObsWindowSpec& s,
                    const std::vector<EntityLayerSpec>& layers,
                    ObsWindowLayout& out) {
    out.cols = s.cols_behind + s.cols_ahead + 1;
    out.rows = s.rows_up + s.rows_down + 1;
    int cells = out.cols * out.rows;

    out.tile_channels = s.tile_channels;
    out.tile_offset   = 0;
    out.tile_size     = cells * s.tile_channels;

    int cursor = out.tile_size;
    out.layers.clear();
    out.layers.reserve(layers.size());
    for (const auto& L : layers) {
        ObsWindowLayout::LayerInfo info;
        info.offset   = cursor;
        info.channels = L.channels;
        info.size     = cells * L.channels;
        cursor += info.size;
        out.layers.push_back(info);
    }

    out.self_offset = cursor;
    out.self_size   = s.self_block_size;
    cursor += out.self_size;

    out.total = cursor;
}

} // namespace

ObsWindow::ObsWindow(ObsWindowSpec spec,
                     TileSampleFn tile_fn,
                     std::vector<EntityLayerSpec> layers)
    : spec_(std::move(spec))
    , tile_fn_(std::move(tile_fn))
    , layers_(std::move(layers))
{
    compute_layout(spec_, layers_, layout_);
}

void ObsWindow::build(int ego_col, int ego_row,
                      const float* self_block, size_t self_len,
                      float* out) const {
    const int cols = layout_.cols;
    const int rows = layout_.rows;
    const int TC   = spec_.tile_channels;

    // Zero the whole buffer so we can rely on accumulate-from-zero for
    // entity layers and zero-fill for out-of-range self values.
    std::memset(out, 0, sizeof(float) * static_cast<size_t>(layout_.total));

    // ─── tile block ──────────────────────────────────────────────────
    float* tile_base = out + layout_.tile_offset;
    bool have_oob = !spec_.oob_tile.empty();
    for (int wr = 0; wr < rows; ++wr) {
        int world_row = ego_row + (wr - spec_.rows_up);
        for (int wc = 0; wc < cols; ++wc) {
            int world_col = ego_col + (wc - spec_.cols_behind);
            float* cell = tile_base + (static_cast<ptrdiff_t>(wr) * cols + wc) * TC;
            bool ok = false;
            if (tile_fn_) ok = tile_fn_(world_col, world_row, cell);
            if (!ok) {
                if (have_oob) {
                    int n = std::min<int>(TC, static_cast<int>(spec_.oob_tile.size()));
                    for (int i = 0; i < n; ++i) cell[i] = spec_.oob_tile[i];
                } // else memset zeros stand
            }
            // Apply tile normalizer.
            if (!spec_.tile_normalize.empty()) {
                int n = std::min<int>(TC, static_cast<int>(spec_.tile_normalize.size()));
                for (int i = 0; i < n; ++i) cell[i] *= spec_.tile_normalize[i];
            }
        }
    }

    // ─── entity layers ───────────────────────────────────────────────
    const int min_world_col = ego_col - spec_.cols_behind;
    const int max_world_col = ego_col + spec_.cols_ahead;
    const int min_world_row = ego_row - spec_.rows_up;
    const int max_world_row = ego_row + spec_.rows_down;

    for (size_t li = 0; li < layers_.size(); ++li) {
        const auto& L = layers_[li];
        const auto& info = layout_.layers[li];
        float* base = out + info.offset;
        size_t n = L.enumerate_fn ? L.enumerate_fn() : 0;
        for (size_t e = 0; e < n; ++e) {
            EntityCell ec = L.sample_fn(e);
            if (ec.col < min_world_col || ec.col > max_world_col) continue;
            if (ec.row < min_world_row || ec.row > max_world_row) continue;
            int wc = ec.col - min_world_col;
            int wr = ec.row - min_world_row;
            float* cell = base + (static_cast<ptrdiff_t>(wr) * cols + wc) * info.channels;
            int vN = std::min<int>(info.channels, static_cast<int>(ec.values.size()));
            if (L.overwrite) {
                for (int i = 0; i < vN; ++i) cell[i] = ec.values[i];
            } else {
                for (int i = 0; i < vN; ++i) cell[i] += ec.values[i];
            }
        }
        // Normalize after accumulation so additive layers stay linear.
        if (!L.normalize.empty()) {
            int normN = std::min<int>(info.channels, static_cast<int>(L.normalize.size()));
            int cells = cols * rows;
            for (int c = 0; c < cells; ++c) {
                float* cell = base + static_cast<ptrdiff_t>(c) * info.channels;
                for (int i = 0; i < normN; ++i) cell[i] *= L.normalize[i];
            }
        }
    }

    // ─── self block ──────────────────────────────────────────────────
    if (layout_.self_size > 0 && self_block && self_len > 0) {
        int n = std::min<int>(layout_.self_size, static_cast<int>(self_len));
        std::memcpy(out + layout_.self_offset, self_block, sizeof(float) * static_cast<size_t>(n));
    }
}

std::vector<float> ObsWindow::build(int ego_col, int ego_row,
                                    const std::vector<float>& self_block) const {
    std::vector<float> out(static_cast<size_t>(layout_.total), 0.0f);
    build(ego_col, ego_row,
          self_block.empty() ? nullptr : self_block.data(),
          self_block.size(),
          out.data());
    return out;
}

} // namespace brogameagent::grid
