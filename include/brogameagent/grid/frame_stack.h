#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace brogameagent::grid {

// ─── FrameStack ───────────────────────────────────────────────────────────
//
// Bounded ring of the last k observation frames concatenated into a single
// flat float buffer. Stateless w.r.t. how observations are produced: the
// caller pushes raw float arrays and reads back the concatenation in
// chronological order [oldest, ..., newest] (Atari-DQN convention).
//
// reset() zero-fills the ring and is the right call at episode boundaries
// — without it, the first decision of a new episode sees observations
// from the previous episode in earlier slots.
//
// Layout of read():
//
//     [ frame_{t-k+1} | frame_{t-k+2} | ... | frame_t ]
//
// Each frame is exactly inner_dim floats. read() always returns a buffer
// of size inner_dim * k. Before k pushes have happened, missing slots
// remain zeroed.

class FrameStack {
public:
    FrameStack(int inner_dim, int k)
        : inner_dim_(inner_dim), k_(k),
          ring_(static_cast<size_t>(inner_dim) * static_cast<size_t>(k), 0.0f) {}

    int inner_dim() const { return inner_dim_; }
    int k()         const { return k_; }
    int out_dim()   const { return inner_dim_ * k_; }

    // Drop all history. Use at episode boundaries.
    void reset() {
        std::fill(ring_.begin(), ring_.end(), 0.0f);
        write_idx_ = 0;
        filled_    = 0;
    }

    // Append `frame` (size inner_dim) into the ring. The oldest frame
    // is overwritten once the ring is full.
    void push(const float* frame) {
        size_t off = static_cast<size_t>(write_idx_) * static_cast<size_t>(inner_dim_);
        std::memcpy(ring_.data() + off, frame, sizeof(float) * static_cast<size_t>(inner_dim_));
        write_idx_ = (write_idx_ + 1) % k_;
        if (filled_ < k_) ++filled_;
    }

    // Write the chronological-order concatenation to `out` (size out_dim()).
    // Slots that haven't been pushed yet are written as zeros (consistent
    // with reset() semantics — the leading frames at episode start are
    // padding rather than stale data).
    void read(float* out) const {
        // The next slot to overwrite is the *oldest* frame's slot; emit
        // starting there, wrap around. When not yet full, the oldest valid
        // frame is at index 0 and we should pad the leading slots with
        // zeros so the freshest frame still lands at the end.
        if (filled_ < k_) {
            size_t pad_frames = static_cast<size_t>(k_ - filled_);
            std::memset(out, 0,
                        sizeof(float) * pad_frames * static_cast<size_t>(inner_dim_));
            // Frames written so far live at indices [0, filled_).
            std::memcpy(out + pad_frames * static_cast<size_t>(inner_dim_),
                        ring_.data(),
                        sizeof(float) * static_cast<size_t>(filled_) *
                            static_cast<size_t>(inner_dim_));
            return;
        }
        // Full ring: oldest is at write_idx_, freshest is at write_idx_-1.
        int oldest = write_idx_;
        for (int i = 0; i < k_; ++i) {
            int slot = (oldest + i) % k_;
            std::memcpy(out + static_cast<size_t>(i) * static_cast<size_t>(inner_dim_),
                        ring_.data() + static_cast<size_t>(slot) *
                            static_cast<size_t>(inner_dim_),
                        sizeof(float) * static_cast<size_t>(inner_dim_));
        }
    }

    std::vector<float> read() const {
        std::vector<float> out(static_cast<size_t>(out_dim()), 0.0f);
        read(out.data());
        return out;
    }

    // Number of frames pushed since last reset (clamped at k).
    int filled() const { return filled_; }

private:
    int inner_dim_;
    int k_;
    int write_idx_ = 0;
    int filled_    = 0;
    std::vector<float> ring_;
};

} // namespace brogameagent::grid
