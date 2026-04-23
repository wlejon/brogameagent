#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace brogameagent::nn {

// ─── Tensor ────────────────────────────────────────────────────────────────
//
// Plain owned float buffer + shape. No broadcasting, no views, no strides
// beyond row-major. Rank is fixed at 2 (matrix) or 1 (vector) in this
// library — anything higher would imply we want a real tensor framework,
// which we don't. Batch dims go into dim 0 of a 2D tensor.
//
// Shape conventions:
//   Vector: shape = (N)        ; data[i]
//   Matrix: shape = (rows,cols); data[r*cols + c]
//
// Memory is std::vector<float> — the default allocator over-aligns enough
// for SSE loads on every platform we care about; AVX2 paths in ops.cpp are
// unaligned loads for simplicity. Measurable alignment work can come later.

struct Tensor {
    std::vector<float> data;
    int rows = 0;   // rank-1 tensors: rows = N, cols = 1
    int cols = 0;

    Tensor() = default;
    Tensor(int r, int c) : data(static_cast<size_t>(r) * c, 0.0f), rows(r), cols(c) {}

    static Tensor vec(int n) { return Tensor(n, 1); }
    static Tensor mat(int r, int c) { return Tensor(r, c); }

    int size() const { return rows * cols; }
    float*       ptr()       { return data.data(); }
    const float* ptr() const { return data.data(); }

    float&       operator()(int r, int c)       { return data[static_cast<size_t>(r) * cols + c]; }
    float        operator()(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    float&       operator[](int i)       { return data[i]; }
    float        operator[](int i) const { return data[i]; }

    void zero() { std::memset(data.data(), 0, data.size() * sizeof(float)); }
    void resize(int r, int c) { rows = r; cols = c; data.assign(static_cast<size_t>(r) * c, 0.0f); }
};

} // namespace brogameagent::nn
