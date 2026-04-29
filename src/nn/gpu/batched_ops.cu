// Batched (inference-only) GPU kernels for the BatchedInferenceServer.
//
// Each op runs B independent forward passes in a single kernel launch. We do
// not modify the existing single-sample kernels — these are additive.
//
// Tensor layout: a (B, D) row-major tensor, so sample b occupies the
// half-open row range [b*D, (b+1)*D). Linear shares one weight matrix W of
// shape (out_dim, in_dim) across all B rows.

#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

// ─── linear_forward_batched ────────────────────────────────────────────────
//
// One block per (sample, output-row tile). Within a block, threadIdx.x walks
// the output rows of the tile and threadIdx.y indexes the sample. Because B
// can be small (1) and out_dim moderate, we keep the parallelism in the
// out_dim axis and one block per sample.
//
// We tile the in_dim axis through shared memory, sized for the whole block.

constexpr int BL_ROWS_PER_BLOCK = 64;   // output rows per block
constexpr int BL_TILE           = 64;   // in_dim tile width

__global__ void linear_forward_batched_kernel(const float* __restrict__ W,
                                              const float* __restrict__ bias,
                                              const float* __restrict__ X,
                                              float* __restrict__ Y,
                                              int B, int out_dim, int in_dim) {
    __shared__ float xtile[BL_TILE];

    const int b   = blockIdx.y;
    const int row = blockIdx.x * BL_ROWS_PER_BLOCK + threadIdx.x;
    if (b >= B) return;

    const float* x_row = X + static_cast<size_t>(b) * in_dim;
    float acc = 0.0f;

    for (int t0 = 0; t0 < in_dim; t0 += BL_TILE) {
        const int t_len = (in_dim - t0) < BL_TILE ? (in_dim - t0) : BL_TILE;

        // Cooperatively load a tile of x_row into shared memory.
        for (int k = threadIdx.x; k < t_len; k += blockDim.x) {
            xtile[k] = x_row[t0 + k];
        }
        __syncthreads();

        if (row < out_dim) {
            const float* wrow = W + static_cast<size_t>(row) * in_dim + t0;
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += wrow[k] * xtile[k];
            }
        }
        __syncthreads();
    }

    if (row < out_dim) {
        Y[static_cast<size_t>(b) * out_dim + row] = bias[row] + acc;
    }
}

// ─── elementwise batched ───────────────────────────────────────────────────
//
// (B, D) is just a flat buffer of size B*D as far as elementwise ops are
// concerned. We reuse the same grid-stride pattern as the single-sample
// elementwise kernels.

constexpr int EW_BLOCK = 256;

__global__ void relu_forward_batched_kernel(const float* __restrict__ x,
                                            float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void tanh_forward_batched_kernel(const float* __restrict__ x,
                                            float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = tanhf(x[i]);
    }
}

__global__ void add_inplace_batched_kernel(float* __restrict__ y,
                                           const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += x[i];
    }
}

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

} // anonymous namespace

void linear_forward_batched_gpu(const GpuTensor& W, const GpuTensor& bias,
                                const GpuTensor& X_BD, GpuTensor& Y_BD) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim) Y_BD.resize(B, out_dim);
    if (B == 0 || out_dim == 0) return;

    dim3 block(BL_ROWS_PER_BLOCK, 1);
    dim3 grid((out_dim + BL_ROWS_PER_BLOCK - 1) / BL_ROWS_PER_BLOCK, B);
    linear_forward_batched_kernel<<<grid, block>>>(
        W.data, bias.data, X_BD.data, Y_BD.data, B, out_dim, in_dim);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void relu_forward_batched_gpu(const GpuTensor& X_BD, GpuTensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols)
        Y_BD.resize(X_BD.rows, X_BD.cols);
    const int n = X_BD.size();
    if (n == 0) return;
    relu_forward_batched_kernel<<<grid_for(n), EW_BLOCK>>>(X_BD.data, Y_BD.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward_batched_gpu(const GpuTensor& X_BD, GpuTensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols)
        Y_BD.resize(X_BD.rows, X_BD.cols);
    const int n = X_BD.size();
    if (n == 0) return;
    tanh_forward_batched_kernel<<<grid_for(n), EW_BLOCK>>>(X_BD.data, Y_BD.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void add_inplace_batched_gpu(GpuTensor& Y_BD, const GpuTensor& X_BD) {
    const int n = Y_BD.size();
    if (n == 0) return;
    add_inplace_batched_kernel<<<grid_for(n), EW_BLOCK>>>(Y_BD.data, X_BD.data, n);
    BGA_CUDA_CHECK(cudaGetLastError());
}

} // namespace brogameagent::nn::gpu
