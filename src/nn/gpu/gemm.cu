#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

namespace {

// y[i] = b[i] + sum_j W[i, j] * x[j]
// One thread per output row. Uses shared memory to cache tiles of x.
constexpr int LF_BLOCK = 128;
constexpr int LF_TILE  = 128;

__global__ void linear_forward_kernel(const float* __restrict__ W,
                                      const float* __restrict__ b,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int out_dim, int in_dim) {
    __shared__ float xtile[LF_TILE];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < in_dim; t0 += LF_TILE) {
        const int t_len = (in_dim - t0) < LF_TILE ? (in_dim - t0) : LF_TILE;

        // Cooperatively load tile of x into shared memory.
        for (int k = tid; k < t_len; k += blockDim.x) {
            xtile[k] = x[t0 + k];
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
        y[row] = b[row] + acc;
    }
}

// dX[j] = sum_i W[i, j] * dY[i]
// One thread per input column. Uses shared memory to cache tiles of dY.
constexpr int LB_DX_BLOCK = 128;
constexpr int LB_DX_TILE  = 128;

__global__ void linear_backward_dx_kernel(const float* __restrict__ W,
                                          const float* __restrict__ dY,
                                          float* __restrict__ dX,
                                          int out_dim, int in_dim) {
    __shared__ float dytile[LB_DX_TILE];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < out_dim; t0 += LB_DX_TILE) {
        const int t_len = (out_dim - t0) < LB_DX_TILE ? (out_dim - t0) : LB_DX_TILE;

        for (int k = tid; k < t_len; k += blockDim.x) {
            dytile[k] = dY[t0 + k];
        }
        __syncthreads();

        if (col < in_dim) {
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += W[static_cast<size_t>(t0 + k) * in_dim + col] * dytile[k];
            }
        }
        __syncthreads();
    }

    if (col < in_dim) {
        dX[col] = acc;
    }
}

// dW[i, j] += dY[i] * x[j]. 2D grid: each thread one (i, j).
__global__ void linear_backward_dw_kernel(const float* __restrict__ dY,
                                          const float* __restrict__ x,
                                          float* __restrict__ dW,
                                          int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    dW[static_cast<size_t>(i) * in_dim + j] += dY[i] * x[j];
}

// dB[i] += dY[i].
__global__ void linear_backward_db_kernel(const float* __restrict__ dY,
                                          float* __restrict__ dB,
                                          int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    dB[i] += dY[i];
}

} // anonymous namespace

void linear_forward_gpu(const GpuTensor& W, const GpuTensor& b,
                        const GpuTensor& x, GpuTensor& y) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (y.rows != out_dim || y.cols != 1) y.resize(out_dim, 1);
    if (out_dim == 0) return;

    const int blocks = (out_dim + LF_BLOCK - 1) / LF_BLOCK;
    linear_forward_kernel<<<blocks, LF_BLOCK>>>(W.data, b.data, x.data, y.data,
                                                out_dim, in_dim);
    BGA_CUDA_CHECK(cudaGetLastError());
}

void linear_backward_gpu(const GpuTensor& W, const GpuTensor& x,
                         const GpuTensor& dY,
                         GpuTensor& dX, GpuTensor& dW, GpuTensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;

    if (dX.rows != in_dim || dX.cols != 1) dX.resize(in_dim, 1);

    // dX = W^T * dY (overwrite)
    if (in_dim > 0) {
        const int blocks = (in_dim + LB_DX_BLOCK - 1) / LB_DX_BLOCK;
        linear_backward_dx_kernel<<<blocks, LB_DX_BLOCK>>>(
            W.data, dY.data, dX.data, out_dim, in_dim);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dW += dY * x^T
    if (out_dim > 0 && in_dim > 0) {
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        linear_backward_dw_kernel<<<grid, block>>>(
            dY.data, x.data, dW.data, out_dim, in_dim);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dB += dY
    if (out_dim > 0) {
        const int blocks = (out_dim + 255) / 256;
        linear_backward_db_kernel<<<blocks, 256>>>(dY.data, dB.data, out_dim);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace brogameagent::nn::gpu
