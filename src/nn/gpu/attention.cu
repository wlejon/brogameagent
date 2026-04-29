#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

// Naive correctness-first kernels mirroring src/nn/attention.cpp.
//
// Shapes:
//   X       : (N, D)
//   Wq/Wk/Wv/Wo : (D, D)        rows = output dim, cols = input dim
//   Q/K/V   : (N, D)
//   Attn    : (N, N)
//   Y_pre_Wo: (N, D)            == Attn @ V
//   O       : (N, D)
//   d_mask  : optional length N (1 valid, 0 invalid). Invalid keys excluded
//             from softmax denom; invalid query rows produce zero output.

namespace {

// Y(i, j) = sum_k X(i, k) * W(j, k)   — i.e. Y = X @ W^T.
// One thread per (i, j).
__global__ void matmul_xwT_kernel(const float* __restrict__ X,
                                  const float* __restrict__ W,
                                  float* __restrict__ Y,
                                  int N, int D_in, int D_out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= D_out) return;
    const float* xr = X + static_cast<size_t>(i) * D_in;
    const float* wr = W + static_cast<size_t>(j) * D_in;
    float acc = 0.0f;
    for (int k = 0; k < D_in; ++k) acc += xr[k] * wr[k];
    Y[static_cast<size_t>(i) * D_out + j] = acc;
}

// scores(i, j) = (Q_i . K_j) / sqrt(D)
// One thread per (i, j) of the (N, N) score matrix.
__global__ void scores_kernel(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              float* __restrict__ S,
                              int N, int D, float inv_sqrtd) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    const float* qr = Q + static_cast<size_t>(i) * D;
    const float* kr = K + static_cast<size_t>(j) * D;
    float s = 0.0f;
    for (int k = 0; k < D; ++k) s += qr[k] * kr[k];
    S[static_cast<size_t>(i) * N + j] = s * inv_sqrtd;
}

// Per-row masked softmax of an (N, N) matrix in place into Attn.
// One block per row i. Invalid query rows (mask[i]==0) get all-zero rows.
// Within a valid row, masked columns (mask[j]==0) get probability 0.
constexpr int ROW_SM_BLOCK = 256;
__global__ void row_masked_softmax_kernel(const float* __restrict__ scores,
                                          float* __restrict__ Attn,
                                          const float* __restrict__ mask,
                                          int N) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(i) * N;
    float* arow = Attn + static_cast<size_t>(i) * N;

    // Invalid query → zero row.
    if (mask && mask[i] == 0.0f) {
        for (int j = tid; j < N; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    // Phase 1: max over valid keys.
    float local_max = -1e30f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] == 0.0f) continue;
        const float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    // Phase 2: exp(x - m), zero on masked, accumulate sum.
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] == 0.0f) {
            arow[j] = 0.0f;
            continue;
        }
        const float e = expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        arow[j] = arow[j] * inv;
    }
}

// Y(i, k) = sum_j Attn(i, j) * V(j, k).  One thread per (i, k).
__global__ void attn_apply_v_kernel(const float* __restrict__ Attn,
                                    const float* __restrict__ V,
                                    float* __restrict__ Y,
                                    int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    const float* arow = Attn + static_cast<size_t>(i) * N;
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) acc += arow[j] * V[static_cast<size_t>(j) * D + k];
    Y[static_cast<size_t>(i) * D + k] = acc;
}

// O(i, c) = mask[i] ? sum_k Y(i, k) * Wo(c, k) : 0
__global__ void output_proj_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ Wo,
                                   const float* __restrict__ mask,
                                   float* __restrict__ O,
                                   int N, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || c >= D) return;
    if (mask && mask[i] == 0.0f) {
        O[static_cast<size_t>(i) * D + c] = 0.0f;
        return;
    }
    const float* yr = Y + static_cast<size_t>(i) * D;
    const float* wr = Wo + static_cast<size_t>(c) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
    O[static_cast<size_t>(i) * D + c] = acc;
}

// ─── Backward ─────────────────────────────────────────────────────────────

// dY(i, k) = mask[i] ? sum_c Wo(c, k) * dO(i, c) : 0
// dWo(c, k) += sum_i mask[i] * dO(i, c) * Y(i, k)   — but we accumulate per (c,k)
//
// Step A: compute dY  (one thread per (i, k))
__global__ void wo_back_dY_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ Wo,
                                  const float* __restrict__ mask,
                                  float* __restrict__ dY,
                                  int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    if (mask && mask[i] == 0.0f) {
        dY[static_cast<size_t>(i) * D + k] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (int c = 0; c < D; ++c) {
        acc += Wo[static_cast<size_t>(c) * D + k] * dO[static_cast<size_t>(i) * D + c];
    }
    dY[static_cast<size_t>(i) * D + k] = acc;
}

// Step B: dWo(c, k) += sum_i (mask[i] ? dO(i,c) * Y(i,k) : 0)
__global__ void wo_back_dW_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ Y,
                                  const float* __restrict__ mask,
                                  float* __restrict__ dWo,
                                  int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (mask && mask[i] == 0.0f) continue;
        acc += dO[static_cast<size_t>(i) * D + c] * Y[static_cast<size_t>(i) * D + k];
    }
    dWo[static_cast<size_t>(c) * D + k] += acc;
}

// dAttn(i, j) = sum_k dY(i, k) * V(j, k)    — one thread per (i, j)
__global__ void dAttn_kernel(const float* __restrict__ dY,
                             const float* __restrict__ V,
                             float* __restrict__ dAttn,
                             int N, int D) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) {
        acc += dY[static_cast<size_t>(i) * D + k] * V[static_cast<size_t>(j) * D + k];
    }
    dAttn[static_cast<size_t>(i) * N + j] = acc;
}

// dV(j, k) = sum_i Attn(i, j) * dY(i, k)    — one thread per (j, k)
__global__ void dV_kernel(const float* __restrict__ Attn,
                          const float* __restrict__ dY,
                          float* __restrict__ dV,
                          int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= N || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += Attn[static_cast<size_t>(i) * N + j] * dY[static_cast<size_t>(i) * D + k];
    }
    dV[static_cast<size_t>(j) * D + k] = acc;
}

// Per-row softmax backward into dScores.
// Row i with mask[i]==0 → all zero.
// Otherwise: row_dz[j] = p_j * (dp_j - sum_l dp_l * p_l), then zero where mask[j]==0,
// finally scale by inv_sqrtd.
__global__ void row_softmax_back_kernel(const float* __restrict__ Attn,
                                        const float* __restrict__ dAttn,
                                        const float* __restrict__ mask,
                                        float* __restrict__ dScores,
                                        int N, float inv_sqrtd) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const float* prow  = Attn  + static_cast<size_t>(i) * N;
    const float* dprow = dAttn + static_cast<size_t>(i) * N;
    float* drow = dScores + static_cast<size_t>(i) * N;

    if (mask && mask[i] == 0.0f) {
        for (int j = tid; j < N; j += blockDim.x) drow[j] = 0.0f;
        return;
    }

    float local = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        local += dprow[j] * prow[j];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] == 0.0f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtd;
        }
    }
}

// dQ(i, k) = sum_j dScores(i, j) * K(j, k)
__global__ void dQ_kernel(const float* __restrict__ dScores,
                          const float* __restrict__ K,
                          float* __restrict__ dQ,
                          int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += dScores[static_cast<size_t>(i) * N + j] * K[static_cast<size_t>(j) * D + k];
    }
    dQ[static_cast<size_t>(i) * D + k] = acc;
}

// dK(j, k) = sum_i dScores(i, j) * Q(i, k)
__global__ void dK_kernel(const float* __restrict__ dScores,
                          const float* __restrict__ Q,
                          float* __restrict__ dK,
                          int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= N || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += dScores[static_cast<size_t>(i) * N + j] * Q[static_cast<size_t>(i) * D + k];
    }
    dK[static_cast<size_t>(j) * D + k] = acc;
}

// Final input-projection backward, mirroring CPU's fused loop:
//   dWq(j, k) += sum_i dQ(i, j) * X(i, k)        (and likewise dWk, dWv)
//   dX(i, k)   = sum_j dQ(i, j) * Wq(j, k)
//              + sum_j dK(i, j) * Wk(j, k)
//              + sum_j dV(i, j) * Wv(j, k)
//
// Two kernels: dW accumulation (one thread per (j, k)) and dX (one per (i, k)).
__global__ void dWqkv_kernel(const float* __restrict__ dQ,
                             const float* __restrict__ dK,
                             const float* __restrict__ dV,
                             const float* __restrict__ X,
                             float* __restrict__ dWq,
                             float* __restrict__ dWk,
                             float* __restrict__ dWv,
                             int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= D || k >= D) return;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float xv = X[static_cast<size_t>(i) * D + k];
        aq += dQ[static_cast<size_t>(i) * D + j] * xv;
        ak += dK[static_cast<size_t>(i) * D + j] * xv;
        av += dV[static_cast<size_t>(i) * D + j] * xv;
    }
    const size_t idx = static_cast<size_t>(j) * D + k;
    dWq[idx] += aq;
    dWk[idx] += ak;
    dWv[idx] += av;
}

__global__ void dX_proj_kernel(const float* __restrict__ dQ,
                               const float* __restrict__ dK,
                               const float* __restrict__ dV,
                               const float* __restrict__ Wq,
                               const float* __restrict__ Wk,
                               const float* __restrict__ Wv,
                               float* __restrict__ dX,
                               int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    float acc = 0.0f;
    for (int j = 0; j < D; ++j) {
        const size_t widx = static_cast<size_t>(j) * D + k;
        acc += dQ[static_cast<size_t>(i) * D + j] * Wq[widx]
             + dK[static_cast<size_t>(i) * D + j] * Wk[widx]
             + dV[static_cast<size_t>(i) * D + j] * Wv[widx];
    }
    dX[static_cast<size_t>(i) * D + k] = acc;
}

inline dim3 grid2d(int x, int y, dim3 block) {
    return dim3((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);
}

} // namespace

void attention_forward_gpu(const GpuTensor& X,
                           const GpuTensor& Wq, const GpuTensor& Wk,
                           const GpuTensor& Wv, const GpuTensor& Wo,
                           const float* d_mask,
                           GpuTensor& Q, GpuTensor& K, GpuTensor& V,
                           GpuTensor& Attn, GpuTensor& Y_pre_Wo,
                           GpuTensor& O) {
    const int N = X.rows;
    const int D = X.cols;
    if (Q.rows != N || Q.cols != D) Q.resize(N, D);
    if (K.rows != N || K.cols != D) K.resize(N, D);
    if (V.rows != N || V.cols != D) V.resize(N, D);
    if (Attn.rows != N || Attn.cols != N) Attn.resize(N, N);
    if (Y_pre_Wo.rows != N || Y_pre_Wo.cols != D) Y_pre_Wo.resize(N, D);
    if (O.rows != N || O.cols != D) O.resize(N, D);
    if (N == 0 || D == 0) return;

    const dim3 block_nd(16, 16);
    const dim3 block_nn(16, 16);
    const dim3 block_dd(16, 16);

    // Q, K, V projections.
    {
        dim3 grid = grid2d(D, N, block_nd);
        matmul_xwT_kernel<<<grid, block_nd>>>(X.data, Wq.data, Q.data, N, D, D);
        matmul_xwT_kernel<<<grid, block_nd>>>(X.data, Wk.data, K.data, N, D, D);
        matmul_xwT_kernel<<<grid, block_nd>>>(X.data, Wv.data, V.data, N, D, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // Scores (N, N).
    GpuTensor scores(N, N);
    {
        const float inv_sqrtd = 1.0f / sqrtf(static_cast<float>(D));
        dim3 grid = grid2d(N, N, block_nn);
        scores_kernel<<<grid, block_nn>>>(Q.data, K.data, scores.data,
                                          N, D, inv_sqrtd);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // Row-masked softmax → Attn.
    row_masked_softmax_kernel<<<N, ROW_SM_BLOCK>>>(scores.data, Attn.data,
                                                   d_mask, N);
    BGA_CUDA_CHECK(cudaGetLastError());

    // Y_pre_Wo = Attn @ V.
    {
        dim3 grid = grid2d(D, N, block_nd);
        attn_apply_v_kernel<<<grid, block_nd>>>(Attn.data, V.data, Y_pre_Wo.data,
                                                N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // O = Y @ Wo^T (with row mask zeroing).
    {
        dim3 grid = grid2d(D, N, block_nd);
        output_proj_kernel<<<grid, block_nd>>>(Y_pre_Wo.data, Wo.data, d_mask,
                                               O.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    (void)block_dd;
}

void attention_backward_gpu(const GpuTensor& dO,
                            const GpuTensor& X,
                            const GpuTensor& Q, const GpuTensor& K,
                            const GpuTensor& V, const GpuTensor& Attn,
                            const GpuTensor& Y_pre_Wo,
                            const GpuTensor& Wq, const GpuTensor& Wk,
                            const GpuTensor& Wv, const GpuTensor& Wo,
                            const float* d_mask,
                            GpuTensor& dX,
                            GpuTensor& dWq, GpuTensor& dWk,
                            GpuTensor& dWv, GpuTensor& dWo) {
    const int N = X.rows;
    const int D = X.cols;
    if (dX.rows != N || dX.cols != D) dX.resize(N, D);
    if (N == 0 || D == 0) return;

    const float inv_sqrtd = 1.0f / sqrtf(static_cast<float>(D));
    const dim3 block_nd(16, 16);
    const dim3 block_nn(16, 16);
    const dim3 block_dd(16, 16);

    // dY (N, D) and dWo accumulation.
    GpuTensor dY(N, D);
    {
        dim3 grid = grid2d(D, N, block_nd);
        wo_back_dY_kernel<<<grid, block_nd>>>(dO.data, Wo.data, d_mask,
                                              dY.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid = grid2d(D, D, block_dd);
        wo_back_dW_kernel<<<grid, block_dd>>>(dO.data, Y_pre_Wo.data, d_mask,
                                              dWo.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dAttn (N, N), dV (N, D).
    GpuTensor dAttn(N, N);
    GpuTensor dV(N, D);
    {
        dim3 grid = grid2d(N, N, block_nn);
        dAttn_kernel<<<grid, block_nn>>>(dY.data, V.data, dAttn.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid = grid2d(D, N, block_nd);
        dV_kernel<<<grid, block_nd>>>(Attn.data, dY.data, dV.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dScores via per-row softmax backward, scaled by inv_sqrtd.
    GpuTensor dScores(N, N);
    row_softmax_back_kernel<<<N, ROW_SM_BLOCK>>>(Attn.data, dAttn.data, d_mask,
                                                 dScores.data, N, inv_sqrtd);
    BGA_CUDA_CHECK(cudaGetLastError());

    // dQ, dK.
    GpuTensor dQ(N, D);
    GpuTensor dK(N, D);
    {
        dim3 grid = grid2d(D, N, block_nd);
        dQ_kernel<<<grid, block_nd>>>(dScores.data, K.data, dQ.data, N, D);
        dK_kernel<<<grid, block_nd>>>(dScores.data, Q.data, dK.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dWq/dWk/dWv accumulation and dX.
    {
        dim3 grid = grid2d(D, D, block_dd);
        dWqkv_kernel<<<grid, block_dd>>>(dQ.data, dK.data, dV.data, X.data,
                                         dWq.data, dWk.data, dWv.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid = grid2d(D, N, block_nd);
        dX_proj_kernel<<<grid, block_nd>>>(dQ.data, dK.data, dV.data,
                                           Wq.data, Wk.data, Wv.data,
                                           dX.data, N, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace brogameagent::nn::gpu
