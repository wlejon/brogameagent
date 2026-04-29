#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#include <cuda_runtime.h>

namespace brogameagent::nn::gpu {

// Naive correctness-first kernels mirroring src/nn/multi_head_attention.cpp.
// Heads are stored stacked along rows:
//   Qh / Kh / Vh   : (h * K, dh)   — head hh = rows [hh*K, (hh+1)*K)
//   Attnh          : (h * K, K)
//   Yconcat        : (K, D)        — head hh occupies cols [hh*dh, (hh+1)*dh)

namespace {

constexpr int ROW_SM_BLOCK = 256;

inline dim3 grid2d(int x, int y, dim3 block) {
    return dim3((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);
}

// Per-head Q/K/V projection: out(i, j) = sum_k X(i, k) * W(hh*dh + j, k)
//   X: (K, D)
//   W: (D, D)
//   Out: (h*K, dh) — writes head hh slice [hh*K, (hh+1)*K)
// One thread per (i, j) within head hh; grid.z = h.
__global__ void mha_proj_kernel(const float* __restrict__ X,
                                const float* __restrict__ W,
                                float* __restrict__ Out,
                                int K, int D, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= dh) return;
    const int row_off = hh * dh;
    const float* xr = X + static_cast<size_t>(i) * D;
    const float* wr = W + static_cast<size_t>(row_off + j) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
    const size_t out_row = static_cast<size_t>(hh) * K + i;
    Out[out_row * dh + j] = acc;
}

// Per-head scores: S(i, j) = (Q_h(i) . K_h(j)) / sqrt(dh)
//   Qh, Kh: (h*K, dh)  S: (h*K, K)
__global__ void mha_scores_kernel(const float* __restrict__ Qh,
                                  const float* __restrict__ Kh,
                                  float* __restrict__ S,
                                  int K, int dh, float inv_sqrtdh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= K) return;
    const size_t qrow = (static_cast<size_t>(hh) * K + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * K + j) * dh;
    float s = 0.0f;
    for (int k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    const size_t srow = (static_cast<size_t>(hh) * K + i) * K;
    S[srow + j] = s * inv_sqrtdh;
}

// Per-row masked softmax of (h*K, K). One block per (head, row).
// blockIdx.x = row index across heads (== hh*K + i).
__global__ void mha_row_softmax_kernel(const float* __restrict__ scores,
                                       float* __restrict__ Attn,
                                       const float* __restrict__ mask,
                                       int K) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int i_within = row % K;   // query index within head
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(row) * K;
    float* arow = Attn + static_cast<size_t>(row) * K;

    if (mask && mask[i_within] == 0.0f) {
        for (int j = tid; j < K; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < K; j += blockDim.x) {
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

    float local_sum = 0.0f;
    for (int j = tid; j < K; j += blockDim.x) {
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

    for (int j = tid; j < K; j += blockDim.x) {
        arow[j] = arow[j] * inv;
    }
}

// Per-head Y_h(i, k) = sum_j Attnh(i, j) * Vh(j, k), then write into
// Yconcat(i, hh*dh + k).
__global__ void mha_attn_apply_v_kernel(const float* __restrict__ Attnh,
                                        const float* __restrict__ Vh,
                                        float* __restrict__ Yconcat,
                                        int K, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || k >= dh) return;
    const size_t arow = (static_cast<size_t>(hh) * K + i) * K;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const size_t vrow = (static_cast<size_t>(hh) * K + j) * dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
}

// O(i, c) = mask[i] ? sum_k Yconcat(i, k) * Wo(c, k) : 0.
__global__ void mha_output_proj_kernel(const float* __restrict__ Y,
                                       const float* __restrict__ Wo,
                                       const float* __restrict__ mask,
                                       float* __restrict__ O,
                                       int K, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || c >= D) return;
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

// ─── Backward kernels ─────────────────────────────────────────────────────

// dWo(c, k) += sum_i mask_i * dO(i, c) * Yconcat(i, k)   (one thread per (c, k))
__global__ void mha_wo_back_dW_kernel(const float* __restrict__ dO,
                                      const float* __restrict__ Y,
                                      const float* __restrict__ mask,
                                      float* __restrict__ dWo,
                                      int K, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        if (mask && mask[i] == 0.0f) continue;
        acc += dO[static_cast<size_t>(i) * D + c] * Y[static_cast<size_t>(i) * D + k];
    }
    dWo[static_cast<size_t>(c) * D + k] += acc;
}

// dYconcat(i, k) = mask_i ? sum_c Wo(c, k) * dO(i, c) : 0
__global__ void mha_wo_back_dY_kernel(const float* __restrict__ dO,
                                      const float* __restrict__ Wo,
                                      const float* __restrict__ mask,
                                      float* __restrict__ dY,
                                      int K, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || k >= D) return;
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

// Per-head dAttn(hh, i, j) = sum_k dYh(hh, i, k) * Vh(hh, j, k).
// dYh is the appropriate slice of dYconcat; we read it directly with col offset.
__global__ void mha_dAttn_kernel(const float* __restrict__ dYconcat,
                                 const float* __restrict__ Vh,
                                 float* __restrict__ dAttn,
                                 int K, int dh, int D) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= K) return;
    float acc = 0.0f;
    for (int k = 0; k < dh; ++k) {
        const float dy = dYconcat[static_cast<size_t>(i) * D + (hh * dh + k)];
        const float vv = Vh[(static_cast<size_t>(hh) * K + j) * dh + k];
        acc += dy * vv;
    }
    dAttn[(static_cast<size_t>(hh) * K + i) * K + j] = acc;
}

// dVh(hh, j, k) = sum_i Attnh(hh, i, j) * dYh(hh, i, k)
__global__ void mha_dV_kernel(const float* __restrict__ Attnh,
                              const float* __restrict__ dYconcat,
                              float* __restrict__ dVh,
                              int K, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (j >= K || k >= dh) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float a  = Attnh[(static_cast<size_t>(hh) * K + i) * K + j];
        const float dy = dYconcat[static_cast<size_t>(i) * D + (hh * dh + k)];
        acc += a * dy;
    }
    dVh[(static_cast<size_t>(hh) * K + j) * dh + k] = acc;
}

// Per-row softmax backward. blockIdx.x indexes rows (hh*K + i).
__global__ void mha_row_softmax_back_kernel(const float* __restrict__ Attn,
                                            const float* __restrict__ dAttn,
                                            const float* __restrict__ mask,
                                            float* __restrict__ dScores,
                                            int K, float inv_sqrtdh) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int i_within = row % K;
    const int tid = threadIdx.x;
    const float* prow  = Attn  + static_cast<size_t>(row) * K;
    const float* dprow = dAttn + static_cast<size_t>(row) * K;
    float* drow = dScores + static_cast<size_t>(row) * K;

    if (mask && mask[i_within] == 0.0f) {
        for (int j = tid; j < K; j += blockDim.x) drow[j] = 0.0f;
        return;
    }

    float local = 0.0f;
    for (int j = tid; j < K; j += blockDim.x) {
        local += dprow[j] * prow[j];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int j = tid; j < K; j += blockDim.x) {
        if (mask && mask[j] == 0.0f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
        }
    }
}

// dQh(hh, i, k) = sum_j dScores(hh, i, j) * Kh(hh, j, k)
__global__ void mha_dQ_kernel(const float* __restrict__ dScores,
                              const float* __restrict__ Kh,
                              float* __restrict__ dQh,
                              int K, int dh) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || k >= dh) return;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const float ds = dScores[(static_cast<size_t>(hh) * K + i) * K + j];
        const float kk = Kh[(static_cast<size_t>(hh) * K + j) * dh + k];
        acc += ds * kk;
    }
    dQh[(static_cast<size_t>(hh) * K + i) * dh + k] = acc;
}

// dKh(hh, j, k) = sum_i dScores(hh, i, j) * Qh(hh, i, k)
__global__ void mha_dK_kernel(const float* __restrict__ dScores,
                              const float* __restrict__ Qh,
                              float* __restrict__ dKh,
                              int K, int dh) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (j >= K || k >= dh) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float ds = dScores[(static_cast<size_t>(hh) * K + i) * K + j];
        const float qq = Qh[(static_cast<size_t>(hh) * K + i) * dh + k];
        acc += ds * qq;
    }
    dKh[(static_cast<size_t>(hh) * K + j) * dh + k] = acc;
}

// Final input-projection backward, mirroring CPU:
//   dWq(hh*dh + j, k) += sum_i dQh(hh, i, j) * X(i, k)        (and Wk, Wv)
// One thread per global (D, D) cell of the dWq/dWk/dWv accumulators.
__global__ void mha_dWqkv_kernel(const float* __restrict__ dQh,
                                 const float* __restrict__ dKh,
                                 const float* __restrict__ dVh,
                                 const float* __restrict__ X,
                                 float* __restrict__ dWq,
                                 float* __restrict__ dWk,
                                 float* __restrict__ dWv,
                                 int K, int D, int dh, int H) {
    const int k_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int wrow  = blockIdx.y * blockDim.y + threadIdx.y;
    if (wrow >= D || k_col >= D) return;
    const int hh = wrow / dh;
    const int j  = wrow % dh;
    (void)H;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float xv = X[static_cast<size_t>(i) * D + k_col];
        aq += dQh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
        ak += dKh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
        av += dVh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
    }
    const size_t idx = static_cast<size_t>(wrow) * D + k_col;
    dWq[idx] += aq;
    dWk[idx] += ak;
    dWv[idx] += av;
}

// dX(i, k) = sum over heads, j: dQh*Wq + dKh*Wk + dVh*Wv at (hh*dh+j, k).
__global__ void mha_dX_proj_kernel(const float* __restrict__ dQh,
                                   const float* __restrict__ dKh,
                                   const float* __restrict__ dVh,
                                   const float* __restrict__ Wq,
                                   const float* __restrict__ Wk,
                                   const float* __restrict__ Wv,
                                   float* __restrict__ dX,
                                   int K, int D, int dh, int H) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || k >= D) return;
    float acc = 0.0f;
    for (int hh = 0; hh < H; ++hh) {
        for (int j = 0; j < dh; ++j) {
            const int wrow = hh * dh + j;
            const size_t widx = static_cast<size_t>(wrow) * D + k;
            const float gq = dQh[(static_cast<size_t>(hh) * K + i) * dh + j];
            const float gk = dKh[(static_cast<size_t>(hh) * K + i) * dh + j];
            const float gv = dVh[(static_cast<size_t>(hh) * K + i) * dh + j];
            acc += gq * Wq[widx] + gk * Wk[widx] + gv * Wv[widx];
        }
    }
    dX[static_cast<size_t>(i) * D + k] = acc;
}

} // namespace

void mha_forward_gpu(const GpuTensor& X,
                     const GpuTensor& Wq, const GpuTensor& Wk,
                     const GpuTensor& Wv, const GpuTensor& Wo,
                     const float* d_mask,
                     int num_heads,
                     GpuTensor& Qh, GpuTensor& Kh, GpuTensor& Vh,
                     GpuTensor& Attnh, GpuTensor& Yconcat,
                     GpuTensor& O) {
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = D / H;
    if (Qh.rows != H * K || Qh.cols != dh) Qh.resize(H * K, dh);
    if (Kh.rows != H * K || Kh.cols != dh) Kh.resize(H * K, dh);
    if (Vh.rows != H * K || Vh.cols != dh) Vh.resize(H * K, dh);
    if (Attnh.rows != H * K || Attnh.cols != K) Attnh.resize(H * K, K);
    if (Yconcat.rows != K || Yconcat.cols != D) Yconcat.resize(K, D);
    if (O.rows != K || O.cols != D) O.resize(K, D);
    if (K == 0 || D == 0 || H == 0) return;

    const dim3 block_kdh(16, 16);
    const dim3 block_kk(16, 16);
    const dim3 block_kd(16, 16);
    const dim3 block_dd(16, 16);

    // Q/K/V projections.
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_proj_kernel<<<grid, block_kdh>>>(X.data, Wq.data, Qh.data, K, D, dh);
        mha_proj_kernel<<<grid, block_kdh>>>(X.data, Wk.data, Kh.data, K, D, dh);
        mha_proj_kernel<<<grid, block_kdh>>>(X.data, Wv.data, Vh.data, K, D, dh);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // Scores → reuse Attnh storage temporarily for raw scores.
    GpuTensor scores(H * K, K);
    {
        const float inv_sqrtdh = 1.0f / sqrtf(static_cast<float>(dh));
        dim3 grid(((K) + block_kk.x - 1) / block_kk.x,
                  ((K) + block_kk.y - 1) / block_kk.y,
                  H);
        mha_scores_kernel<<<grid, block_kk>>>(Qh.data, Kh.data, scores.data,
                                              K, dh, inv_sqrtdh);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // Row-masked softmax across all H*K rows.
    mha_row_softmax_kernel<<<H * K, ROW_SM_BLOCK>>>(scores.data, Attnh.data,
                                                    d_mask, K);
    BGA_CUDA_CHECK(cudaGetLastError());

    // Attn @ V → Yconcat.
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_attn_apply_v_kernel<<<grid, block_kdh>>>(Attnh.data, Vh.data,
                                                     Yconcat.data, K, dh, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // Output projection.
    {
        dim3 grid(((D) + block_kd.x - 1) / block_kd.x,
                  ((K) + block_kd.y - 1) / block_kd.y);
        mha_output_proj_kernel<<<grid, block_kd>>>(Yconcat.data, Wo.data, d_mask,
                                                   O.data, K, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    (void)block_dd;
}

void mha_backward_gpu(const GpuTensor& dO,
                      const GpuTensor& X,
                      const GpuTensor& Qh, const GpuTensor& Kh,
                      const GpuTensor& Vh, const GpuTensor& Attnh,
                      const GpuTensor& Yconcat,
                      const GpuTensor& Wq, const GpuTensor& Wk,
                      const GpuTensor& Wv, const GpuTensor& Wo,
                      const float* d_mask,
                      int num_heads,
                      GpuTensor& dX,
                      GpuTensor& dWq, GpuTensor& dWk,
                      GpuTensor& dWv, GpuTensor& dWo) {
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = D / H;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    if (K == 0 || D == 0 || H == 0) return;

    const float inv_sqrtdh = 1.0f / sqrtf(static_cast<float>(dh));
    const dim3 block_kd(16, 16);
    const dim3 block_kk(16, 16);
    const dim3 block_dd(16, 16);
    const dim3 block_kdh(16, 16);

    // dYconcat (K, D) and dWo accumulation.
    GpuTensor dYconcat(K, D);
    {
        dim3 grid(((D) + block_kd.x - 1) / block_kd.x,
                  ((K) + block_kd.y - 1) / block_kd.y);
        mha_wo_back_dY_kernel<<<grid, block_kd>>>(dO.data, Wo.data, d_mask,
                                                  dYconcat.data, K, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(((D) + block_dd.x - 1) / block_dd.x,
                  ((D) + block_dd.y - 1) / block_dd.y);
        mha_wo_back_dW_kernel<<<grid, block_dd>>>(dO.data, Yconcat.data, d_mask,
                                                  dWo.data, K, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dAttn (H*K, K), dVh (H*K, dh).
    GpuTensor dAttn(H * K, K);
    GpuTensor dVh(H * K, dh);
    {
        dim3 grid(((K) + block_kk.x - 1) / block_kk.x,
                  ((K) + block_kk.y - 1) / block_kk.y,
                  H);
        mha_dAttn_kernel<<<grid, block_kk>>>(dYconcat.data, Vh.data, dAttn.data,
                                             K, dh, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_dV_kernel<<<grid, block_kdh>>>(Attnh.data, dYconcat.data, dVh.data,
                                           K, dh, D);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dScores (H*K, K) via per-row softmax backward.
    GpuTensor dScores(H * K, K);
    mha_row_softmax_back_kernel<<<H * K, ROW_SM_BLOCK>>>(Attnh.data, dAttn.data,
                                                         d_mask, dScores.data,
                                                         K, inv_sqrtdh);
    BGA_CUDA_CHECK(cudaGetLastError());

    // dQh, dKh.
    GpuTensor dQh(H * K, dh);
    GpuTensor dKh(H * K, dh);
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_dQ_kernel<<<grid, block_kdh>>>(dScores.data, Kh.data, dQh.data,
                                           K, dh);
        mha_dK_kernel<<<grid, block_kdh>>>(dScores.data, Qh.data, dKh.data,
                                           K, dh);
        BGA_CUDA_CHECK(cudaGetLastError());
    }

    // dWq/dWk/dWv accumulation and dX.
    {
        dim3 grid(((D) + block_dd.x - 1) / block_dd.x,
                  ((D) + block_dd.y - 1) / block_dd.y);
        mha_dWqkv_kernel<<<grid, block_dd>>>(dQh.data, dKh.data, dVh.data, X.data,
                                             dWq.data, dWk.data, dWv.data,
                                             K, D, dh, H);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(((D) + block_kd.x - 1) / block_kd.x,
                  ((K) + block_kd.y - 1) / block_kd.y);
        mha_dX_proj_kernel<<<grid, block_kd>>>(dQh.data, dKh.data, dVh.data,
                                               Wq.data, Wk.data, Wv.data,
                                               dX.data, K, D, dh, H);
        BGA_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace brogameagent::nn::gpu
