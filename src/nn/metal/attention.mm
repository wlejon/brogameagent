#include <brogameagent/nn/gpu/ops.h>
#include <brogameagent/nn/gpu/runtime.h>

#import "internal.h"

#include <cmath>

namespace brogameagent::nn::gpu {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;

namespace {

constexpr NSUInteger ROW_SM_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint ROW_SM_BLOCK = 256;

// Y(i, j) = sum_k X(i, k) * W(j, k)
kernel void k_matmul_xwT(device const float* X [[buffer(0)]],
                         device const float* W [[buffer(1)]],
                         device float*       Y [[buffer(2)]],
                         constant uint& N      [[buffer(3)]],
                         constant uint& D_in   [[buffer(4)]],
                         constant uint& D_out  [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y;
    if (i >= N || j >= D_out) return;
    device const float* xr = X + i * D_in;
    device const float* wr = W + j * D_in;
    float acc = 0.0f;
    for (uint k = 0; k < D_in; ++k) acc += xr[k] * wr[k];
    Y[i * D_out + j] = acc;
}

kernel void k_scores(device const float* Q [[buffer(0)]],
                     device const float* K [[buffer(1)]],
                     device float*       S [[buffer(2)]],
                     constant uint& N      [[buffer(3)]],
                     constant uint& D      [[buffer(4)]],
                     constant float& inv_sqrtd [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y;
    if (i >= N || j >= N) return;
    device const float* qr = Q + i * D;
    device const float* kr = K + j * D;
    float s = 0.0f;
    for (uint k = 0; k < D; ++k) s += qr[k] * kr[k];
    S[i * N + j] = s * inv_sqrtd;
}

kernel void k_row_masked_softmax(device const float* scores [[buffer(0)]],
                                 device float*       Attn   [[buffer(1)]],
                                 device const float* mask   [[buffer(2)]],
                                 constant uint& has_mask    [[buffer(3)]],
                                 constant uint& N           [[buffer(4)]],
                                 uint i [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    device const float* srow = scores + i * N;
    device float*       arow = Attn + i * N;
    if (has_mask && mask[i] == 0.0f) {
        for (uint j = tid; j < N; j += tg_size) arow[j] = 0.0f;
        return;
    }
    float local_max = -1e30f;
    for (uint j = tid; j < N; j += tg_size) {
        if (has_mask && mask[j] == 0.0f) continue;
        float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = sdata[0];

    float local_sum = 0.0f;
    for (uint j = tid; j < N; j += tg_size) {
        if (has_mask && mask[j] == 0.0f) {
            arow[j] = 0.0f;
            continue;
        }
        float e = exp(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = sdata[0];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint j = tid; j < N; j += tg_size) {
        arow[j] = arow[j] * inv;
    }
}

kernel void k_attn_apply_v(device const float* Attn [[buffer(0)]],
                           device const float* V    [[buffer(1)]],
                           device float*       Y    [[buffer(2)]],
                           constant uint& N         [[buffer(3)]],
                           constant uint& D         [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= N || k >= D) return;
    device const float* arow = Attn + i * N;
    float acc = 0.0f;
    for (uint j = 0; j < N; ++j) acc += arow[j] * V[j * D + k];
    Y[i * D + k] = acc;
}

kernel void k_output_proj(device const float* Y    [[buffer(0)]],
                          device const float* Wo   [[buffer(1)]],
                          device const float* mask [[buffer(2)]],
                          constant uint& has_mask  [[buffer(3)]],
                          device float*       O    [[buffer(4)]],
                          constant uint& N         [[buffer(5)]],
                          constant uint& D         [[buffer(6)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x; uint i = gid.y;
    if (i >= N || c >= D) return;
    if (has_mask && mask[i] == 0.0f) {
        O[i * D + c] = 0.0f;
        return;
    }
    device const float* yr = Y + i * D;
    device const float* wr = Wo + c * D;
    float acc = 0.0f;
    for (uint k = 0; k < D; ++k) acc += yr[k] * wr[k];
    O[i * D + c] = acc;
}

kernel void k_wo_back_dY(device const float* dO  [[buffer(0)]],
                         device const float* Wo  [[buffer(1)]],
                         device const float* mask[[buffer(2)]],
                         constant uint& has_mask [[buffer(3)]],
                         device float*       dY  [[buffer(4)]],
                         constant uint& N        [[buffer(5)]],
                         constant uint& D        [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= N || k >= D) return;
    if (has_mask && mask[i] == 0.0f) {
        dY[i * D + k] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (uint c = 0; c < D; ++c) {
        acc += Wo[c * D + k] * dO[i * D + c];
    }
    dY[i * D + k] = acc;
}

kernel void k_wo_back_dW(device const float* dO  [[buffer(0)]],
                         device const float* Y   [[buffer(1)]],
                         device const float* mask[[buffer(2)]],
                         constant uint& has_mask [[buffer(3)]],
                         device float*       dWo [[buffer(4)]],
                         constant uint& N        [[buffer(5)]],
                         constant uint& D        [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint c = gid.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
        if (has_mask && mask[i] == 0.0f) continue;
        acc += dO[i * D + c] * Y[i * D + k];
    }
    dWo[c * D + k] += acc;
}

kernel void k_dAttn(device const float* dY    [[buffer(0)]],
                    device const float* V     [[buffer(1)]],
                    device float*       dAttn [[buffer(2)]],
                    constant uint& N          [[buffer(3)]],
                    constant uint& D          [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y;
    if (i >= N || j >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < D; ++k) {
        acc += dY[i * D + k] * V[j * D + k];
    }
    dAttn[i * N + j] = acc;
}

kernel void k_dV(device const float* Attn [[buffer(0)]],
                 device const float* dY   [[buffer(1)]],
                 device float*       dV   [[buffer(2)]],
                 constant uint& N         [[buffer(3)]],
                 constant uint& D         [[buffer(4)]],
                 uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y;
    if (j >= N || k >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
        acc += Attn[i * N + j] * dY[i * D + k];
    }
    dV[j * D + k] = acc;
}

kernel void k_row_softmax_back(device const float* Attn    [[buffer(0)]],
                               device const float* dAttn   [[buffer(1)]],
                               device const float* mask    [[buffer(2)]],
                               constant uint& has_mask     [[buffer(3)]],
                               device float*       dScores [[buffer(4)]],
                               constant uint& N            [[buffer(5)]],
                               constant float& inv_sqrtd   [[buffer(6)]],
                               uint i [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    device const float* prow  = Attn  + i * N;
    device const float* dprow = dAttn + i * N;
    device float*       drow  = dScores + i * N;
    if (has_mask && mask[i] == 0.0f) {
        for (uint j = tid; j < N; j += tg_size) drow[j] = 0.0f;
        return;
    }
    float local = 0.0f;
    for (uint j = tid; j < N; j += tg_size) {
        local += dprow[j] * prow[j];
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = sdata[0];
    for (uint j = tid; j < N; j += tg_size) {
        if (has_mask && mask[j] == 0.0f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtd;
        }
    }
}

kernel void k_dQ(device const float* dScores [[buffer(0)]],
                 device const float* K       [[buffer(1)]],
                 device float*       dQ      [[buffer(2)]],
                 constant uint& N            [[buffer(3)]],
                 constant uint& D            [[buffer(4)]],
                 uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= N || k >= D) return;
    float acc = 0.0f;
    for (uint j = 0; j < N; ++j) {
        acc += dScores[i * N + j] * K[j * D + k];
    }
    dQ[i * D + k] = acc;
}

kernel void k_dK(device const float* dScores [[buffer(0)]],
                 device const float* Q       [[buffer(1)]],
                 device float*       dK      [[buffer(2)]],
                 constant uint& N            [[buffer(3)]],
                 constant uint& D            [[buffer(4)]],
                 uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y;
    if (j >= N || k >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
        acc += dScores[i * N + j] * Q[i * D + k];
    }
    dK[j * D + k] = acc;
}

kernel void k_dWqkv(device const float* dQ  [[buffer(0)]],
                    device const float* dK  [[buffer(1)]],
                    device const float* dV  [[buffer(2)]],
                    device const float* X   [[buffer(3)]],
                    device float*       dWq [[buffer(4)]],
                    device float*       dWk [[buffer(5)]],
                    device float*       dWv [[buffer(6)]],
                    constant uint& N        [[buffer(7)]],
                    constant uint& D        [[buffer(8)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y;
    if (j >= D || k >= D) return;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (uint i = 0; i < N; ++i) {
        float xv = X[i * D + k];
        aq += dQ[i * D + j] * xv;
        ak += dK[i * D + j] * xv;
        av += dV[i * D + j] * xv;
    }
    uint idx = j * D + k;
    dWq[idx] += aq;
    dWk[idx] += ak;
    dWv[idx] += av;
}

kernel void k_dX_proj(device const float* dQ  [[buffer(0)]],
                      device const float* dK  [[buffer(1)]],
                      device const float* dV  [[buffer(2)]],
                      device const float* Wq  [[buffer(3)]],
                      device const float* Wk  [[buffer(4)]],
                      device const float* Wv  [[buffer(5)]],
                      device float*       dX  [[buffer(6)]],
                      constant uint& N        [[buffer(7)]],
                      constant uint& D        [[buffer(8)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= N || k >= D) return;
    float acc = 0.0f;
    for (uint j = 0; j < D; ++j) {
        uint widx = j * D + k;
        acc += dQ[i * D + j] * Wq[widx]
             + dK[i * D + j] * Wk[widx]
             + dV[i * D + j] * Wv[widx];
    }
    dX[i * D + k] = acc;
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_proj, @"k_matmul_xwT")
DEF_PSO(pso_scores, @"k_scores")
DEF_PSO(pso_rsm, @"k_row_masked_softmax")
DEF_PSO(pso_av, @"k_attn_apply_v")
DEF_PSO(pso_op, @"k_output_proj")
DEF_PSO(pso_wodY, @"k_wo_back_dY")
DEF_PSO(pso_wodW, @"k_wo_back_dW")
DEF_PSO(pso_dAttn, @"k_dAttn")
DEF_PSO(pso_dV, @"k_dV")
DEF_PSO(pso_rsmb, @"k_row_softmax_back")
DEF_PSO(pso_dQ, @"k_dQ")
DEF_PSO(pso_dK, @"k_dK")
DEF_PSO(pso_dWqkv, @"k_dWqkv")
DEF_PSO(pso_dXp, @"k_dX_proj")
#undef DEF_PSO

void enc2d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
           NSUInteger nx, NSUInteger ny) {
    NSUInteger w = 16, h = 16;
    if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
    [enc dispatchThreads:MTLSizeMake(nx, ny, 1)
   threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
}

void run2d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        enc2d(enc, pso, nx, ny);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (rows == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(ROW_SM_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
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

    id<MTLBuffer> bX = buffer_for(X);
    NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bWq = buffer_for(Wq);
    NSUInteger oWq = buffer_offset_for(Wq);
    id<MTLBuffer> bWk = buffer_for(Wk);
    NSUInteger oWk = buffer_offset_for(Wk);
    id<MTLBuffer> bWv = buffer_for(Wv);
    NSUInteger oWv = buffer_offset_for(Wv);
    id<MTLBuffer> bWo = buffer_for(Wo);
    NSUInteger oWo = buffer_offset_for(Wo);
    id<MTLBuffer> bQ = buffer_for(Q);
    NSUInteger oQ = buffer_offset_for(Q);
    id<MTLBuffer> bK = buffer_for(K);
    NSUInteger oK = buffer_offset_for(K);
    id<MTLBuffer> bV = buffer_for(V);
    NSUInteger oV = buffer_offset_for(V);
    id<MTLBuffer> bA = buffer_for(Attn);
    NSUInteger oA = buffer_offset_for(Attn);
    id<MTLBuffer> bY = buffer_for(Y_pre_Wo);
    NSUInteger oY = buffer_offset_for(Y_pre_Wo);
    id<MTLBuffer> bO = buffer_for(O);
    NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? metal_impl::pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(D));

    // Q, K, V projections (Y = X @ W^T).
    auto proj = ^(id<MTLBuffer> bW, NSUInteger oWp, id<MTLBuffer> bOut, NSUInteger oOut) {
        run2d(pso_proj(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX offset:oX atIndex:0];
            [enc setBuffer:bW offset:oWp atIndex:1];
            [enc setBuffer:bOut offset:oOut atIndex:2];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        });
    };
    proj(bWq, oWq, bQ, oQ);
    proj(bWk, oWk, bK, oK);
    proj(bWv, oWv, bV, oV);

    // Scores → temp buffer.
    GpuTensor scores(N, N);
    id<MTLBuffer> bS = buffer_for(scores);
    NSUInteger oS = buffer_offset_for(scores);
    run2d(pso_scores(), N, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQ offset:oQ atIndex:0];
        [enc setBuffer:bK offset:oK atIndex:1];
        [enc setBuffer:bS offset:oS atIndex:2];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&inv_sqrtd length:sizeof(float) atIndex:5];
    });

    // Row-masked softmax.
    run_rows(pso_rsm(), N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS offset:oS atIndex:0];
        [enc setBuffer:bA offset:oA atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
    });

    // Y_pre_Wo = Attn @ V.
    run2d(pso_av(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA offset:oA atIndex:0];
        [enc setBuffer:bV offset:oV atIndex:1];
        [enc setBuffer:bY offset:oY atIndex:2];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });

    // O = Y @ Wo^T (with row-mask zeroing).
    run2d(pso_op(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bY offset:oY atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bO offset:oO atIndex:4];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });
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
    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(D));
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t has_mask = d_mask ? 1u : 0u;

    id<MTLBuffer> bdO = buffer_for(dO);
    NSUInteger odO = buffer_offset_for(dO);
    id<MTLBuffer> bX = buffer_for(X);
    NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bQ = buffer_for(Q);
    NSUInteger oQ = buffer_offset_for(Q);
    id<MTLBuffer> bK = buffer_for(K);
    NSUInteger oK = buffer_offset_for(K);
    id<MTLBuffer> bV = buffer_for(V);
    NSUInteger oV = buffer_offset_for(V);
    id<MTLBuffer> bA = buffer_for(Attn);
    NSUInteger oA = buffer_offset_for(Attn);
    id<MTLBuffer> bY = buffer_for(Y_pre_Wo);
    NSUInteger oY = buffer_offset_for(Y_pre_Wo);
    id<MTLBuffer> bWq = buffer_for(Wq);
    NSUInteger oWq = buffer_offset_for(Wq);
    id<MTLBuffer> bWk = buffer_for(Wk);
    NSUInteger oWk = buffer_offset_for(Wk);
    id<MTLBuffer> bWv = buffer_for(Wv);
    NSUInteger oWv = buffer_offset_for(Wv);
    id<MTLBuffer> bWo = buffer_for(Wo);
    NSUInteger oWo = buffer_offset_for(Wo);
    id<MTLBuffer> bdX = buffer_for(dX);
    NSUInteger odX = buffer_offset_for(dX);
    id<MTLBuffer> bdWq = buffer_for(dWq);
    NSUInteger odWq = buffer_offset_for(dWq);
    id<MTLBuffer> bdWk = buffer_for(dWk);
    NSUInteger odWk = buffer_offset_for(dWk);
    id<MTLBuffer> bdWv = buffer_for(dWv);
    NSUInteger odWv = buffer_offset_for(dWv);
    id<MTLBuffer> bdWo = buffer_for(dWo);
    NSUInteger odWo = buffer_offset_for(dWo);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? metal_impl::pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : buffer_offset_for(X);

    GpuTensor dY(N, D);
    id<MTLBuffer> bdY = buffer_for(dY);
    NSUInteger odY = buffer_offset_for(dY);
    run2d(pso_wodY(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bdY offset:odY atIndex:4];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });
    run2d(pso_wodW(), D, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bdWo offset:odWo atIndex:4];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });

    GpuTensor dAttn(N, N);
    GpuTensor dV(N, D);
    id<MTLBuffer> bdA = buffer_for(dAttn);
    NSUInteger odA = buffer_offset_for(dAttn);
    id<MTLBuffer> bdV = buffer_for(dV);
    NSUInteger odV = buffer_offset_for(dV);
    run2d(pso_dAttn(), N, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdY offset:odY atIndex:0];
        [enc setBuffer:bV offset:oV atIndex:1];
        [enc setBuffer:bdA offset:odA atIndex:2];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
    run2d(pso_dV(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA offset:oA atIndex:0];
        [enc setBuffer:bdY offset:odY atIndex:1];
        [enc setBuffer:bdV offset:odV atIndex:2];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });

    GpuTensor dScores(N, N);
    id<MTLBuffer> bdS = buffer_for(dScores);
    NSUInteger odS = buffer_offset_for(dScores);
    run_rows(pso_rsmb(), N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA offset:oA atIndex:0];
        [enc setBuffer:bdA offset:odA atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bdS offset:odS atIndex:4];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&inv_sqrtd length:sizeof(float) atIndex:6];
    });

    GpuTensor dQ(N, D), dK(N, D);
    id<MTLBuffer> bdQ = buffer_for(dQ);
    NSUInteger odQ = buffer_offset_for(dQ);
    id<MTLBuffer> bdK = buffer_for(dK);
    NSUInteger odK = buffer_offset_for(dK);
    run2d(pso_dQ(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdS offset:odS atIndex:0];
        [enc setBuffer:bK offset:oK atIndex:1];
        [enc setBuffer:bdQ offset:odQ atIndex:2];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
    run2d(pso_dK(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdS offset:odS atIndex:0];
        [enc setBuffer:bQ offset:oQ atIndex:1];
        [enc setBuffer:bdK offset:odK atIndex:2];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });

    run2d(pso_dWqkv(), D, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdQ offset:odQ atIndex:0];
        [enc setBuffer:bdK offset:odK atIndex:1];
        [enc setBuffer:bdV offset:odV atIndex:2];
        [enc setBuffer:bX offset:oX atIndex:3];
        [enc setBuffer:bdWq offset:odWq atIndex:4];
        [enc setBuffer:bdWk offset:odWk atIndex:5];
        [enc setBuffer:bdWv offset:odWv atIndex:6];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:8];
    });
    run2d(pso_dXp(), D, N, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdQ offset:odQ atIndex:0];
        [enc setBuffer:bdK offset:odK atIndex:1];
        [enc setBuffer:bdV offset:odV atIndex:2];
        [enc setBuffer:bWq offset:oWq atIndex:3];
        [enc setBuffer:bWk offset:oWk atIndex:4];
        [enc setBuffer:bWv offset:oWv atIndex:5];
        [enc setBuffer:bdX offset:odX atIndex:6];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:8];
    });
}

} // namespace brogameagent::nn::gpu
