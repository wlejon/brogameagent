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

// Per-head Q/K/V projection.
kernel void k_mha_proj(device const float* X   [[buffer(0)]],
                       device const float* W   [[buffer(1)]],
                       device float*       Out [[buffer(2)]],
                       constant uint& K        [[buffer(3)]],
                       constant uint& D        [[buffer(4)]],
                       constant uint& dh       [[buffer(5)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= K || j >= dh) return;
    uint row_off = hh * dh;
    device const float* xr = X + i * D;
    device const float* wr = W + (row_off + j) * D;
    float acc = 0.0f;
    for (uint k = 0; k < D; ++k) acc += xr[k] * wr[k];
    uint out_row = hh * K + i;
    Out[out_row * dh + j] = acc;
}

kernel void k_mha_scores(device const float* Qh [[buffer(0)]],
                         device const float* Kh [[buffer(1)]],
                         device float*       S  [[buffer(2)]],
                         constant uint& K       [[buffer(3)]],
                         constant uint& dh      [[buffer(4)]],
                         constant float& inv_sqrtdh [[buffer(5)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= K || j >= K) return;
    uint qrow = (hh * K + i) * dh;
    uint krow = (hh * K + j) * dh;
    float s = 0.0f;
    for (uint k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    uint srow = (hh * K + i) * K;
    S[srow + j] = s * inv_sqrtdh;
}

kernel void k_mha_row_softmax(device const float* scores [[buffer(0)]],
                              device float*       Attn   [[buffer(1)]],
                              device const float* mask   [[buffer(2)]],
                              constant uint& has_mask    [[buffer(3)]],
                              constant uint& K           [[buffer(4)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    uint i_within = row % K;
    device const float* srow = scores + row * K;
    device float*       arow = Attn + row * K;
    if (has_mask && mask[i_within] == 0.0f) {
        for (uint j = tid; j < K; j += tg_size) arow[j] = 0.0f;
        return;
    }
    float local_max = -1e30f;
    for (uint j = tid; j < K; j += tg_size) {
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
    for (uint j = tid; j < K; j += tg_size) {
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
    for (uint j = tid; j < K; j += tg_size) {
        arow[j] = arow[j] * inv;
    }
}

kernel void k_mha_attn_apply_v(device const float* Attnh   [[buffer(0)]],
                               device const float* Vh      [[buffer(1)]],
                               device float*       Yconcat [[buffer(2)]],
                               constant uint& K            [[buffer(3)]],
                               constant uint& dh           [[buffer(4)]],
                               constant uint& D            [[buffer(5)]],
                               uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= K || k >= dh) return;
    uint arow = (hh * K + i) * K;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        uint vrow = (hh * K + j) * dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[i * D + (hh * dh + k)] = acc;
}

kernel void k_mha_output_proj(device const float* Y    [[buffer(0)]],
                              device const float* Wo   [[buffer(1)]],
                              device const float* mask [[buffer(2)]],
                              constant uint& has_mask  [[buffer(3)]],
                              device float*       O    [[buffer(4)]],
                              constant uint& K         [[buffer(5)]],
                              constant uint& D         [[buffer(6)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x; uint i = gid.y;
    if (i >= K || c >= D) return;
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

kernel void k_mha_wo_back_dW(device const float* dO   [[buffer(0)]],
                             device const float* Y    [[buffer(1)]],
                             device const float* mask [[buffer(2)]],
                             constant uint& has_mask  [[buffer(3)]],
                             device float*       dWo  [[buffer(4)]],
                             constant uint& K         [[buffer(5)]],
                             constant uint& D         [[buffer(6)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint c = gid.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < K; ++i) {
        if (has_mask && mask[i] == 0.0f) continue;
        acc += dO[i * D + c] * Y[i * D + k];
    }
    dWo[c * D + k] += acc;
}

kernel void k_mha_wo_back_dY(device const float* dO  [[buffer(0)]],
                             device const float* Wo  [[buffer(1)]],
                             device const float* mask[[buffer(2)]],
                             constant uint& has_mask [[buffer(3)]],
                             device float*       dY  [[buffer(4)]],
                             constant uint& K        [[buffer(5)]],
                             constant uint& D        [[buffer(6)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= K || k >= D) return;
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

kernel void k_mha_dAttn(device const float* dYconcat [[buffer(0)]],
                        device const float* Vh       [[buffer(1)]],
                        device float*       dAttn    [[buffer(2)]],
                        constant uint& K             [[buffer(3)]],
                        constant uint& dh            [[buffer(4)]],
                        constant uint& D             [[buffer(5)]],
                        uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= K || j >= K) return;
    float acc = 0.0f;
    for (uint k = 0; k < dh; ++k) {
        float dy = dYconcat[i * D + (hh * dh + k)];
        float vv = Vh[(hh * K + j) * dh + k];
        acc += dy * vv;
    }
    dAttn[(hh * K + i) * K + j] = acc;
}

kernel void k_mha_dV(device const float* Attnh    [[buffer(0)]],
                     device const float* dYconcat [[buffer(1)]],
                     device float*       dVh      [[buffer(2)]],
                     constant uint& K             [[buffer(3)]],
                     constant uint& dh            [[buffer(4)]],
                     constant uint& D             [[buffer(5)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y; uint hh = gid.z;
    if (j >= K || k >= dh) return;
    float acc = 0.0f;
    for (uint i = 0; i < K; ++i) {
        float a  = Attnh[(hh * K + i) * K + j];
        float dy = dYconcat[i * D + (hh * dh + k)];
        acc += a * dy;
    }
    dVh[(hh * K + j) * dh + k] = acc;
}

kernel void k_mha_row_softmax_back(device const float* Attn    [[buffer(0)]],
                                   device const float* dAttn   [[buffer(1)]],
                                   device const float* mask    [[buffer(2)]],
                                   constant uint& has_mask     [[buffer(3)]],
                                   device float*       dScores [[buffer(4)]],
                                   constant uint& K            [[buffer(5)]],
                                   constant float& inv_sqrtdh  [[buffer(6)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    uint i_within = row % K;
    device const float* prow  = Attn  + row * K;
    device const float* dprow = dAttn + row * K;
    device float*       drow  = dScores + row * K;
    if (has_mask && mask[i_within] == 0.0f) {
        for (uint j = tid; j < K; j += tg_size) drow[j] = 0.0f;
        return;
    }
    float local = 0.0f;
    for (uint j = tid; j < K; j += tg_size) {
        local += dprow[j] * prow[j];
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = sdata[0];
    for (uint j = tid; j < K; j += tg_size) {
        if (has_mask && mask[j] == 0.0f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
        }
    }
}

kernel void k_mha_dQ(device const float* dScores [[buffer(0)]],
                     device const float* Kh      [[buffer(1)]],
                     device float*       dQh     [[buffer(2)]],
                     constant uint& K            [[buffer(3)]],
                     constant uint& dh           [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= K || k >= dh) return;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        float ds = dScores[(hh * K + i) * K + j];
        float kk = Kh[(hh * K + j) * dh + k];
        acc += ds * kk;
    }
    dQh[(hh * K + i) * dh + k] = acc;
}

kernel void k_mha_dK(device const float* dScores [[buffer(0)]],
                     device const float* Qh      [[buffer(1)]],
                     device float*       dKh     [[buffer(2)]],
                     constant uint& K            [[buffer(3)]],
                     constant uint& dh           [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y; uint hh = gid.z;
    if (j >= K || k >= dh) return;
    float acc = 0.0f;
    for (uint i = 0; i < K; ++i) {
        float ds = dScores[(hh * K + i) * K + j];
        float qq = Qh[(hh * K + i) * dh + k];
        acc += ds * qq;
    }
    dKh[(hh * K + j) * dh + k] = acc;
}

kernel void k_mha_dWqkv(device const float* dQh [[buffer(0)]],
                        device const float* dKh [[buffer(1)]],
                        device const float* dVh [[buffer(2)]],
                        device const float* X   [[buffer(3)]],
                        device float*       dWq [[buffer(4)]],
                        device float*       dWk [[buffer(5)]],
                        device float*       dWv [[buffer(6)]],
                        constant uint& K        [[buffer(7)]],
                        constant uint& D        [[buffer(8)]],
                        constant uint& dh       [[buffer(9)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint k_col = gid.x; uint wrow = gid.y;
    if (wrow >= D || k_col >= D) return;
    uint hh = wrow / dh;
    uint j = wrow % dh;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (uint i = 0; i < K; ++i) {
        float xv = X[i * D + k_col];
        aq += dQh[(hh * K + i) * dh + j] * xv;
        ak += dKh[(hh * K + i) * dh + j] * xv;
        av += dVh[(hh * K + i) * dh + j] * xv;
    }
    uint idx = wrow * D + k_col;
    dWq[idx] += aq;
    dWk[idx] += ak;
    dWv[idx] += av;
}

kernel void k_mha_dX_proj(device const float* dQh [[buffer(0)]],
                          device const float* dKh [[buffer(1)]],
                          device const float* dVh [[buffer(2)]],
                          device const float* Wq  [[buffer(3)]],
                          device const float* Wk  [[buffer(4)]],
                          device const float* Wv  [[buffer(5)]],
                          device float*       dX  [[buffer(6)]],
                          constant uint& K        [[buffer(7)]],
                          constant uint& D        [[buffer(8)]],
                          constant uint& dh       [[buffer(9)]],
                          constant uint& H        [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= K || k >= D) return;
    float acc = 0.0f;
    for (uint hh = 0; hh < H; ++hh) {
        for (uint j = 0; j < dh; ++j) {
            uint wrow = hh * dh + j;
            uint widx = wrow * D + k;
            float gq = dQh[(hh * K + i) * dh + j];
            float gk = dKh[(hh * K + i) * dh + j];
            float gv = dVh[(hh * K + i) * dh + j];
            acc += gq * Wq[widx] + gk * Wk[widx] + gv * Wv[widx];
        }
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
DEF_PSO(pso_proj, @"k_mha_proj")
DEF_PSO(pso_scores, @"k_mha_scores")
DEF_PSO(pso_rsm, @"k_mha_row_softmax")
DEF_PSO(pso_av, @"k_mha_attn_apply_v")
DEF_PSO(pso_op, @"k_mha_output_proj")
DEF_PSO(pso_wodW, @"k_mha_wo_back_dW")
DEF_PSO(pso_wodY, @"k_mha_wo_back_dY")
DEF_PSO(pso_dAttn, @"k_mha_dAttn")
DEF_PSO(pso_dV, @"k_mha_dV")
DEF_PSO(pso_rsmb, @"k_mha_row_softmax_back")
DEF_PSO(pso_dQ, @"k_mha_dQ")
DEF_PSO(pso_dK, @"k_mha_dK")
DEF_PSO(pso_dWqkv, @"k_mha_dWqkv")
DEF_PSO(pso_dXp, @"k_mha_dX_proj")
#undef DEF_PSO

void run3d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny, NSUInteger nz,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger w = 8, h = 8;
        if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
        [enc dispatchThreads:MTLSizeMake(nx, ny, nz)
        threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void run2d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger w = 16, h = 16;
        if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1)
        threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
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
    const int dh = (H > 0) ? D / H : 0;
    if (Qh.rows != H * K || Qh.cols != dh) Qh.resize(H * K, dh);
    if (Kh.rows != H * K || Kh.cols != dh) Kh.resize(H * K, dh);
    if (Vh.rows != H * K || Vh.cols != dh) Vh.resize(H * K, dh);
    if (Attnh.rows != H * K || Attnh.cols != K) Attnh.resize(H * K, K);
    if (Yconcat.rows != K || Yconcat.cols != D) Yconcat.resize(K, D);
    if (O.rows != K || O.cols != D) O.resize(K, D);
    if (K == 0 || D == 0 || H == 0) return;

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
    id<MTLBuffer> bQh = buffer_for(Qh);
    NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh);
    NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh);
    NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bAh = buffer_for(Attnh);
    NSUInteger oAh = buffer_offset_for(Attnh);
    id<MTLBuffer> bYc = buffer_for(Yconcat);
    NSUInteger oYc = buffer_offset_for(Yconcat);
    id<MTLBuffer> bO  = buffer_for(O);
    NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bM  = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? metal_impl::pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;

    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhU = static_cast<uint32_t>(dh);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    auto proj = ^(id<MTLBuffer> bW, NSUInteger oWp, id<MTLBuffer> bOut, NSUInteger oOut) {
        run3d(pso_proj(), dh, K, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX offset:oX atIndex:0];
            [enc setBuffer:bW offset:oWp atIndex:1];
            [enc setBuffer:bOut offset:oOut atIndex:2];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        });
    };
    proj(bWq, oWq, bQh, oQh);
    proj(bWk, oWk, bKh, oKh);
    proj(bWv, oWv, bVh, oVh);

    GpuTensor scores(H * K, K);
    id<MTLBuffer> bS = buffer_for(scores);
    NSUInteger oS = buffer_offset_for(scores);
    run3d(pso_scores(), K, K, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh offset:oQh atIndex:0];
        [enc setBuffer:bKh offset:oKh atIndex:1];
        [enc setBuffer:bS offset:oS atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&inv_sqrtdh length:sizeof(float) atIndex:5];
    });

    run_rows(pso_rsm(), H * K, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS offset:oS atIndex:0];
        [enc setBuffer:bAh offset:oAh atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:4];
    });

    run3d(pso_av(), dh, K, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bAh offset:oAh atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
    });

    run2d(pso_op(), D, K, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc offset:oYc atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bO offset:oO atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });
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
    const int dh = (H > 0) ? D / H : 0;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D);
    if (K == 0 || D == 0 || H == 0) return;

    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhU = static_cast<uint32_t>(dh);
    const uint32_t Hu = static_cast<uint32_t>(H);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    id<MTLBuffer> bdO = buffer_for(dO);
    NSUInteger odO = buffer_offset_for(dO);
    id<MTLBuffer> bX = buffer_for(X);
    NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bQh = buffer_for(Qh);
    NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh);
    NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh);
    NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bAh = buffer_for(Attnh);
    NSUInteger oAh = buffer_offset_for(Attnh);
    id<MTLBuffer> bYc = buffer_for(Yconcat);
    NSUInteger oYc = buffer_offset_for(Yconcat);
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

    GpuTensor dYconcat(K, D);
    id<MTLBuffer> bdY = buffer_for(dYconcat);
    NSUInteger odY = buffer_offset_for(dYconcat);
    run2d(pso_wodY(), D, K, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bdY offset:odY atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });
    run2d(pso_wodW(), D, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bYc offset:oYc atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bdWo offset:odWo atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });

    GpuTensor dAttn(H * K, K);
    GpuTensor dVh(H * K, dh);
    id<MTLBuffer> bdA = buffer_for(dAttn);
    NSUInteger odA = buffer_offset_for(dAttn);
    id<MTLBuffer> bdVh = buffer_for(dVh);
    NSUInteger odVh = buffer_offset_for(dVh);
    run3d(pso_dAttn(), K, K, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdY offset:odY atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bdA offset:odA atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
    });
    run3d(pso_dV(), dh, K, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bAh offset:oAh atIndex:0];
        [enc setBuffer:bdY offset:odY atIndex:1];
        [enc setBuffer:bdVh offset:odVh atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
    });

    GpuTensor dScores(H * K, K);
    id<MTLBuffer> bdS = buffer_for(dScores);
    NSUInteger odS = buffer_offset_for(dScores);
    run_rows(pso_rsmb(), H * K, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bAh offset:oAh atIndex:0];
        [enc setBuffer:bdA offset:odA atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bdS offset:odS atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&inv_sqrtdh length:sizeof(float) atIndex:6];
    });

    GpuTensor dQh(H * K, dh), dKh(H * K, dh);
    id<MTLBuffer> bdQh = buffer_for(dQh);
    NSUInteger odQh = buffer_offset_for(dQh);
    id<MTLBuffer> bdKh = buffer_for(dKh);
    NSUInteger odKh = buffer_offset_for(dKh);
    run3d(pso_dQ(), dh, K, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdS offset:odS atIndex:0];
        [enc setBuffer:bKh offset:oKh atIndex:1];
        [enc setBuffer:bdQh offset:odQh atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:4];
    });
    run3d(pso_dK(), dh, K, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdS offset:odS atIndex:0];
        [enc setBuffer:bQh offset:oQh atIndex:1];
        [enc setBuffer:bdKh offset:odKh atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:4];
    });

    run2d(pso_dWqkv(), D, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdQh offset:odQh atIndex:0];
        [enc setBuffer:bdKh offset:odKh atIndex:1];
        [enc setBuffer:bdVh offset:odVh atIndex:2];
        [enc setBuffer:bX offset:oX atIndex:3];
        [enc setBuffer:bdWq offset:odWq atIndex:4];
        [enc setBuffer:bdWk offset:odWk atIndex:5];
        [enc setBuffer:bdWv offset:odWv atIndex:6];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:9];
    });
    run2d(pso_dXp(), D, K, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdQh offset:odQh atIndex:0];
        [enc setBuffer:bdKh offset:odKh atIndex:1];
        [enc setBuffer:bdVh offset:odVh atIndex:2];
        [enc setBuffer:bWq offset:oWq atIndex:3];
        [enc setBuffer:bWk offset:oWk atIndex:4];
        [enc setBuffer:bWv offset:oWv atIndex:5];
        [enc setBuffer:bdX offset:odX atIndex:6];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:10];
    });
}

} // namespace brogameagent::nn::gpu
