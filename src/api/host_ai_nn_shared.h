#pragma once

// Shared plumbing for the bro.ai.game.nn / .learn / .grid bindings.
//
// The old surface had its own AITensor class. It is gone: a tensor argument
// is now a bro.tensor GpuTensor handle (brotensor's own binding) or a plain
// Float32Array, which this header turns into a non-owning CPU view. Nothing
// here allocates a second tensor type.

#include "host_ai_mcts_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <brogameagent/learn/generic_replay_buffer.h>
#include <brogameagent/learn/generic_trainer.h>
#include <brogameagent/learn/inference_backend.h>
#include <brogameagent/learn/inference_server.h>
#include <brogameagent/learn/neural_adapters.h>
#include <brogameagent/learn/replay_buffer.h>
#include <brogameagent/learn/trainer.h>
#include <brogameagent/nn/circuits.h>
#include <brogameagent/nn/encoder.h>
#include <brogameagent/nn/factored.h>
#include <brogameagent/nn/heads.h>
#include <brogameagent/nn/net.h>
#include <brogameagent/nn/net_tx.h>
#include <brogameagent/nn/policy_value_net.h>

#include <array>
#include <memory>
#include <vector>

namespace brogameagent::api {

namespace nn = brogameagent::nn;

inline constexpr uint32_t kHostLinearTag        = 0x4C494E52u;  // 'LINR'
inline constexpr uint32_t kHostReluTag          = 0x52454C55u;  // 'RELU'
inline constexpr uint32_t kHostTanhTag          = 0x54414E48u;  // 'TANH'
inline constexpr uint32_t kHostDeepSetsTag      = 0x44535453u;  // 'DSTS'
inline constexpr uint32_t kHostValueHeadTag     = 0x56484541u;  // 'VHEA'
inline constexpr uint32_t kHostFactoredHeadTag  = 0x46504844u;  // 'FPHD'
inline constexpr uint32_t kHostSingleHeroNetTag = 0x53484E54u;  // 'SHNT'
inline constexpr uint32_t kHostPolicyValueNetTag= 0x50564E54u;  // 'PVNT'
inline constexpr uint32_t kHostHeroNetTxTag     = 0x53485458u;  // 'SHTX'
inline constexpr uint32_t kHostWeightsHandleTag = 0x57484E44u;  // 'WHND'

struct HostLinear { uint32_t tag = kHostLinearTag; nn::Linear l; };
struct HostRelu { uint32_t tag = kHostReluTag; nn::Relu r; };
struct HostTanh { uint32_t tag = kHostTanhTag; nn::Tanh t; };
struct HostDeepSets { uint32_t tag = kHostDeepSetsTag; nn::DeepSetsEncoder e; };
struct HostValueHead { uint32_t tag = kHostValueHeadTag; nn::ValueHead v; };
struct HostFactoredHead { uint32_t tag = kHostFactoredHeadTag; nn::FactoredPolicyHead h; };
struct HostSingleHeroNet { uint32_t tag = kHostSingleHeroNetTag; std::shared_ptr<nn::SingleHeroNet> net; };
struct HostPolicyValueNet { uint32_t tag = kHostPolicyValueNetTag; std::shared_ptr<nn::PolicyValueNet> net; };
struct HostHeroNetTx { uint32_t tag = kHostHeroNetTxTag; std::shared_ptr<nn::SingleHeroNetTX> net; };
struct HostWeightsHandle { uint32_t tag = kHostWeightsHandleTag; std::shared_ptr<nn::WeightsHandle> handle; };

template <typename T, uint32_t Tag>
inline T* unwrapTagged(Value v) {
    if (!ev::isObject(v)) return nullptr;
    auto* h = static_cast<T*>(ev::handleData(v));
    return (h && h->tag == Tag) ? h : nullptr;
}

inline HostLinear* unwrapLinear(Value v) { return unwrapTagged<HostLinear, kHostLinearTag>(v); }
inline HostRelu* unwrapRelu(Value v) { return unwrapTagged<HostRelu, kHostReluTag>(v); }
inline HostTanh* unwrapTanh(Value v) { return unwrapTagged<HostTanh, kHostTanhTag>(v); }
inline HostDeepSets* unwrapDeepSets(Value v) { return unwrapTagged<HostDeepSets, kHostDeepSetsTag>(v); }
inline HostValueHead* unwrapValueHead(Value v) { return unwrapTagged<HostValueHead, kHostValueHeadTag>(v); }
inline HostFactoredHead* unwrapFactoredHead(Value v) { return unwrapTagged<HostFactoredHead, kHostFactoredHeadTag>(v); }
inline HostSingleHeroNet* unwrapSingleHeroNet(Value v) { return unwrapTagged<HostSingleHeroNet, kHostSingleHeroNetTag>(v); }
inline HostPolicyValueNet* unwrapPolicyValueNet(Value v) { return unwrapTagged<HostPolicyValueNet, kHostPolicyValueNetTag>(v); }
inline HostHeroNetTx* unwrapHeroNetTx(Value v) { return unwrapTagged<HostHeroNetTx, kHostHeroNetTxTag>(v); }
inline HostWeightsHandle* unwrapWeightsHandle(Value v) { return unwrapTagged<HostWeightsHandle, kHostWeightsHandleTag>(v); }

extern HostClass g_linearClass;
extern HostClass g_reluClass;
extern HostClass g_tanhClass;
extern HostClass g_deepSetsClass;
extern HostClass g_valueHeadClass;
extern HostClass g_factoredHeadClass;
extern HostClass g_singleHeroNetClass;
extern HostClass g_policyValueNetClass;
extern HostClass g_heroNetTxClass;
extern HostClass g_weightsHandleClass;

inline std::shared_ptr<nn::SingleHeroNet> singleHeroNetShared(Value v) {
    auto* h = unwrapSingleHeroNet(v);
    return h ? h->net : std::shared_ptr<nn::SingleHeroNet>{};
}
inline std::shared_ptr<nn::PolicyValueNet> policyValueNetShared(Value v) {
    auto* h = unwrapPolicyValueNet(v);
    return h ? h->net : std::shared_ptr<nn::PolicyValueNet>{};
}
inline std::shared_ptr<nn::SingleHeroNetTX> heroNetTxShared(Value v) {
    auto* h = unwrapHeroNetTx(v);
    return h ? h->net : std::shared_ptr<nn::SingleHeroNetTX>{};
}
inline std::shared_ptr<nn::WeightsHandle> weightsHandleShared(Value v) {
    auto* h = unwrapWeightsHandle(v);
    return h ? h->handle : std::shared_ptr<nn::WeightsHandle>{};
}

/// A tensor argument: a bro.tensor GpuTensor handle, or a Float32Array the
/// helper views in place (rank-1, `n x 1`). Holds the view so the pointer
/// stays valid for the duration of the call.
///
/// Copying re-points `ptr` at the copy's own view: a TensorArg is returned by
/// value, and without NRVO (MSVC Debug) a defaulted copy would leave `ptr`
/// aimed at the dead temporary's view.
///
/// A view's data points INTO THE MOVING BRONZE HEAP (embed.h's typed-array
/// pointer contract): it is valid only until the next allocating embed call.
/// Resolve every tensor argument of a call with tensorArgs(), which does the
/// allocating GpuTensor checks first and takes the in-place views last, and
/// take nothing else that allocates before the op runs.
struct TensorArg {
    brotensor::Tensor* ptr = nullptr;
    brotensor::Tensor view;

    TensorArg() = default;
    // A copy must alias the same caller memory. brotensor::Tensor's own copy
    // is a deep clone, so the view is rebuilt from the source view's fields
    // rather than copied, and `ptr` re-pointed at this object's view (the
    // source's would dangle once it is destroyed).
    TensorArg(const TensorArg& o) { *this = o; }
    TensorArg& operator=(const TensorArg& o) {
        if (this == &o) return *this;
        if (o.ptr == &o.view) {
            view = brotensor::Tensor::view(o.view.device, o.view.data, o.view.rows,
                                           o.view.cols, o.view.dtype);
            ptr = &view;
        } else {
            ptr = o.ptr;
        }
        return *this;
    }

    explicit operator bool() const { return ptr != nullptr; }
    brotensor::Tensor& operator*() const { return *ptr; }
};

/// True for an instance of bro.tensor's native GpuTensor class. Checked by
/// prototype rather than by a tag, because brotensor's handle carries
/// brotensor's tag, not one of ours — and `getTensorFromHandle` casts blind.
/// ALLOCATES (it calls Object.getPrototypeOf).
bool isGpuTensorValue(Value v);

/// One tensor argument. Allocates when `v` is not a typed array, so a
/// Float32Array view taken by an EARLIER tensorArg in the same call may be
/// stale afterwards — prefer tensorArgs() whenever a call takes more than one.
TensorArg tensorArg(Value v);

/// Every tensor argument of a call, resolved so no view goes stale: the
/// GpuTensor checks (which allocate) run first over the rooted `args` slots,
/// then the Float32Array views are taken with no allocation in between.
/// `idx` names the argument positions; a position past the end is an empty
/// TensorArg.
template <size_t N>
std::array<TensorArg, N> tensorArgs(std::span<const Value> args, const std::array<size_t, N>& idx) {
    std::array<TensorArg, N> out{};
    for (size_t k = 0; k < N; ++k) {
        const size_t i = idx[k];
        if (i >= args.size() || ev::isTypedArray(args[i])) continue;
        // args[i] is re-read from its rooted slot each time, so it is current
        // across the previous iteration's allocation.
        out[k] = tensorArg(args[i]);
    }
    for (size_t k = 0; k < N; ++k) {
        const size_t i = idx[k];
        if (i >= args.size() || !ev::isTypedArray(args[i])) continue;
        out[k] = tensorArg(args[i]);  // a typed array: a view, no allocation
    }
    return out;
}

/// Read a TypedArray's bytes. Returns {nullptr, 0} when `v` is not one.
struct RawBytes {
    uint8_t* data = nullptr;
    size_t size = 0;
};
RawBytes rawBytes(Value v);

/// A Float32Array's elements in place (mask arguments), or nullptr.
float* floatPtr(Value v, size_t& count);

Value makeUint8Array(const uint8_t* data, size_t count);

/// The old bindings answered seeds and weight versions as BigInt. bronze's
/// embed API has no BigInt constructor, so this goes through the global
/// `BigInt(<decimal string>)`, which is exact for the full uint64 range.
Value makeBigIntValue(uint64_t v);
/// BigInt or Number in, uint64 out; `def` for undefined/null.
uint64_t readSeedArg(Value v, uint64_t def);

Value makeIntArrayValue(const std::vector<int>& v);
/// An array-like of int32s. `checked` (arguments and options): a NaN or
/// out-of-range element throws a RangeError. Unchecked (a callback's result
/// read mid-search): such an element reads as -1.
std::vector<int> readIntArrayValue(Value arr, bool checked = true);
std::vector<int> readIntArrayProp(Value obj, const char* key);
/// Factored-action head sizes: RangeError unless every head is in
/// [1, 2^24], there are at most 64, and their product (the flat action
/// count) fits int32 — a zero head divides by zero in decode, and an
/// overflowing product sizes the flat buffers wrong.
void checkHeadSizes(const std::vector<int>& sizes, const char* what);
std::vector<int> readHeadSizes(Value arr, const char* what);
/// An int32 option (RangeError for NaN / out of range); `def` when absent.
int getIntProp(Value obj, const char* key, int def);
/// The same, for a size or count: also RangeError below 0.
int getCountProp(Value obj, const char* key, int def);

/// "gpu" → brotensor's default device, throwing when the build has none;
/// anything else → CPU. GPU-first: there is no silent CPU fallback.
bool resolveDevice(const std::string& name, brotensor::Device& out, std::string& error);

void ensureAINnCircuitClassesInstalled();
void ensureAINnNetClassesInstalled();
void installAINnCircuits(ObjectBuilder& nnNs);
void installAINnNets(ObjectBuilder& nnNs);
void installAINnOps(ObjectBuilder& nnNs);

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
