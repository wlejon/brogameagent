// bro.ai.game.nn — the free ops: the brotensor forward/backward primitives
// the old binding exposed one-for-one, the factored-policy helpers, and the
// policy-head shape constants.
//
// Tensor arguments follow the same rule as the classes: a bro.tensor
// GpuTensor handle or a Float32Array viewed in place. A mask argument may be
// either too; it is brought to the device its op reads it on (see maskArg).

#include "host_ai_nn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <exception>

namespace brogameagent::api {

namespace {

template <typename Fn>
Value guardedOp(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return ev::undefined();
}

// A resolved optional mask: `ptr` is null for no mask, else `need` FP32
// entries on the device the op reads the mask on. That is the caller's own
// buffer when it already lives there, else `staged`, a copy moved there.
struct Mask {
    const float* ptr = nullptr;
    brotensor::Tensor staged;
};

// An optional mask argument, `m` being its slot resolved by tensorArgs():
// absent (undefined / null) is no mask; anything else must be a Float32Array
// or a GpuTensor of at least `need` FP32 elements, because the kernels read
// `need` floats from it unchecked. `dev` is where the op reads it — brotensor
// takes a mask as a raw pointer on its operands' device, so a Float32Array
// feeding a GPU op is uploaded (a GPU kernel cannot read host memory) and a
// GpuTensor feeding a host op is downloaded. Staging allocates only
// brotensor memory, never on the JS heap, so the Float32Array views taken by
// tensorArgs() stay valid.
bool maskArg(std::span<const Value> a, size_t i, const TensorArg& m, size_t need,
             brotensor::Device dev, const char* what, Mask& out) {
    out.ptr = nullptr;
    if (i >= a.size() || ev::isUndefined(a[i]) || ev::isNull(a[i])) return true;
    if (!m || m.ptr->dtype != brotensor::Dtype::FP32 ||
        static_cast<size_t>(m.ptr->size()) < need) {
        ev::throwTypeError(std::string(what) + " must be a Float32Array or GpuTensor of at least " +
                           std::to_string(need) + " FP32 entries");
        return false;
    }
    if (m.ptr->device == dev) {
        out.ptr = static_cast<const float*>(m.ptr->data);
        return true;
    }
    try {
        out.staged = m.ptr->to(dev);
    } catch (const std::exception& e) {
        ev::throwError(std::string(what) + ": " + e.what());
        return false;
    }
    out.ptr = static_cast<const float*>(out.staged.data);
    return true;
}

std::vector<int> offsetsFromSizes(const std::vector<int>& sizes) {
    std::vector<int> offs;
    offs.reserve(sizes.size() + 1);
    int o = 0;
    for (int s : sizes) {
        offs.push_back(o);
        o += s;
    }
    offs.push_back(o);
    return offs;
}

} // namespace

void installAINnOps(ObjectBuilder& nnNs) {
    nnNs.def("linearForward", 4, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 4) return ev::throwTypeError("linearForward(W,b,x,y)");
        auto [W, b, x, y] = tensorArgs<4>(a, {0, 1, 2, 3});
        if (!W || !b || !x || !y) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::linear_forward(*W.ptr, *b.ptr, *x.ptr, *y.ptr); });
    });

    nnNs.def("linearBackward", 6, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 6) return ev::throwTypeError("linearBackward(W,x,dY,dX,dW,dB)");
        auto [W, x, dY, dX, dW, dB] = tensorArgs<6>(a, {0, 1, 2, 3, 4, 5});
        if (!W || !x || !dY || !dX || !dW || !dB) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] {
            brotensor::linear_backward(*W.ptr, *x.ptr, *dY.ptr, *dX.ptr, *dW.ptr, *dB.ptr);
        });
    });

    nnNs.def("reluForward", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("reluForward(x,y)");
        auto [x, y] = tensorArgs<2>(a, {0, 1});
        if (!x || !y) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::relu_forward(*x.ptr, *y.ptr); });
    });

    nnNs.def("reluBackward", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::throwTypeError("reluBackward(x,dY,dX)");
        auto [x, dY, dX] = tensorArgs<3>(a, {0, 1, 2});
        if (!x || !dY || !dX) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::relu_backward(*x.ptr, *dY.ptr, *dX.ptr); });
    });

    nnNs.def("tanhForward", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("tanhForward(x,y)");
        auto [x, y] = tensorArgs<2>(a, {0, 1});
        if (!x || !y) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::tanh_forward(*x.ptr, *y.ptr); });
    });

    nnNs.def("tanhBackward", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::throwTypeError("tanhBackward(y,dY,dX)");
        auto [y, dY, dX] = tensorArgs<3>(a, {0, 1, 2});
        if (!y || !dY || !dX) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::tanh_backward(*y.ptr, *dY.ptr, *dX.ptr); });
    });

    nnNs.def("softmaxForward", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("softmaxForward(logits,probs,mask?)");
        auto [l, p, m] = tensorArgs<3>(a, {0, 1, 2});
        if (!l || !p) return ev::throwTypeError("expected Tensors");
        Mask mask;
        if (!maskArg(a, 2, m, static_cast<size_t>(l.ptr->size()), l.ptr->device,
                     "softmaxForward: mask", mask))
            return ev::undefined();
        return guardedOp([&] { brotensor::softmax_forward(*l.ptr, *p.ptr, mask.ptr); });
    });

    nnNs.def("softmaxBackward", 3, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) return ev::throwTypeError("softmaxBackward(probs,dProbs,dLogits)");
        auto [p, dp, dl] = tensorArgs<3>(a, {0, 1, 2});
        if (!p || !dp || !dl) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::softmax_backward(*p.ptr, *dp.ptr, *dl.ptr); });
    });

    nnNs.def("softmaxXent", 5, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 4) return ev::throwTypeError("softmaxXent(logits,target,probs,dLogits,mask?)");
        auto [l, t, p, dl, m] = tensorArgs<5>(a, {0, 1, 2, 3, 4});
        if (!l || !t || !p || !dl) return ev::throwTypeError("expected Tensors");
        Mask mask;
        if (!maskArg(a, 4, m, static_cast<size_t>(l.ptr->size()), l.ptr->device,
                     "softmaxXent: mask", mask))
            return ev::undefined();
        float loss = 0.0f;
        try {
            loss = brotensor::softmax_xent(*l.ptr, *t.ptr, *p.ptr, *dl.ptr, mask.ptr);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return ev::fromDouble(loss);
    });

    nnNs.def("mseScalar", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("mseScalar(pred,target)");
        float dPred = 0.0f;
        float loss = brotensor::mse_scalar(static_cast<float>(numAt(a, 0)),
                                           static_cast<float>(numAt(a, 1)), dPred);
        ObjectBuilder o;
        o.set("loss", static_cast<double>(loss));
        o.set("dPred", static_cast<double>(dPred));
        return o.get();
    });

    nnNs.def("addInplace", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("addInplace(y,x)");
        auto [y, x] = tensorArgs<2>(a, {0, 1});
        if (!x || !y) return ev::throwTypeError("expected Tensors");
        return guardedOp([&] { brotensor::add_inplace(*y.ptr, *x.ptr); });
    });

    nnNs.def("addScalarInplace", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("addScalarInplace(y,s)");
        TensorArg y = tensorArg(a[0]);
        if (!y) return ev::throwTypeError("expected Tensor");
        return guardedOp([&] {
            brotensor::add_scalar_inplace(*y.ptr, static_cast<float>(numAt(a, 1)));
        });
    });

    nnNs.def("xavierInit", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("xavierInit(W, seed?)");
        TensorArg W = tensorArg(a[0]);
        if (!W) return ev::throwTypeError("expected Tensor");
        uint64_t s = a.size() >= 2 ? readSeedArg(a[1], 0xC0DE1234ULL) : 0xC0DE1234ULL;
        try {
            brotensor::xavier_init(*W.ptr, s);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return makeBigIntValue(s);
    });

    nnNs.def("factoredSoftmax", 4, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) {
            return ev::throwTypeError("factoredSoftmax(logits,probs,atkMask?,abilMask?)");
        }
        auto [l, p, am, bm] = tensorArgs<4>(a, {0, 1, 2, 3});
        if (!l || !p) return ev::throwTypeError("expected Tensors");
        // The factored ops take host masks on every device (they compose the
        // trailing no-op flag and upload for a GPU op themselves).
        Mask aMask, bMask;
        if (!maskArg(a, 2, am, nn::FactoredPolicyHead::N_ATTACK - 1, brotensor::Device::CPU,
                     "factoredSoftmax: atkMask", aMask) ||
            !maskArg(a, 3, bm, nn::FactoredPolicyHead::N_ABILITY - 1, brotensor::Device::CPU,
                     "factoredSoftmax: abilMask", bMask))
            return ev::undefined();
        return guardedOp([&] { nn::factored_softmax(*l.ptr, *p.ptr, aMask.ptr, bMask.ptr); });
    });

    nnNs.def("factoredXent", 8, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 6) {
            return ev::throwTypeError(
                "factoredXent(logits,mTgt,aTgt,abTgt,probs,dLogits,atkMask?,abilMask?)");
        }
        auto [l, mt, at, abt, p, dl, am, bm] = tensorArgs<8>(a, {0, 1, 2, 3, 4, 5, 6, 7});
        if (!l || !mt || !at || !abt || !p || !dl) return ev::throwTypeError("expected Tensors");
        Mask aMask, bMask;
        if (!maskArg(a, 6, am, nn::FactoredPolicyHead::N_ATTACK - 1, brotensor::Device::CPU,
                     "factoredXent: atkMask", aMask) ||
            !maskArg(a, 7, bm, nn::FactoredPolicyHead::N_ABILITY - 1, brotensor::Device::CPU,
                     "factoredXent: abilMask", bMask))
            return ev::undefined();
        float loss = 0.0f;
        try {
            loss = nn::factored_xent(*l.ptr, *mt.ptr, *at.ptr, *abt.ptr, *p.ptr, *dl.ptr,
                                     aMask.ptr, bMask.ptr);
        } catch (const std::exception& e) {
            return ev::throwError(e.what());
        }
        return ev::fromDouble(loss);
    });

    // ── Multi-head policy helpers ─────────────────────────────────────────
    // These take Float32Arrays rather than tensors, the way the trainer
    // worker shuffles data — unchanged from before the transition.

    nnNs.def("factoredToFlat", 4, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) {
            return ev::throwTypeError("factoredToFlat(logits, headSizes, flatPrior, headMasks?)");
        }
        // headSizes first: reading an array allocates, and every float* below
        // points into the moving heap, so none may be taken before it.
        auto sizes = readHeadSizes(a[1], "factoredToFlat: headSizes");
        size_t lN = 0;
        float* logits = floatPtr(a[0], lN);
        if (!logits) return ev::throwTypeError("logits must be Float32Array");
        if (sizes.empty()) return ev::throwTypeError("headSizes must be a non-empty int array");
        auto offsets = offsetsFromSizes(sizes);
        if (static_cast<int>(lN) < offsets.back()) {
            return ev::throwTypeError("logits shorter than sum(headSizes)");
        }
        size_t fN = 0;
        float* flat = floatPtr(a[2], fN);
        if (!flat) return ev::throwTypeError("flatPrior must be Float32Array");
        const int total = nn::flat_action_count(sizes);
        if (static_cast<int>(fN) < total) {
            return ev::throwTypeError("flatPrior shorter than prod(headSizes)");
        }
        size_t mN = 0;
        float* masks = (a.size() >= 4 && !ev::isUndefined(a[3]) && !ev::isNull(a[3]))
                           ? floatPtr(a[3], mN)
                           : nullptr;
        if (masks && static_cast<int>(mN) < offsets.back()) {
            return ev::throwTypeError("headMasks shorter than sum(headSizes)");
        }
        return guardedOp([&] { nn::factored_to_flat(logits, sizes, offsets, flat, masks); });
    });

    nnNs.def("flatActionCount", 1, [](Value, std::span<const Value> a) -> Value {
        if (a.empty()) return ev::throwTypeError("flatActionCount(headSizes)");
        return ev::fromDouble(
            nn::flat_action_count(readHeadSizes(a[0], "flatActionCount: headSizes")));
    });

    nnNs.def("decodeFlatAction", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("decodeFlatAction(flat, headSizes)");
        auto sizes = readHeadSizes(a[1], "decodeFlatAction: headSizes");
        if (sizes.empty()) return ev::throwTypeError("headSizes must be non-empty");
        // A negative flat index would decode to negative per-head actions.
        int flat = static_cast<int>(intAt(a, 0, 0, nn::flat_action_count(sizes) - 1,
                                          "decodeFlatAction: flat"));
        auto strides = nn::head_strides(sizes);
        std::vector<int> out(sizes.size());
        nn::decode_flat_action(flat, sizes, strides, out.data());
        return makeIntArrayValue(out);
    });

    nnNs.def("encodeFlatAction", 2, [](Value, std::span<const Value> a) -> Value {
        if (a.size() < 2) return ev::throwTypeError("encodeFlatAction(perHead, headSizes)");
        auto perHead = readIntArrayValue(a[0]);
        auto sizes = readHeadSizes(a[1], "encodeFlatAction: headSizes");
        if (sizes.empty() || perHead.size() != sizes.size()) {
            return ev::throwTypeError("perHead/headSizes length mismatch");
        }
        // Each action within its head, so the flat index stays in range.
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (perHead[i] < 0 || perHead[i] >= sizes[i]) {
                throwIntRange("encodeFlatAction: perHead[" + std::to_string(i) + "]",
                              perHead[i], 0, sizes[i] - 1);
            }
        }
        auto strides = nn::head_strides(sizes);
        return ev::fromDouble(
            nn::encode_flat_action(perHead.data(), strides, static_cast<int>(sizes.size())));
    });

    // Policy-head shape constants, mirrored so a caller can size its buffers
    // without instantiating a head.
    nnNs.set("N_MOVE", static_cast<double>(nn::FactoredPolicyHead::N_MOVE));
    nnNs.set("N_ATTACK", static_cast<double>(nn::FactoredPolicyHead::N_ATTACK));
    nnNs.set("N_ABILITY", static_cast<double>(nn::FactoredPolicyHead::N_ABILITY));
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
