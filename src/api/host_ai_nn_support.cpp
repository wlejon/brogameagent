// Marshalling shared by the nn / learn / grid bindings: tensor arguments,
// raw byte buffers, BigInt seeds & weight versions, int arrays, and the
// device name -> brotensor::Device resolution the `to("gpu")` methods use.

#include "host_ai_nn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <brotensor/api.h>

#include <cctype>
#include <cstdio>

namespace brogameagent::api {

// ---------------------------------------------------------------------------
// Tensor arguments
// ---------------------------------------------------------------------------

bool isGpuTensorValue(Value v) {
    if (!ev::isObject(v) || ev::isTypedArray(v)) return false;
    if (!ev::handleData(v)) return false;
    // Rooted before anything below allocates.
    ev::Persistent subject(v);

    ev::GlobalValue proto = ev::nativeClassPrototype("__bro_native.tensor.GpuTensor");
    if (!proto.found || !ev::isObject(proto.value)) return false;
    ev::Persistent protoSlot(proto.value);

    ev::GlobalValue objectG = ev::globalValue("Object");
    if (!objectG.found || !ev::isObject(objectG.value)) return false;
    ev::Persistent objectSlot(objectG.value);

    ev::Persistent getProto(ev::getProperty(objectSlot.get(), "getPrototypeOf"));
    if (!ev::isFunction(getProto.get())) return false;

    Value arg = subject.get();
    ev::CallResult r = ev::call(getProto.get(), objectSlot.get(), std::span<const Value>(&arg, 1));
    if (r.thrown) return false;
    return r.value == protoSlot.get();
}

TensorArg tensorArg(Value v) {
    TensorArg out;
    if (ev::isUndefined(v) || ev::isNull(v)) return out;

    // A Float32Array is viewed in place: the old surface's AITensor is gone,
    // so a plain typed array is how a caller hands over CPU-side floats
    // without allocating a GpuTensor for them.
    if (auto info = ev::typedArrayInfo(v)) {
        if (info.data && info.elementKind == ev::elements::Float32) {
            out.view = brotensor::Tensor::view(brotensor::Device::CPU, info.data,
                                               static_cast<int>(info.elementCount), 1,
                                               brotensor::Dtype::FP32);
            out.ptr = &out.view;
            return out;
        }
        return out;
    }

    // isGpuTensorValue allocates; the handle is re-read from a root after it.
    ev::Persistent root(v);
    if (!isGpuTensorValue(root.get())) return out;
    out.ptr = brotensor::api::getTensorFromHandle(root.get());
    return out;
}

RawBytes rawBytes(Value v) {
    RawBytes out;
    if (auto info = ev::typedArrayInfo(v)) {
        out.data = info.data;
        out.size = info.byteLength;
        return out;
    }
    if (auto buf = ev::arrayBufferInfo(v)) {
        out.data = buf.data;
        out.size = buf.byteLength;
    }
    return out;
}

float* floatPtr(Value v, size_t& count) {
    count = 0;
    if (ev::isUndefined(v) || ev::isNull(v)) return nullptr;
    auto info = ev::typedArrayInfo(v);
    if (!info.data || info.elementKind != ev::elements::Float32) return nullptr;
    count = info.elementCount;
    return reinterpret_cast<float*>(info.data);
}

// ---------------------------------------------------------------------------
// Typed arrays / arrays / BigInt
// ---------------------------------------------------------------------------

Value makeUint8Array(const uint8_t* data, size_t count) {
    ev::Persistent view(ev::createTypedArray(ev::elements::Uint8, static_cast<uint32_t>(count)));
    if (!ev::isObject(view.get())) return ev::undefined();
    if (data && count > 0) {
        ev::fillTypedArray(view.get(), std::span<const uint8_t>(data, count));
    }
    return view.get();
}

Value makeBigIntValue(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));

    ev::GlobalValue bigIntG = ev::globalValue("BigInt");
    if (!bigIntG.found || !ev::isFunction(bigIntG.value)) {
        // No BigInt in this realm — a Number is the closest honest answer.
        return ev::fromDouble(static_cast<double>(v));
    }
    ev::Persistent fn(bigIntG.value);
    Value arg = ev::fromUtf8(buf);
    ev::CallResult r = ev::call(fn.get(), ev::undefined(), std::span<const Value>(&arg, 1));
    if (r.thrown) return ev::fromDouble(static_cast<double>(v));
    return r.value;
}

uint64_t readSeedArg(Value v, uint64_t def) {
    if (ev::isUndefined(v) || ev::isNull(v) || ev::isObject(v)) return def;
    return checkedU64(v, "seed", def);
}

Value makeIntArrayValue(const std::vector<int>& v) {
    return hostArrayOf(v.size(), [&](size_t i) { return ev::fromDouble(v[i]); });
}

std::vector<int> readIntArrayValue(Value arr, bool checked) {
    std::vector<int> out;
    if (!ev::isObject(arr)) return out;
    ev::Persistent root(arr);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return out;
    const uint32_t len = toLength(lenV);
    out.reserve(reserveHint(len));
    for (uint32_t i = 0; i < len; ++i) {
        Value e = ev::getElement(root.get(), i);
        double d = (!ev::isUndefined(e) && !ev::isObject(e)) ? ev::toDouble(e) : 0.0;
        out.push_back(checked ? checkedI32(d, "array element " + std::to_string(i))
                              : intOr(d, -1));
    }
    return out;
}

std::vector<int> readIntArrayProp(Value obj, const char* key) {
    // The old readIntArray dropped entries <= 0 (a hidden-width list has no
    // zero-width layer); readIntArrayValue keeps them, which is what the
    // decode/encode helpers want.
    std::vector<int> out;
    if (!ev::isObject(obj)) return out;
    ev::Persistent root(obj);
    std::vector<int> raw = readIntArrayValue(ev::getProperty(root.get(), key));
    out.reserve(raw.size());
    for (int w : raw) {
        if (w > 0) out.push_back(w);
    }
    return out;
}

void checkHeadSizes(const std::vector<int>& sizes, const char* what) {
    constexpr size_t kMaxHeads = 64;
    constexpr int kMaxHead = 1 << 24;
    if (sizes.size() > kMaxHeads) {
        throw JsRangeError(std::string(what) + " may hold at most " +
                           std::to_string(kMaxHeads) + " heads");
    }
    int64_t product = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] < 1 || sizes[i] > kMaxHead) {
            throwIntRange(std::string(what) + "[" + std::to_string(i) + "]", sizes[i], 1,
                          kMaxHead);
        }
        product *= sizes[i];
        if (product > std::numeric_limits<int32_t>::max()) {
            throw JsRangeError(std::string(what) +
                               ": the product of the head sizes must fit in int32");
        }
    }
}

std::vector<int> readHeadSizes(Value arr, const char* what) {
    std::vector<int> sizes = readIntArrayValue(arr);
    checkHeadSizes(sizes, what);
    return sizes;
}

int getIntProp(Value obj, const char* key, int def) {
    return getI32Property(obj, key, def);
}

int getCountProp(Value obj, const char* key, int def) {
    return getI32Property(obj, key, def, 0);
}

// ---------------------------------------------------------------------------
// Device resolution
// ---------------------------------------------------------------------------

bool resolveDevice(const std::string& name, brotensor::Device& out, std::string& error) {
    std::string dev = name;
    for (auto& c : dev) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    out = brotensor::Device::CPU;
    if (dev != "gpu") return true;

    brotensor::init();
    out = brotensor::default_device();
    if (out == brotensor::Device::CPU) {
        // GPU-first: a CPU-only build refuses rather than pretending. The old
        // binding threw exactly this message.
        error = "no GPU backend available (CPU-only build)";
        return false;
    }
    return true;
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
