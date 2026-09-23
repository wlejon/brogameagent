#pragma once

#include "embed/embed.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brogameagent::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

// ─── checked number → integer conversion ────────────────────────────────────
//
// A JS number cast straight to an integer type is undefined behaviour for
// NaN, ±Infinity and anything out of the type's range, and a silently
// wrapped id / count / iteration budget is a bug the caller never sees. Every
// option and argument that becomes an integer goes through checkedInt: it
// truncates toward zero like ToIntegerOrInfinity and throws JsRangeError for
// NaN or a value outside [lo, hi]. guardNative (object_builder.h), which
// wraps every native entry point, turns that into a JS RangeError.
//
// Values parsed from a callback's RESULT while a native search is running
// must not throw (the search would unwind with its world mid-rollout); those
// use intOr, which answers a fallback instead.

/// Thrown by the checked conversions; becomes a JS RangeError.
struct JsRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// JS-safe integer bound: numbers past 2^53 are not integers a caller can
/// name exactly.
inline constexpr int64_t kMaxSafeInt = (int64_t{1} << 53) - 1;

[[noreturn]] inline void throwIntRange(std::string_view what, double d, int64_t lo, int64_t hi) {
    std::string msg(what);
    msg += " must be an integer in [";
    msg += std::to_string(lo);
    msg += ", ";
    msg += std::to_string(hi);
    msg += "], got ";
    if (std::isnan(d)) msg += "NaN";
    else if (std::isinf(d)) msg += d > 0 ? "Infinity" : "-Infinity";
    else msg += std::to_string(d);
    throw JsRangeError(msg);
}

/// `d` truncated toward zero, or JsRangeError naming `what` when it is NaN
/// or outside [lo, hi]. Bounds wider than ±2^53 are clamped to it.
inline int64_t checkedInt(double d, int64_t lo, int64_t hi, std::string_view what) {
    if (lo < -kMaxSafeInt) lo = -kMaxSafeInt;
    if (hi > kMaxSafeInt) hi = kMaxSafeInt;
    const double t = std::trunc(d);  // NaN stays NaN, ±Inf stays ±Inf
    if (!(t >= static_cast<double>(lo) && t <= static_cast<double>(hi))) {
        throwIntRange(what, d, lo, hi);
    }
    return static_cast<int64_t>(t);
}

inline int32_t checkedI32(double d, std::string_view what,
                          int32_t lo = std::numeric_limits<int32_t>::min(),
                          int32_t hi = std::numeric_limits<int32_t>::max()) {
    return static_cast<int32_t>(checkedInt(d, lo, hi, what));
}

inline uint32_t checkedU32(double d, std::string_view what,
                           uint32_t lo = 0,
                           uint32_t hi = std::numeric_limits<uint32_t>::max()) {
    return static_cast<uint32_t>(checkedInt(d, lo, hi, what));
}

/// Never throws: `d` truncated when it lies in [lo, hi], else `fallback`.
/// For callback results read in the middle of a native search.
inline int32_t intOr(double d, int32_t fallback,
                     int32_t lo = std::numeric_limits<int32_t>::min(),
                     int32_t hi = std::numeric_limits<int32_t>::max()) {
    const double t = std::trunc(d);
    if (!(t >= static_cast<double>(lo) && t <= static_cast<double>(hi))) return fallback;
    return static_cast<int32_t>(t);
}

/// A u64 from a BigInt (its low 64 bits) or a number in [0, 2^64). Any other
/// number throws JsRangeError; a non-number answers `def`.
inline uint64_t checkedU64(Value v, std::string_view what, uint64_t def = 0) {
    if (ev::isBigInt(v)) return ev::toUint64(v);
    if (!ev::isNumber(v)) return def;
    const double t = std::trunc(ev::toDouble(v));
    if (!(t >= 0.0 && t < 18446744073709551616.0)) {
        std::string msg(what);
        msg += " must be a non-negative integer below 2^64 (or a BigInt)";
        throw JsRangeError(msg);
    }
    return static_cast<uint64_t>(t);
}

/// "argument N" for the positional helpers' messages.
inline std::string argName(size_t i) {
    return "argument " + std::to_string(i);
}

inline double numAt(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return 0.0;
    Value v = args[i];
    if (!ev::isNumber(v)) return 0.0;
    double d = ev::toDouble(v);
    return std::isnan(d) ? 0.0 : d;
}

// The integer argument readers: a missing or non-number argument is 0 (the
// historical default); a number that is NaN or out of range throws. `what`
// names the parameter in the message.

inline int64_t intAt(std::span<const Value> args, size_t i, int64_t lo, int64_t hi,
                     std::string_view what = {}) {
    if (i >= args.size() || !ev::isNumber(args[i])) {
        return 0 < lo ? lo : (0 > hi ? hi : 0);
    }
    return checkedInt(ev::toDouble(args[i]), lo, hi, what.empty() ? argName(i) : what);
}

inline int32_t i32At(std::span<const Value> args, size_t i, std::string_view what = {}) {
    return static_cast<int32_t>(intAt(args, i, std::numeric_limits<int32_t>::min(),
                                      std::numeric_limits<int32_t>::max(), what));
}

inline uint32_t u32At(std::span<const Value> args, size_t i, std::string_view what = {}) {
    return static_cast<uint32_t>(intAt(args, i, 0, std::numeric_limits<uint32_t>::max(), what));
}

inline int64_t i64At(std::span<const Value> args, size_t i, std::string_view what = {}) {
    return intAt(args, i, -kMaxSafeInt, kMaxSafeInt, what);
}

/// A size / dimension / count argument: an int in [0, 2^31 - 1].
inline int dimAt(std::span<const Value> args, size_t i, std::string_view what = {}) {
    return static_cast<int>(intAt(args, i, 0, std::numeric_limits<int32_t>::max(), what));
}

/// A 32-bit seed: any safe integer, taken modulo 2^32 (so Date.now() or a
/// negative number are usable seeds); NaN / ±Infinity throw.
inline uint32_t seed32At(std::span<const Value> args, size_t i, std::string_view what = {}) {
    return static_cast<uint32_t>(static_cast<uint64_t>(
        intAt(args, i, -kMaxSafeInt, kMaxSafeInt, what)));
}

inline uint64_t u64At(std::span<const Value> args, size_t i, std::string_view what = {}) {
    if (i >= args.size()) return 0;
    return checkedU64(args[i], what.empty() ? argName(i) : what);
}

inline bool boolAt(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return false;
    return ev::toBool(args[i]);
}

inline std::string strAt(std::span<const Value> args, size_t i) {
    if (i >= args.size() || ev::isUndefined(args[i]) || ev::isNull(args[i]) || ev::isSymbol(args[i])) return "";
    return ev::toUtf8(args[i]);
}

inline Value argAt(std::span<const Value> args, size_t i) {
    return i < args.size() ? args[i] : ev::undefined();
}

inline bool hasArg(std::span<const Value> args, size_t i) {
    return i < args.size() && !ev::isUndefined(args[i]);
}

class ArgReader {
public:
    explicit ArgReader(std::span<const Value> args) : args_(args) {}

    double getDouble(size_t i, double def = 0.0) const {
        if (i >= args_.size() || !ev::isNumber(args_[i])) return def;
        double d = ev::toDouble(args_[i]);
        return std::isnan(d) ? def : d;
    }
    int getInt(size_t i, int def = 0) const {
        if (i >= args_.size() || !ev::isNumber(args_[i])) return def;
        return i32At(args_, i);
    }
    uint32_t getUint(size_t i, uint32_t def = 0) const {
        if (i >= args_.size() || !ev::isNumber(args_[i])) return def;
        return u32At(args_, i);
    }
    bool getBool(size_t i, bool def = false) const {
        return hasArg(args_, i) ? boolAt(args_, i) : def;
    }
    std::string getString(size_t i, const std::string& def = "") const {
        return hasArg(args_, i) ? strAt(args_, i) : def;
    }
    Value get(size_t i) const {
        return argAt(args_, i);
    }
    bool has(size_t i) const {
        return hasArg(args_, i);
    }
    size_t count() const {
        return args_.size();
    }

private:
    std::span<const Value> args_;
};

} // namespace brogameagent::api
