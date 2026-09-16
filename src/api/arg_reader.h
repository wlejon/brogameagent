#pragma once

#include "embed/embed.h"

#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace brogameagent::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

inline double numAt(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return 0.0;
    Value v = args[i];
    if (ev::isObject(v)) return 0.0;
    double d = ev::toDouble(v);
    return std::isnan(d) ? 0.0 : d;
}

inline int32_t i32At(std::span<const Value> args, size_t i) {
    return static_cast<int32_t>(static_cast<int64_t>(numAt(args, i)));
}

inline uint32_t u32At(std::span<const Value> args, size_t i) {
    return static_cast<uint32_t>(static_cast<int64_t>(numAt(args, i)));
}

inline int64_t i64At(std::span<const Value> args, size_t i) {
    return static_cast<int64_t>(numAt(args, i));
}

inline uint64_t u64At(std::span<const Value> args, size_t i) {
    if (i >= args.size()) return 0;
    Value v = args[i];
    if (ev::isObject(v)) return 0;
    return ev::toUint64(v);
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
        return hasArg(args_, i) ? numAt(args_, i) : def;
    }
    int getInt(size_t i, int def = 0) const {
        return hasArg(args_, i) ? i32At(args_, i) : def;
    }
    uint32_t getUint(size_t i, uint32_t def = 0) const {
        return hasArg(args_, i) ? u32At(args_, i) : def;
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
