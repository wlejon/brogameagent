#pragma once
// BGA_CHECK: a precondition on what a caller passed (a tensor's shape, a
// serialized blob's length, an index), checked in every build. An `assert`
// is gone in Release, where a wrong size reads or writes past a buffer; this
// throws std::invalid_argument naming the condition instead, which a binding
// turns into the JS error the call throws.

#include <stdexcept>
#include <string>

namespace brogameagent::nn {

[[noreturn]] inline void checkFailed(const char* cond, const char* file, int line) {
    throw std::invalid_argument(std::string("brogameagent: ") + cond + " does not hold (" + file + ":" +
                                std::to_string(line) + ")");
}

}  // namespace brogameagent::nn

#define BGA_CHECK(cond) \
    ((cond) ? static_cast<void>(0) : ::brogameagent::nn::checkFailed(#cond, __FILE__, __LINE__))
