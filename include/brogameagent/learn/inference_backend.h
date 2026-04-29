#pragma once

// IInferenceBackend
// ─────────────────
// Abstraction for "given an observation vector, return (logits, value)".
// MCTS leaf evaluation calls evaluate() once per expansion. Two implementations:
//
//   DirectBackend  — wraps a PolicyValueNet*; calls forward() synchronously.
//                    Always available (CPU-build-safe).
//   ServerBackend  — submits to a BatchedInferenceServer. CUDA only.
//
// The interface lets MCTS integrate either path with no other changes:
//   GenericMcts holds prior_fn / value_fn callables already; build them on
//   top of an IInferenceBackend* once and inject via set_prior_fn /
//   set_value_fn.

#include <memory>
#include <vector>

namespace brogameagent::nn { class PolicyValueNet; }

namespace brogameagent::learn {

#ifdef BGA_HAS_CUDA
class BatchedInferenceServer;
#endif

struct EvalResult {
    std::vector<float> logits;
    float value = 0.0f;
};

class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual EvalResult evaluate(const std::vector<float>& obs) = 0;

    // Width of the policy logits returned by evaluate(). Used by callers that
    // need to know the action space ahead of time (e.g. building a prior_fn
    // with mask + softmax).
    virtual int num_actions() const = 0;
    virtual int in_dim() const = 0;
};

// Synchronous direct call into a PolicyValueNet. Works on either CPU or GPU
// device — uses the appropriate forward path automatically.
class DirectBackend : public IInferenceBackend {
public:
    explicit DirectBackend(brogameagent::nn::PolicyValueNet* net);
    EvalResult evaluate(const std::vector<float>& obs) override;
    int num_actions() const override;
    int in_dim() const override;

private:
    brogameagent::nn::PolicyValueNet* net_;
};

#ifdef BGA_HAS_CUDA
// Submits requests to a BatchedInferenceServer. The caller owns the server.
class ServerBackend : public IInferenceBackend {
public:
    explicit ServerBackend(BatchedInferenceServer* server,
                           int num_actions, int in_dim);
    EvalResult evaluate(const std::vector<float>& obs) override;
    int num_actions() const override { return num_actions_; }
    int in_dim() const override { return in_dim_; }

private:
    BatchedInferenceServer* server_;
    int num_actions_;
    int in_dim_;
};
#endif

} // namespace brogameagent::learn
