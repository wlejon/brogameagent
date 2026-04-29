#pragma once

// BatchedInferenceServer
// ──────────────────────
// Forms a batch from concurrent calls to evaluate(), runs one batched GPU
// forward of a PolicyValueNet, and returns each caller's per-row result.
//
// Compiled only when BGA_HAS_CUDA is defined; the class is GPU-only.

#ifdef BGA_HAS_CUDA

#include "brogameagent/learn/batched_net.h"

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace brogameagent::nn { class PolicyValueNet; }

namespace brogameagent::learn {

class BatchedInferenceServer {
public:
    struct Config {
        int max_batch_size = 64;     // run a batch as soon as it's this full
        int max_wait_micros = 500;   // OR after this long since the first request
    };

    struct EvalResult {
        std::vector<float> logits;   // length net->logits_dim()
        float value = 0.0f;
    };

    // Primary constructor — takes any BatchedNet implementation.
    BatchedInferenceServer(BatchedNet* net, Config cfg);

    // Back-compat overload for existing call sites that pass a
    // PolicyValueNet*. Implemented as a thin upcast to BatchedNet*.
    BatchedInferenceServer(brogameagent::nn::PolicyValueNet* net, Config cfg);

    ~BatchedInferenceServer();

    BatchedInferenceServer(const BatchedInferenceServer&) = delete;
    BatchedInferenceServer& operator=(const BatchedInferenceServer&) = delete;

    // Blocking. Thread-safe. Throws std::runtime_error if the server is
    // shutting down or the obs has the wrong size.
    EvalResult evaluate(const std::vector<float>& obs);

    // Async variant. The future is satisfied when the request's batch runs.
    std::future<EvalResult> evaluate_async(const std::vector<float>& obs);

    int batches_run() const { return batches_run_; }

private:
    struct Pending {
        std::vector<float> obs;
        std::promise<EvalResult> promise;
    };

    void worker_loop_();
    void run_batch_(std::vector<std::unique_ptr<Pending>>& batch);

    BatchedNet* net_;
    Config cfg_;
    int in_dim_;
    int num_actions_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<std::unique_ptr<Pending>> queue_;
    bool stop_ = false;

    std::thread worker_;
    int batches_run_ = 0;
};

} // namespace brogameagent::learn

#endif // BGA_HAS_CUDA
