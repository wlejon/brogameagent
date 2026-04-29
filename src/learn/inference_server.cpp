#ifdef BGA_HAS_CUDA

#include "brogameagent/learn/inference_server.h"

#include "brogameagent/nn/gpu/runtime.h"
#include "brogameagent/nn/gpu/tensor.h"
#include "brogameagent/nn/tensor.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace brogameagent::learn {

namespace gpu = brogameagent::nn::gpu;
using brogameagent::nn::Tensor;

BatchedInferenceServer::BatchedInferenceServer(
    brogameagent::nn::PolicyValueNet* net, Config cfg)
    : net_(net),
      cfg_(cfg),
      in_dim_(net ? net->in_dim() : 0),
      num_actions_(net ? net->num_actions() : 0) {
    if (!net_) throw std::runtime_error("BatchedInferenceServer: null net");
    if (net_->device() != brogameagent::nn::Device::GPU) {
        throw std::runtime_error(
            "BatchedInferenceServer: net must be on Device::GPU");
    }
    if (cfg_.max_batch_size < 1) cfg_.max_batch_size = 1;
    if (cfg_.max_wait_micros < 0) cfg_.max_wait_micros = 0;
    worker_ = std::thread(&BatchedInferenceServer::worker_loop_, this);
}

BatchedInferenceServer::~BatchedInferenceServer() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();

    // Reject anything that may have been enqueued after the worker ran its
    // last drain. (Worker drains once after stop_ becomes true, so this is
    // belt-and-suspenders for races during shutdown.)
    std::lock_guard<std::mutex> lk(mu_);
    while (!queue_.empty()) {
        auto p = std::move(queue_.front());
        queue_.pop();
        try {
            throw std::runtime_error(
                "BatchedInferenceServer: shutting down");
        } catch (...) {
            p->promise.set_exception(std::current_exception());
        }
    }
}

std::future<BatchedInferenceServer::EvalResult>
BatchedInferenceServer::evaluate_async(const std::vector<float>& obs) {
    if (static_cast<int>(obs.size()) != in_dim_) {
        throw std::runtime_error(
            "BatchedInferenceServer::evaluate: obs.size() != net->in_dim()");
    }
    auto pending = std::make_unique<Pending>();
    pending->obs = obs;
    auto fut = pending->promise.get_future();
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) {
            throw std::runtime_error(
                "BatchedInferenceServer: server is shutting down");
        }
        queue_.push(std::move(pending));
    }
    cv_.notify_one();
    return fut;
}

BatchedInferenceServer::EvalResult
BatchedInferenceServer::evaluate(const std::vector<float>& obs) {
    auto fut = evaluate_async(obs);
    return fut.get();
}

void BatchedInferenceServer::worker_loop_() {
    // Reusable host staging buffer + GPU tensors. Grown only when needed.
    Tensor host_X;       // (B, in_dim)
    Tensor host_logits;  // (B, num_actions)
    Tensor host_values;  // (B, 1)
    gpu::GpuTensor X_BD, logits_BD, values_B1;

    std::vector<std::unique_ptr<Pending>> batch;
    batch.reserve(cfg_.max_batch_size);

    while (true) {
        // ── Wait for first request (or shutdown) ──────────────────────────
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });

        if (stop_ && queue_.empty()) return;

        // Grab the first request immediately.
        auto first_seen = std::chrono::steady_clock::now();
        batch.clear();
        batch.push_back(std::move(queue_.front()));
        queue_.pop();

        // ── Coalesce more requests up to max_batch_size or until timeout ──
        while (static_cast<int>(batch.size()) < cfg_.max_batch_size) {
            if (!queue_.empty()) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop();
                continue;
            }
            if (stop_) break;

            // Wait for either more requests, shutdown, or timeout.
            const auto deadline = first_seen +
                std::chrono::microseconds(cfg_.max_wait_micros);
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            cv_.wait_until(lk, deadline,
                           [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) break;
            // Loop continues; will drain queue_ on next iteration.
        }

        lk.unlock();

        // ── Run the batch ────────────────────────────────────────────────
        const int B = static_cast<int>(batch.size());

        // Stage observations into a host (B, in_dim) buffer.
        if (host_X.rows != B || host_X.cols != in_dim_) {
            host_X.resize(B, in_dim_);
        }
        for (int b = 0; b < B; ++b) {
            const auto& obs = batch[b]->obs;
            const size_t off = static_cast<size_t>(b) * in_dim_;
            for (int j = 0; j < in_dim_; ++j) host_X.data[off + j] = obs[j];
        }

        try {
            gpu::upload(host_X, X_BD);
            net_->forward_batched(X_BD, logits_BD, values_B1);
            gpu::download(logits_BD, host_logits);
            gpu::download(values_B1, host_values);
            gpu::cuda_sync();

            for (int b = 0; b < B; ++b) {
                EvalResult r;
                r.logits.assign(num_actions_, 0.0f);
                const size_t off = static_cast<size_t>(b) * num_actions_;
                for (int j = 0; j < num_actions_; ++j)
                    r.logits[j] = host_logits.data[off + j];
                r.value = host_values.data[b];
                batch[b]->promise.set_value(std::move(r));
            }
            ++batches_run_;
        } catch (...) {
            auto eptr = std::current_exception();
            for (int b = 0; b < B; ++b) {
                try {
                    batch[b]->promise.set_exception(eptr);
                } catch (...) {
                    // promise already satisfied — ignore.
                }
            }
        }
    }
}

void BatchedInferenceServer::run_batch_(
    std::vector<std::unique_ptr<Pending>>& /*batch*/) {
    // Unused: logic is inlined into worker_loop_ for simplicity.
}

} // namespace brogameagent::learn

#endif // BGA_HAS_CUDA
