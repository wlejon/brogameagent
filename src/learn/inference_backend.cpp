#include "brogameagent/learn/inference_backend.h"

#include "brogameagent/nn/policy_value_net.h"
#include "brogameagent/nn/tensor.h"

#ifdef BGA_HAS_CUDA
#include "brogameagent/learn/inference_server.h"
#include "brogameagent/nn/gpu/runtime.h"
#include "brogameagent/nn/gpu/tensor.h"
#endif

#include <stdexcept>

namespace brogameagent::learn {

namespace nn = brogameagent::nn;

// ─── DirectBackend ────────────────────────────────────────────────────────

DirectBackend::DirectBackend(nn::PolicyValueNet* net) : net_(net) {
    if (!net_) throw std::runtime_error("DirectBackend: null net");
}

int DirectBackend::num_actions() const { return net_->num_actions(); }
int DirectBackend::in_dim()      const { return net_->in_dim(); }

EvalResult DirectBackend::evaluate(const std::vector<float>& obs) {
    EvalResult r;
    if (static_cast<int>(obs.size()) != net_->in_dim()) {
        throw std::runtime_error(
            "DirectBackend::evaluate: obs.size() != net->in_dim()");
    }
    nn::Tensor x = nn::Tensor::vec(net_->in_dim());
    for (int i = 0; i < net_->in_dim(); ++i) x[i] = obs[i];
    nn::Tensor logits = nn::Tensor::vec(net_->num_actions());
    float v = 0.0f;

#ifdef BGA_HAS_CUDA
    if (net_->device() == nn::Device::GPU) {
        // Use the batched-1 GPU forward to avoid the single-sample overhead
        // path's caches. Stage as a (1, in_dim) host buffer.
        nn::Tensor host_X(1, net_->in_dim());
        for (int i = 0; i < net_->in_dim(); ++i) host_X.data[i] = obs[i];
        nn::gpu::GpuTensor X_BD, logits_BD, values_B1;
        nn::gpu::upload(host_X, X_BD);
        net_->forward_batched(X_BD, logits_BD, values_B1);
        nn::Tensor host_logits, host_values;
        nn::gpu::download(logits_BD, host_logits);
        nn::gpu::download(values_B1, host_values);
        nn::gpu::cuda_sync();
        r.logits.assign(net_->num_actions(), 0.0f);
        for (int j = 0; j < net_->num_actions(); ++j)
            r.logits[j] = host_logits.data[j];
        r.value = host_values.data[0];
        return r;
    }
#endif

    net_->forward(x, v, logits);
    r.logits.assign(net_->num_actions(), 0.0f);
    for (int j = 0; j < net_->num_actions(); ++j) r.logits[j] = logits[j];
    r.value = v;
    return r;
}

#ifdef BGA_HAS_CUDA
// ─── ServerBackend ────────────────────────────────────────────────────────

ServerBackend::ServerBackend(BatchedInferenceServer* server,
                             int num_actions, int in_dim)
    : server_(server), num_actions_(num_actions), in_dim_(in_dim) {
    if (!server_) throw std::runtime_error("ServerBackend: null server");
}

EvalResult ServerBackend::evaluate(const std::vector<float>& obs) {
    auto sr = server_->evaluate(obs);
    EvalResult r;
    r.logits = std::move(sr.logits);
    r.value  = sr.value;
    return r;
}
#endif

} // namespace brogameagent::learn
