#pragma once

#include "circuits.h"
#include "tensor.h"

#include <cstdint>

namespace brogameagent::nn {

// ─── GRUCell ───────────────────────────────────────────────────────────────
//
// Standard GRU update:
//   r = sigmoid(W_ir x + W_hr h_prev + b_r)
//   z = sigmoid(W_iz x + W_hz h_prev + b_z)
//   n = tanh  (W_in x + r * (W_hn h_prev + b_hn) + b_in)
//   h = (1 - z) * n + z * h_prev
// Single-sample. Six weight matrices + four bias vectors.

class GRUCell : public ICircuit {
public:
    GRUCell() = default;
    GRUCell(int in_dim, int hidden, uint64_t& rng_state) { init(in_dim, hidden, rng_state); }

    void init(int in_dim, int hidden, uint64_t& rng_state);

    void forward(const Tensor& x, const Tensor& h_prev, Tensor& h);
    void backward(const Tensor& dH, Tensor& dX, Tensor& dH_prev);

    int in_dim() const { return W_ir_.cols; }
    int hidden() const { return W_ir_.rows; }

    const char* name() const override { return "GRUCell"; }
    int  num_params() const override;
    void zero_grad() override;
    void sgd_step(float lr, float momentum) override;
    void adam_step(float lr, float beta1, float beta2, float eps, int step);
    void save_to(std::vector<uint8_t>& out) const override;
    void load_from(const uint8_t* data, size_t& offset, size_t size) override;

    Tensor& W_ir() { return W_ir_; }
    Tensor& W_hr() { return W_hr_; }
    Tensor& W_iz() { return W_iz_; }
    Tensor& W_hz() { return W_hz_; }
    Tensor& W_in() { return W_in_; }
    Tensor& W_hn() { return W_hn_; }
    Tensor& b_r()  { return b_r_; }
    Tensor& b_z()  { return b_z_; }
    Tensor& b_in() { return b_in_; }
    Tensor& b_hn() { return b_hn_; }

    Tensor& dW_ir() { return dW_ir_; }
    Tensor& dW_iz() { return dW_iz_; }
    Tensor& dW_in() { return dW_in_; }
    Tensor& dW_hr() { return dW_hr_; }
    Tensor& dW_hz() { return dW_hz_; }
    Tensor& dW_hn() { return dW_hn_; }

private:
    // (H, I) input-to-hidden and (H, H) hidden-to-hidden.
    Tensor W_ir_, W_iz_, W_in_;
    Tensor W_hr_, W_hz_, W_hn_;
    Tensor b_r_, b_z_, b_in_, b_hn_;

    // Gradients.
    Tensor dW_ir_, dW_iz_, dW_in_;
    Tensor dW_hr_, dW_hz_, dW_hn_;
    Tensor db_r_, db_z_, db_in_, db_hn_;

    // Velocities.
    Tensor vW_ir_, vW_iz_, vW_in_;
    Tensor vW_hr_, vW_hz_, vW_hn_;
    Tensor vb_r_, vb_z_, vb_in_, vb_hn_;

    // Adam moment buffers.
    Tensor mW_ir_, mW_iz_, mW_in_;
    Tensor mW_hr_, mW_hz_, mW_hn_;
    Tensor mb_r_, mb_z_, mb_in_, mb_hn_;
    Tensor vAW_ir_, vAW_iz_, vAW_in_;
    Tensor vAW_hr_, vAW_hz_, vAW_hn_;
    Tensor vAb_r_, vAb_z_, vAb_in_, vAb_hn_;

    // Caches for backward.
    Tensor x_cache_, h_prev_cache_;
    Tensor r_, z_, n_;          // sigmoid/tanh outputs
    Tensor hn_pre_;             // W_hn h_prev + b_hn (pre r-gate, post-bias)
};

} // namespace brogameagent::nn
