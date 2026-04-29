#include "brogameagent/nn/gru.h"
#include "brogameagent/nn/ops.h"

#include <cassert>
#include <cmath>

namespace brogameagent::nn {

static inline void mat_vec(const Tensor& W, const Tensor& x, Tensor& y) {
    const int out = W.rows, in = W.cols;
    assert(x.size() == in && y.size() == out);
    for (int i = 0; i < out; ++i) {
        float acc = 0.0f;
        const float* row = W.ptr() + static_cast<size_t>(i) * in;
        for (int j = 0; j < in; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
}

void GRUCell::init(int in_dim, int hidden, uint64_t& rng_state) {
    const int I = in_dim, H = hidden;
    W_ir_.resize(H, I); W_iz_.resize(H, I); W_in_.resize(H, I);
    W_hr_.resize(H, H); W_hz_.resize(H, H); W_hn_.resize(H, H);
    b_r_.resize(H, 1); b_z_.resize(H, 1); b_in_.resize(H, 1); b_hn_.resize(H, 1);
    xavier_init(W_ir_, rng_state); xavier_init(W_iz_, rng_state); xavier_init(W_in_, rng_state);
    xavier_init(W_hr_, rng_state); xavier_init(W_hz_, rng_state); xavier_init(W_hn_, rng_state);

    dW_ir_.resize(H, I); dW_iz_.resize(H, I); dW_in_.resize(H, I);
    dW_hr_.resize(H, H); dW_hz_.resize(H, H); dW_hn_.resize(H, H);
    db_r_.resize(H, 1); db_z_.resize(H, 1); db_in_.resize(H, 1); db_hn_.resize(H, 1);

    vW_ir_.resize(H, I); vW_iz_.resize(H, I); vW_in_.resize(H, I);
    vW_hr_.resize(H, H); vW_hz_.resize(H, H); vW_hn_.resize(H, H);
    vb_r_.resize(H, 1); vb_z_.resize(H, 1); vb_in_.resize(H, 1); vb_hn_.resize(H, 1);

    mW_ir_.resize(H, I); mW_iz_.resize(H, I); mW_in_.resize(H, I);
    mW_hr_.resize(H, H); mW_hz_.resize(H, H); mW_hn_.resize(H, H);
    mb_r_.resize(H, 1); mb_z_.resize(H, 1); mb_in_.resize(H, 1); mb_hn_.resize(H, 1);
    vAW_ir_.resize(H, I); vAW_iz_.resize(H, I); vAW_in_.resize(H, I);
    vAW_hr_.resize(H, H); vAW_hz_.resize(H, H); vAW_hn_.resize(H, H);
    vAb_r_.resize(H, 1); vAb_z_.resize(H, 1); vAb_in_.resize(H, 1); vAb_hn_.resize(H, 1);
    mW_ir_.zero(); mW_iz_.zero(); mW_in_.zero();
    mW_hr_.zero(); mW_hz_.zero(); mW_hn_.zero();
    mb_r_.zero(); mb_z_.zero(); mb_in_.zero(); mb_hn_.zero();
    vAW_ir_.zero(); vAW_iz_.zero(); vAW_in_.zero();
    vAW_hr_.zero(); vAW_hz_.zero(); vAW_hn_.zero();
    vAb_r_.zero(); vAb_z_.zero(); vAb_in_.zero(); vAb_hn_.zero();

    x_cache_.resize(I, 1);
    h_prev_cache_.resize(H, 1);
    r_.resize(H, 1); z_.resize(H, 1); n_.resize(H, 1);
    hn_pre_.resize(H, 1);
}

int GRUCell::num_params() const {
    return W_ir_.size() + W_iz_.size() + W_in_.size()
         + W_hr_.size() + W_hz_.size() + W_hn_.size()
         + b_r_.size() + b_z_.size() + b_in_.size() + b_hn_.size();
}

void GRUCell::forward(const Tensor& x, const Tensor& h_prev, Tensor& h) {
    const int H = hidden();
    assert(h.size() == H);
    x_cache_ = x;
    h_prev_cache_ = h_prev;

    Tensor tmp_i(H, 1), tmp_h(H, 1);

    // r
    mat_vec(W_ir_, x, tmp_i);
    mat_vec(W_hr_, h_prev, tmp_h);
    for (int i = 0; i < H; ++i) r_[i] = tmp_i[i] + tmp_h[i] + b_r_[i];
    sigmoid_forward(r_, r_);

    // z
    mat_vec(W_iz_, x, tmp_i);
    mat_vec(W_hz_, h_prev, tmp_h);
    for (int i = 0; i < H; ++i) z_[i] = tmp_i[i] + tmp_h[i] + b_z_[i];
    sigmoid_forward(z_, z_);

    // n: tanh( W_in x + b_in + r * (W_hn h_prev + b_hn) )
    mat_vec(W_in_, x, tmp_i);
    mat_vec(W_hn_, h_prev, tmp_h);
    for (int i = 0; i < H; ++i) hn_pre_[i] = tmp_h[i] + b_hn_[i];
    for (int i = 0; i < H; ++i) n_[i] = tmp_i[i] + b_in_[i] + r_[i] * hn_pre_[i];
    tanh_forward(n_, n_);

    // h = (1 - z) * n + z * h_prev
    for (int i = 0; i < H; ++i) h[i] = (1.0f - z_[i]) * n_[i] + z_[i] * h_prev[i];
}

void GRUCell::backward(const Tensor& dH, Tensor& dX, Tensor& dH_prev) {
    const int H = hidden(), I = in_dim();
    assert(dH.size() == H);

    // h = (1 - z) * n + z * h_prev
    Tensor dz(H, 1), dn(H, 1);
    Tensor dh_prev_direct(H, 1);
    for (int i = 0; i < H; ++i) {
        dn[i] = dH[i] * (1.0f - z_[i]);
        dz[i] = dH[i] * (h_prev_cache_[i] - n_[i]);
        dh_prev_direct[i] = dH[i] * z_[i];
    }

    // n = tanh(n_pre) where n_pre = W_in x + b_in + r * hn_pre
    Tensor dn_pre(H, 1);
    tanh_backward(n_, dn, dn_pre);

    // Contribution paths from n_pre:
    //   d(W_in x + b_in) = dn_pre
    //   d r = dn_pre * hn_pre
    //   d hn_pre = dn_pre * r   (hn_pre = W_hn h_prev + b_hn)
    Tensor dr(H, 1), d_hn_pre(H, 1);
    for (int i = 0; i < H; ++i) {
        dr[i]       = dn_pre[i] * hn_pre_[i];
        d_hn_pre[i] = dn_pre[i] * r_[i];
    }

    // Sigmoid backprop for r and z (r_ and z_ are post-sigmoid).
    Tensor dr_pre(H, 1), dz_pre(H, 1);
    sigmoid_backward(r_, dr, dr_pre);
    sigmoid_backward(z_, dz, dz_pre);

    // Now we have grads on the pre-activation "inputs" for r, z, n:
    //   r_pre = W_ir x + W_hr h_prev + b_r     -> dr_pre
    //   z_pre = W_iz x + W_hz h_prev + b_z     -> dz_pre
    //   W_in x + b_in                          -> dn_pre   (separate path to W_in/b_in)
    //   hn_pre = W_hn h_prev + b_hn            -> d_hn_pre

    // Biases.
    for (int i = 0; i < H; ++i) {
        db_r_[i]  += dr_pre[i];
        db_z_[i]  += dz_pre[i];
        db_in_[i] += dn_pre[i];
        db_hn_[i] += d_hn_pre[i];
    }

    // dX and input-side weight grads.
    dX.zero();
    for (int i = 0; i < H; ++i) {
        const float gr = dr_pre[i];
        const float gz = dz_pre[i];
        const float gn = dn_pre[i];
        for (int j = 0; j < I; ++j) {
            dW_ir_(i, j) += gr * x_cache_[j];
            dW_iz_(i, j) += gz * x_cache_[j];
            dW_in_(i, j) += gn * x_cache_[j];
            dX[j] += gr * W_ir_(i, j) + gz * W_iz_(i, j) + gn * W_in_(i, j);
        }
    }

    // dH_prev and hidden-side weight grads.
    for (int i = 0; i < H; ++i) dH_prev[i] = dh_prev_direct[i];
    for (int i = 0; i < H; ++i) {
        const float gr = dr_pre[i];
        const float gz = dz_pre[i];
        const float gn = d_hn_pre[i];
        for (int j = 0; j < H; ++j) {
            dW_hr_(i, j) += gr * h_prev_cache_[j];
            dW_hz_(i, j) += gz * h_prev_cache_[j];
            dW_hn_(i, j) += gn * h_prev_cache_[j];
            dH_prev[j] += gr * W_hr_(i, j) + gz * W_hz_(i, j) + gn * W_hn_(i, j);
        }
    }
}

void GRUCell::zero_grad() {
    dW_ir_.zero(); dW_iz_.zero(); dW_in_.zero();
    dW_hr_.zero(); dW_hz_.zero(); dW_hn_.zero();
    db_r_.zero(); db_z_.zero(); db_in_.zero(); db_hn_.zero();
}

static void sgd_one(Tensor& W, Tensor& vW, const Tensor& dW, float lr, float m) {
    const int n = W.size();
    float* w = W.ptr(); float* v = vW.ptr(); const float* g = dW.ptr();
    for (int i = 0; i < n; ++i) {
        v[i] = m * v[i] + g[i];
        w[i] -= lr * v[i];
    }
}

void GRUCell::sgd_step(float lr, float momentum) {
    sgd_one(W_ir_, vW_ir_, dW_ir_, lr, momentum);
    sgd_one(W_iz_, vW_iz_, dW_iz_, lr, momentum);
    sgd_one(W_in_, vW_in_, dW_in_, lr, momentum);
    sgd_one(W_hr_, vW_hr_, dW_hr_, lr, momentum);
    sgd_one(W_hz_, vW_hz_, dW_hz_, lr, momentum);
    sgd_one(W_hn_, vW_hn_, dW_hn_, lr, momentum);
    sgd_one(b_r_,  vb_r_,  db_r_,  lr, momentum);
    sgd_one(b_z_,  vb_z_,  db_z_,  lr, momentum);
    sgd_one(b_in_, vb_in_, db_in_, lr, momentum);
    sgd_one(b_hn_, vb_hn_, db_hn_, lr, momentum);
}

void GRUCell::adam_step(float lr, float b1, float b2, float eps, int step) {
    adam_step_cpu(W_ir_, dW_ir_, mW_ir_, vAW_ir_, lr, b1, b2, eps, step);
    adam_step_cpu(W_iz_, dW_iz_, mW_iz_, vAW_iz_, lr, b1, b2, eps, step);
    adam_step_cpu(W_in_, dW_in_, mW_in_, vAW_in_, lr, b1, b2, eps, step);
    adam_step_cpu(W_hr_, dW_hr_, mW_hr_, vAW_hr_, lr, b1, b2, eps, step);
    adam_step_cpu(W_hz_, dW_hz_, mW_hz_, vAW_hz_, lr, b1, b2, eps, step);
    adam_step_cpu(W_hn_, dW_hn_, mW_hn_, vAW_hn_, lr, b1, b2, eps, step);
    adam_step_cpu(b_r_,  db_r_,  mb_r_,  vAb_r_,  lr, b1, b2, eps, step);
    adam_step_cpu(b_z_,  db_z_,  mb_z_,  vAb_z_,  lr, b1, b2, eps, step);
    adam_step_cpu(b_in_, db_in_, mb_in_, vAb_in_, lr, b1, b2, eps, step);
    adam_step_cpu(b_hn_, db_hn_, mb_hn_, vAb_hn_, lr, b1, b2, eps, step);
}

void GRUCell::save_to(std::vector<uint8_t>& out) const {
    tensor_write(W_ir_, out); tensor_write(W_iz_, out); tensor_write(W_in_, out);
    tensor_write(W_hr_, out); tensor_write(W_hz_, out); tensor_write(W_hn_, out);
    tensor_write(b_r_, out);  tensor_write(b_z_, out);  tensor_write(b_in_, out); tensor_write(b_hn_, out);
}

void GRUCell::load_from(const uint8_t* data, size_t& offset, size_t size) {
    tensor_read(W_ir_, data, offset, size); tensor_read(W_iz_, data, offset, size); tensor_read(W_in_, data, offset, size);
    tensor_read(W_hr_, data, offset, size); tensor_read(W_hz_, data, offset, size); tensor_read(W_hn_, data, offset, size);
    tensor_read(b_r_, data, offset, size);  tensor_read(b_z_, data, offset, size);  tensor_read(b_in_, data, offset, size); tensor_read(b_hn_, data, offset, size);
    const int H = W_ir_.rows, I = W_ir_.cols;
    dW_ir_.resize(H, I); dW_iz_.resize(H, I); dW_in_.resize(H, I);
    dW_hr_.resize(H, H); dW_hz_.resize(H, H); dW_hn_.resize(H, H);
    db_r_.resize(H, 1); db_z_.resize(H, 1); db_in_.resize(H, 1); db_hn_.resize(H, 1);
    vW_ir_.resize(H, I); vW_iz_.resize(H, I); vW_in_.resize(H, I);
    vW_hr_.resize(H, H); vW_hz_.resize(H, H); vW_hn_.resize(H, H);
    vb_r_.resize(H, 1); vb_z_.resize(H, 1); vb_in_.resize(H, 1); vb_hn_.resize(H, 1);
    mW_ir_.resize(H, I); mW_iz_.resize(H, I); mW_in_.resize(H, I);
    mW_hr_.resize(H, H); mW_hz_.resize(H, H); mW_hn_.resize(H, H);
    mb_r_.resize(H, 1); mb_z_.resize(H, 1); mb_in_.resize(H, 1); mb_hn_.resize(H, 1);
    vAW_ir_.resize(H, I); vAW_iz_.resize(H, I); vAW_in_.resize(H, I);
    vAW_hr_.resize(H, H); vAW_hz_.resize(H, H); vAW_hn_.resize(H, H);
    vAb_r_.resize(H, 1); vAb_z_.resize(H, 1); vAb_in_.resize(H, 1); vAb_hn_.resize(H, 1);
    mW_ir_.zero(); mW_iz_.zero(); mW_in_.zero();
    mW_hr_.zero(); mW_hz_.zero(); mW_hn_.zero();
    mb_r_.zero(); mb_z_.zero(); mb_in_.zero(); mb_hn_.zero();
    vAW_ir_.zero(); vAW_iz_.zero(); vAW_in_.zero();
    vAW_hr_.zero(); vAW_hz_.zero(); vAW_hn_.zero();
    vAb_r_.zero(); vAb_z_.zero(); vAb_in_.zero(); vAb_hn_.zero();
    x_cache_.resize(I, 1);
    h_prev_cache_.resize(H, 1);
    r_.resize(H, 1); z_.resize(H, 1); n_.resize(H, 1); hn_pre_.resize(H, 1);
}

} // namespace brogameagent::nn
