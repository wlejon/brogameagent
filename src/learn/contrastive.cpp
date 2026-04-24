#include "brogameagent/learn/contrastive.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace brogameagent::learn {

using nn::Tensor;

float infonce_loss(const std::vector<Tensor>& anchors,
                   const std::vector<Tensor>& positives,
                   std::vector<Tensor>& dAnchors,
                   std::vector<Tensor>& dPositives,
                   float temperature) {
    const int B = static_cast<int>(anchors.size());
    assert(B > 0);
    assert(static_cast<int>(positives.size()) == B);
    const int D = anchors[0].size();
    const float tau = temperature;
    const float eps = 1e-8f;

    // Precompute norms and normalized vectors.
    std::vector<float> na(B, 0.0f), np(B, 0.0f);
    std::vector<Tensor> ahat(B, Tensor::vec(D));
    std::vector<Tensor> phat(B, Tensor::vec(D));
    for (int i = 0; i < B; ++i) {
        float na2 = 0.0f, np2 = 0.0f;
        for (int k = 0; k < D; ++k) { na2 += anchors[i][k] * anchors[i][k]; np2 += positives[i][k] * positives[i][k]; }
        na[i] = std::sqrt(na2 + eps);
        np[i] = std::sqrt(np2 + eps);
        const float inva = 1.0f / na[i];
        const float invp = 1.0f / np[i];
        for (int k = 0; k < D; ++k) {
            ahat[i][k] = anchors[i][k]   * inva;
            phat[i][k] = positives[i][k] * invp;
        }
    }

    // Score matrix s(i,j) = ahat_i . phat_j / tau.
    std::vector<float> s(static_cast<size_t>(B) * B, 0.0f);
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < B; ++j) {
            float d = 0.0f;
            for (int k = 0; k < D; ++k) d += ahat[i][k] * phat[j][k];
            s[static_cast<size_t>(i) * B + j] = d / tau;
        }
    }

    // Row-softmax q(i,j), loss = -log q(i,i) averaged.
    std::vector<float> q(static_cast<size_t>(B) * B, 0.0f);
    float loss = 0.0f;
    for (int i = 0; i < B; ++i) {
        float m = -1e30f;
        for (int j = 0; j < B; ++j) if (s[static_cast<size_t>(i)*B + j] > m) m = s[static_cast<size_t>(i)*B + j];
        float sum = 0.0f;
        for (int j = 0; j < B; ++j) {
            const float e = std::exp(s[static_cast<size_t>(i)*B + j] - m);
            q[static_cast<size_t>(i)*B + j] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < B; ++j) q[static_cast<size_t>(i)*B + j] *= inv;
        const float p_ii = q[static_cast<size_t>(i)*B + i];
        loss -= std::log(p_ii > 1e-12f ? p_ii : 1e-12f);
    }
    const float inv_B = 1.0f / static_cast<float>(B);
    loss *= inv_B;

    // dL/ds(i,j) = (q(i,j) - [i==j]) / B.
    // For each i:
    //   dL/da_i = sum_j dL/ds(i,j) * [ phat_j / (na_i tau) - (s(i,j)/na_i) * ahat_i ]
    // For each j:
    //   dL/dp_j = sum_i dL/ds(i,j) * [ ahat_i / (np_j tau) - (s(i,j)/np_j) * phat_j ]
    dAnchors.assign(B, Tensor::vec(D));
    dPositives.assign(B, Tensor::vec(D));
    for (int i = 0; i < B; ++i) {
        const float inv_na = 1.0f / na[i];
        float sum_ds_sij = 0.0f;
        for (int j = 0; j < B; ++j) {
            const float ds = (q[static_cast<size_t>(i)*B + j] - (i == j ? 1.0f : 0.0f)) * inv_B;
            const float sij = s[static_cast<size_t>(i)*B + j];
            const float coef_p = ds * (inv_na / tau);
            for (int k = 0; k < D; ++k) dAnchors[i][k] += coef_p * phat[j][k];
            sum_ds_sij += ds * sij;
        }
        // subtract (sum_j ds*s) / na_i * ahat_i
        const float coef_a = -sum_ds_sij * inv_na;
        for (int k = 0; k < D; ++k) dAnchors[i][k] += coef_a * ahat[i][k];
    }
    for (int j = 0; j < B; ++j) {
        const float inv_np = 1.0f / np[j];
        float sum_ds_sij = 0.0f;
        for (int i = 0; i < B; ++i) {
            const float ds = (q[static_cast<size_t>(i)*B + j] - (i == j ? 1.0f : 0.0f)) * inv_B;
            const float sij = s[static_cast<size_t>(i)*B + j];
            const float coef_a = ds * (inv_np / tau);
            for (int k = 0; k < D; ++k) dPositives[j][k] += coef_a * ahat[i][k];
            sum_ds_sij += ds * sij;
        }
        const float coef_p = -sum_ds_sij * inv_np;
        for (int k = 0; k < D; ++k) dPositives[j][k] += coef_p * phat[j][k];
    }

    return loss;
}

} // namespace brogameagent::learn
