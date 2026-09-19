#pragma once

// Payload cells and class handles for bro.ai.game.learn.
//
// Where the pre-transition binding pinned a JS reference on the wrapper
// object (`JS_SetPropertyStr(this_val, "__net", ...)`) to keep a net alive
// for a raw pointer, these cells hold an ev::Persistent instead: a
// Persistent is itself a GC root, so the anchoring is the cell's own
// business and nothing leaks onto the JS surface.

#include "host_ai_nn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <brogameagent/learn/gumbel.h>
#include <brogameagent/learn/search_trace.h>

#include <memory>
#include <vector>

namespace brogameagent::api {

namespace learn = brogameagent::learn;

inline constexpr uint32_t kHostReplayBufferTag   = 0x52504C42u;  // 'RPLB'
inline constexpr uint32_t kHostNeuralEvalTag     = 0x4E455641u;  // 'NEVA'
inline constexpr uint32_t kHostNeuralPriorTag    = 0x4E505249u;  // 'NPRI'
inline constexpr uint32_t kHostGumbelPriorTag    = 0x474D4250u;  // 'GMBP'
inline constexpr uint32_t kHostExItTrainerTag    = 0x45584954u;  // 'EXIT'
inline constexpr uint32_t kHostGenericBufferTag  = 0x47524250u;  // 'GRBP'
inline constexpr uint32_t kHostGenericTrainerTag = 0x47455854u;  // 'GEXT'
inline constexpr uint32_t kHostInferServerTag    = 0x494E4653u;  // 'INFS'
inline constexpr uint32_t kHostDirectBackendTag  = 0x44424B44u;  // 'DBKD'
inline constexpr uint32_t kHostServerBackendTag  = 0x53424B44u;  // 'SBKD'

struct HostReplayBuffer {
    uint32_t tag = kHostReplayBufferTag;
    std::shared_ptr<learn::ReplayBuffer> buf;
};

struct HostNeuralEvaluator {
    uint32_t tag = kHostNeuralEvalTag;
    std::shared_ptr<learn::NeuralEvaluator> eval;
    std::shared_ptr<nn::SingleHeroNet> netRef;
    std::shared_ptr<nn::WeightsHandle> handleRef;
};

struct HostNeuralPrior {
    uint32_t tag = kHostNeuralPriorTag;
    std::shared_ptr<learn::NeuralPrior> prior;
    std::shared_ptr<nn::SingleHeroNet> netRef;
    std::shared_ptr<nn::WeightsHandle> handleRef;
};

struct HostGumbelPrior {
    uint32_t tag = kHostGumbelPriorTag;
    std::shared_ptr<learn::GumbelNoisePrior> prior;
    std::shared_ptr<bgm::IPrior> innerRef;
};

struct HostExItTrainer {
    uint32_t tag = kHostExItTrainerTag;
    std::unique_ptr<learn::ExItTrainer> trainer;
    std::shared_ptr<nn::SingleHeroNet> netRef;
    std::shared_ptr<nn::WeightsHandle> handleRef;
    std::shared_ptr<learn::ReplayBuffer> bufRef;
};

struct HostGenericBuffer {
    uint32_t tag = kHostGenericBufferTag;
    std::shared_ptr<learn::GenericReplayBuffer> buf;
};

struct HostGenericTrainer {
    uint32_t tag = kHostGenericTrainerTag;
    std::unique_ptr<learn::GenericExItTrainer> trainer;
    // Exactly one of the two is set after setNet(); the other stays null.
    std::shared_ptr<nn::PolicyValueNet> pvnRef;
    std::shared_ptr<nn::SingleHeroNetTX> txRef;
    std::shared_ptr<nn::WeightsHandle> handleRef;
    std::shared_ptr<learn::GenericReplayBuffer> bufRef;
};

struct HostInferenceServer {
    uint32_t tag = kHostInferServerTag;
    std::unique_ptr<learn::BatchedInferenceServer> server;
    std::shared_ptr<learn::BatchedNet> netRef;
};

struct HostDirectBackend {
    uint32_t tag = kHostDirectBackendTag;
    std::unique_ptr<learn::IInferenceBackend> backend;
    std::shared_ptr<learn::BatchedNet> netRef;
};

struct HostServerBackend {
    uint32_t tag = kHostServerBackendTag;
    std::unique_ptr<learn::IInferenceBackend> backend;
    std::shared_ptr<learn::BatchedNet> netRef;
    ev::Persistent serverRef;
};

inline HostReplayBuffer* unwrapReplayBuffer(Value v) {
    return unwrapTagged<HostReplayBuffer, kHostReplayBufferTag>(v);
}
inline HostNeuralEvaluator* unwrapNeuralEvaluator(Value v) {
    return unwrapTagged<HostNeuralEvaluator, kHostNeuralEvalTag>(v);
}
inline HostNeuralPrior* unwrapNeuralPrior(Value v) {
    return unwrapTagged<HostNeuralPrior, kHostNeuralPriorTag>(v);
}
inline HostGumbelPrior* unwrapGumbelPrior(Value v) {
    return unwrapTagged<HostGumbelPrior, kHostGumbelPriorTag>(v);
}
inline HostExItTrainer* unwrapExItTrainer(Value v) {
    return unwrapTagged<HostExItTrainer, kHostExItTrainerTag>(v);
}
inline HostGenericBuffer* unwrapGenericBuffer(Value v) {
    return unwrapTagged<HostGenericBuffer, kHostGenericBufferTag>(v);
}
inline HostGenericTrainer* unwrapGenericTrainer(Value v) {
    return unwrapTagged<HostGenericTrainer, kHostGenericTrainerTag>(v);
}
inline HostInferenceServer* unwrapInferenceServer(Value v) {
    return unwrapTagged<HostInferenceServer, kHostInferServerTag>(v);
}
inline HostDirectBackend* unwrapDirectBackend(Value v) {
    return unwrapTagged<HostDirectBackend, kHostDirectBackendTag>(v);
}
inline HostServerBackend* unwrapServerBackend(Value v) {
    return unwrapTagged<HostServerBackend, kHostServerBackendTag>(v);
}

extern HostClass g_replayBufferClass;
extern HostClass g_neuralEvaluatorClass;
extern HostClass g_neuralPriorClass;
extern HostClass g_gumbelPriorClass;
extern HostClass g_exitTrainerClass;
extern HostClass g_genericBufferClass;
extern HostClass g_genericTrainerClass;
extern HostClass g_inferenceServerClass;
extern HostClass g_directBackendClass;
extern HostClass g_serverBackendClass;

/// PolicyValueNet or SingleHeroNetTX — both are learn::BatchedNet. A plain
/// SingleHeroNet is not batched-inference capable and answers null.
std::shared_ptr<learn::BatchedNet> batchedNetShared(Value v);

/// Shared helpers for the two learn translation units.
Value makeSituationObject(const learn::Situation& s);
bool readSituationValue(Value v, learn::Situation& out);
Value makeGenericSituationObject(const learn::GenericSituation& s);
bool readGenericSituationValue(Value v, learn::GenericSituation& out);

void ensureAILearnCoreClassesInstalled();
void ensureAILearnGenericClassesInstalled();
void installAILearnCore(ObjectBuilder& learnNs);
void installAILearnGeneric(ObjectBuilder& learnNs);

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
