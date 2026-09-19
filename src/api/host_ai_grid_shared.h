#pragma once

// Payload cells and helpers for bro.ai.game.grid — the grid-world /
// platformer training kit.
//
// grid/harness.h embeds a learn::GenericTrainer, which pulls brotensor, so
// the whole surface is gated on the neural layer exactly as it was before
// the transition.

#include "host_ai_learn_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <brogameagent/grid/bc_ingest.h>
#include <brogameagent/grid/best_crop.h>
#include <brogameagent/grid/failure_tape.h>
#include <brogameagent/grid/frame_stack.h>
#include <brogameagent/grid/generic_recorder.h>
#include <brogameagent/grid/harness.h>
#include <brogameagent/grid/obs_window.h>
#include <brogameagent/grid/shaping.h>

#include <any>
#include <memory>
#include <random>
#include <vector>

namespace brogameagent::api {

namespace grid = brogameagent::grid;

inline constexpr uint32_t kHostObsWindowTag     = 0x4F425357u;  // 'OBSW'
inline constexpr uint32_t kHostFrameStackTag    = 0x46525354u;  // 'FRST'
inline constexpr uint32_t kHostFailureTapeTag   = 0x46544150u;  // 'FTAP'
inline constexpr uint32_t kHostBestCropTag      = 0x42435250u;  // 'BCRP'
inline constexpr uint32_t kHostShaperTag        = 0x50534852u;  // 'PSHR'
inline constexpr uint32_t kHostStallTag         = 0x53544C44u;  // 'STLD'
inline constexpr uint32_t kHostGridRecorderTag  = 0x47524543u;  // 'GREC'
inline constexpr uint32_t kHostGridReaderTag    = 0x47524452u;  // 'GRDR'
inline constexpr uint32_t kHostGridTrainerTag   = 0x47545252u;  // 'GTRR'

struct HostObsWindow {
    uint32_t tag = kHostObsWindowTag;
    std::unique_ptr<grid::ObsWindow> win;
    // The JS callbacks the spec's samplers call. A Persistent is a GC root,
    // so holding one here is all the anchoring the old gc_mark achieved.
    ev::Persistent tileFn;
    std::vector<ev::Persistent> enumerateFns;
    std::vector<ev::Persistent> sampleFns;
};

struct HostFrameStack {
    uint32_t tag = kHostFrameStackTag;
    std::unique_ptr<grid::FrameStack> fs;
};

struct HostFailureTape {
    uint32_t tag = kHostFailureTapeTag;
    std::unique_ptr<grid::FailureTape> tape;
};

struct HostBestCrop {
    uint32_t tag = kHostBestCropTag;
    std::unique_ptr<grid::BestCrop> pool;
    std::mt19937_64 rng{0xC0DE1234ULL};
};

struct HostShaper {
    uint32_t tag = kHostShaperTag;
    std::unique_ptr<grid::PotentialShaper> sh;
};

struct HostStallDetector {
    uint32_t tag = kHostStallTag;
    std::unique_ptr<grid::StallDetector> det;
};

struct HostGridRecorder {
    uint32_t tag = kHostGridRecorderTag;
    std::unique_ptr<grid::GenericRecorder> rec;
    std::vector<grid::FieldDef> roster, frame, events;
};

struct HostGridReader {
    uint32_t tag = kHostGridReaderTag;
    std::unique_ptr<grid::GenericReplayReader> rr;
};

struct HostGridTrainer {
    uint32_t tag = kHostGridTrainerTag;
    std::unique_ptr<grid::GridTrainer> tr;
};

inline HostObsWindow* unwrapObsWindow(Value v) {
    return unwrapTagged<HostObsWindow, kHostObsWindowTag>(v);
}
inline HostFrameStack* unwrapFrameStack(Value v) {
    return unwrapTagged<HostFrameStack, kHostFrameStackTag>(v);
}
inline HostFailureTape* unwrapFailureTape(Value v) {
    return unwrapTagged<HostFailureTape, kHostFailureTapeTag>(v);
}
inline HostBestCrop* unwrapBestCrop(Value v) {
    return unwrapTagged<HostBestCrop, kHostBestCropTag>(v);
}
inline HostShaper* unwrapShaper(Value v) {
    return unwrapTagged<HostShaper, kHostShaperTag>(v);
}
inline HostStallDetector* unwrapStallDetector(Value v) {
    return unwrapTagged<HostStallDetector, kHostStallTag>(v);
}
inline HostGridRecorder* unwrapGridRecorder(Value v) {
    return unwrapTagged<HostGridRecorder, kHostGridRecorderTag>(v);
}
inline HostGridReader* unwrapGridReader(Value v) {
    return unwrapTagged<HostGridReader, kHostGridReaderTag>(v);
}
inline HostGridTrainer* unwrapGridTrainer(Value v) {
    return unwrapTagged<HostGridTrainer, kHostGridTrainerTag>(v);
}

extern HostClass g_obsWindowClass;
extern HostClass g_frameStackClass;
extern HostClass g_failureTapeClass;
extern HostClass g_bestCropClass;
extern HostClass g_shaperClass;
extern HostClass g_stallDetectorClass;
extern HostClass g_gridRecorderClass;
extern HostClass g_gridReaderClass;
extern HostClass g_gridTrainerClass;

/// An opaque JS env snapshot travelling through std::any. ev::Persistent is
/// copyable and is itself a GC root, so it needs no holder of its own.
using JsSnapshot = ev::Persistent;

std::vector<float> readFloats(Value v);
std::string readStringPropOr(Value obj, const char* key, const char* def);
std::vector<grid::FailureStep> readFailureTail(Value arr);

/// A GenericSituation as bro.ai.game.grid spells it (the same shape
/// bro.ai.game.learn uses).
learn::GenericSituation situationFromValue(Value v);

void ensureAIGridClassesInstalled();
void ensureAIGridRecordingClassesInstalled();
void installAIGridCore(ObjectBuilder& gridNs);
void installAIGridRecording(ObjectBuilder& gridNs);

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
