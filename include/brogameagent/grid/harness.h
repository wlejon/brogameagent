#pragma once

#include "brogameagent/grid/best_crop.h"
#include "brogameagent/grid/failure_tape.h"
#include "brogameagent/learn/generic_replay_buffer.h"
#include "brogameagent/learn/generic_trainer.h"
#include "brogameagent/nn/net.h"               // WeightsHandle
#include "brogameagent/nn/policy_value_net.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace brogameagent::grid {

// ─── EpisodeSummary ───────────────────────────────────────────────────────
//
// One completed episode's metadata, fed in alongside the situations its
// search produced. Drives BestCrop ingestion, FailureTape recording, and
// the trailing-mean tracker that decides whether a checkpoint is the new
// best.

struct EpisodeSummary {
    float            total_return = 0.0f;
    int              depth        = 0;
    bool             failed       = false;
    // Optional: BestCrop entry (inserted only when start_snapshot is set
    // AND the episode wasn't a failure).
    std::any         start_snapshot;
    std::vector<int> action_prefix;
    // Optional: FailureTape tail (inserted only when failed = true).
    std::vector<FailureStep> failure_tail;
};

// ─── Events ───────────────────────────────────────────────────────────────

struct GridEvent {
    enum class Kind {
        WeightsUpdated,    // version, total_steps
        BestRotated,       // path, mean_return
        EpisodeIngested,   // episode_count, mean_return
    };
    Kind        kind;
    uint64_t    version       = 0;
    int         total_steps   = 0;
    int         episode_count = 0;
    float       mean_return   = 0.0f;
    std::string path;
};

// ─── GridTrainer ──────────────────────────────────────────────────────────
//
// Owns: PolicyValueNet, GenericReplayBuffer, GenericExItTrainer, WeightsHandle,
// optional ckpt-ring writer, and trailing-mean best-tracker. Producers feed
// situations and episode summaries via ingest_*; the trainer thread (when
// running) drains these into the buffer + bookkeeping, runs SGD, and emits
// events.
//
// Two operating modes:
//   - Async (start()/stop()): one trainer thread, lock-free MPSC ring of
//     pending situations + episodes, polled-and-cleared at each loop tick.
//   - Sync (step_sync()): no thread; caller drives drain + train cadence.
//     Use for tests and single-thread workflows.
//
// Events are queued and visible via poll_events() (non-blocking). An
// optional callback set via on_event() fires from the trainer thread —
// the callback must be thread-safe and short.

struct GridTrainerConfig {
    nn::PolicyValueNet::Config         net{};
    int                                buffer_capacity = 65536;
    learn::GenericTrainerConfig        trainer{};

    // Optional: ckpt files written every `trainer.publish_every` SGD steps,
    // rotating through `ckpt_ring_size` slots in `ckpt_dir`. The "best"
    // ckpt is copied to <ckpt_dir>/best.bin when trailing mean rises.
    std::string ckpt_dir;
    int         ckpt_ring_size = 8;
    int         best_window    = 50;

    // Best/worst-case overflow knobs.
    int  ingest_burst    = 64;     // max situations drained per trainer tick
    int  steps_per_tick  = 1;      // SGD steps per trainer tick
};

struct GridTrainerStats {
    int      total_steps           = 0;
    int      total_publishes       = 0;
    int      episodes_ingested     = 0;
    float    trailing_mean_return  = 0.0f;
    float    best_mean_return      = 0.0f;
    int      buffer_size           = 0;
    bool     running               = false;
};

class GridTrainer {
public:
    explicit GridTrainer(GridTrainerConfig cfg);
    ~GridTrainer();
    GridTrainer(const GridTrainer&) = delete;
    GridTrainer& operator=(const GridTrainer&) = delete;

    // ─── Producer side (callable from any thread) ─────────────────────
    void ingest_situation(learn::GenericSituation s);
    void ingest_episode(EpisodeSummary e);

    // Convenience: BC warmup. Pushes situations directly to the buffer
    // without going through the ring (synchronous; call before start()).
    void warmup_with(const std::vector<learn::GenericSituation>& sits);

    // ─── Lifecycle ────────────────────────────────────────────────────
    void start();
    void stop();
    bool running() const { return running_.load(std::memory_order_acquire); }

    // Synchronous tick: drain rings, run sgd_steps, maybe publish/ckpt.
    // Safe to call only when not running().
    void step_sync(int sgd_steps = 1);

    // ─── Accessors ────────────────────────────────────────────────────
    nn::WeightsHandle&             weights()      { return handle_; }
    nn::PolicyValueNet&            net()          { return net_; }
    learn::GenericReplayBuffer&    buffer()       { return buffer_; }
    BestCrop&                      best_crop()    { return best_; }
    FailureTape&                   failure_tape() { return tape_; }

    GridTrainerStats stats() const;

    // ─── Events ───────────────────────────────────────────────────────
    std::vector<GridEvent> poll_events();
    void on_event(std::function<void(const GridEvent&)> cb) { ev_cb_ = std::move(cb); }

private:
    // Trainer-thread loop.
    void run_loop();

    // Drain at most `max` situations from the ring into the buffer,
    // tracking how many were actually consumed.
    int drain_situations(int max);
    int drain_episodes();

    // One SGD step + bookkeeping. Returns true if a publish happened.
    bool train_step();

    // Push one event to the queue (and the optional callback).
    void emit(GridEvent e);

    // ─── State ────────────────────────────────────────────────────────
    GridTrainerConfig          cfg_;
    nn::PolicyValueNet         net_;
    learn::GenericReplayBuffer buffer_;
    learn::GenericExItTrainer  trainer_;
    nn::WeightsHandle          handle_;
    BestCrop                   best_;
    FailureTape                tape_;

    // Trailing-mean tracker.
    std::vector<float> ret_window_;
    int                ret_write_idx_ = 0;
    int                ret_filled_    = 0;
    float              best_mean_     = 0.0f;
    bool               best_initialized_ = false;
    int                ckpt_ring_idx_ = 0;
    int                episodes_seen_ = 0;

    // Async loop control.
    std::atomic<bool>   running_{false};
    std::atomic<bool>   stop_flag_{false};
    std::thread         loop_thread_;

    // Event surface.
    std::vector<GridEvent>                          events_;
    std::function<void(const GridEvent&)>           ev_cb_;
    // events_ is a single-consumer queue but producer is the trainer
    // thread; use an atomic flag so poll_events() can swap-and-clear
    // without a mutex.
    std::atomic<bool>                               events_lock_{false};

    // ─── MPSC rings (Vyukov-style, lock-free) ─────────────────────────
    template <class T>
    struct MpscRing {
        struct Slot {
            std::atomic<uint64_t> seq{0};
            T                     value{};
        };
        std::unique_ptr<Slot[]>     slots;
        size_t                      cap = 0;
        std::atomic<uint64_t>       head{0};   // consumer
        std::atomic<uint64_t>       tail{0};   // producer

        void init(size_t capacity) {
            cap = capacity;
            slots.reset(new Slot[capacity]);
            for (size_t i = 0; i < capacity; ++i)
                slots[i].seq.store(static_cast<uint64_t>(i), std::memory_order_relaxed);
        }
        // Returns false if the ring is full (drop on overflow).
        bool push(T v) {
            uint64_t pos = tail.load(std::memory_order_relaxed);
            for (;;) {
                Slot& s = slots[pos % cap];
                uint64_t seq = s.seq.load(std::memory_order_acquire);
                int64_t  diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
                if (diff == 0) {
                    if (tail.compare_exchange_weak(pos, pos + 1,
                                                   std::memory_order_relaxed)) {
                        s.value = std::move(v);
                        s.seq.store(pos + 1, std::memory_order_release);
                        return true;
                    }
                } else if (diff < 0) {
                    return false;     // full
                } else {
                    pos = tail.load(std::memory_order_relaxed);
                }
            }
        }
        bool pop(T& out) {
            uint64_t pos = head.load(std::memory_order_relaxed);
            Slot& s = slots[pos % cap];
            uint64_t seq = s.seq.load(std::memory_order_acquire);
            int64_t  diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                head.store(pos + 1, std::memory_order_relaxed);
                out = std::move(s.value);
                s.seq.store(pos + cap, std::memory_order_release);
                return true;
            }
            return false;            // empty (or not yet ready)
        }
    };

    MpscRing<learn::GenericSituation> sit_ring_;
    MpscRing<EpisodeSummary>          ep_ring_;
};

} // namespace brogameagent::grid
