#include "brogameagent/grid/harness.h"

#include <chrono>
#include <cstdio>
#include <fstream>

namespace brogameagent::grid {

namespace {

// Ring sizes: powers of two not strictly required for the seq-no MPSC
// (modulo works on any size), but we keep generous capacity so producers
// rarely see the "full → drop" path.
constexpr size_t kSituationRingCap = 1u << 14;   // 16K
constexpr size_t kEpisodeRingCap   = 1u << 10;   // 1K

bool write_blob_to_file(const std::string& path, const std::vector<uint8_t>& blob) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
    return f.good();
}

bool copy_file(const std::string& src, const std::string& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << in.rdbuf();
    return out.good();
}

} // namespace

GridTrainer::GridTrainer(GridTrainerConfig cfg)
    : cfg_(std::move(cfg)), buffer_(static_cast<size_t>(std::max(1, cfg_.buffer_capacity)))
{
    net_.init(cfg_.net);
    trainer_.set_net(&net_);
    trainer_.set_buffer(&buffer_);
    trainer_.set_weights_handle(&handle_);
    trainer_.set_config(cfg_.trainer);

    if (cfg_.best_window > 0) ret_window_.assign(static_cast<size_t>(cfg_.best_window), 0.0f);

    sit_ring_.init(kSituationRingCap);
    ep_ring_.init(kEpisodeRingCap);

    // Publish initial weights so consumers can snapshot before any SGD.
    handle_.publish(net_.save(), 1);
}

GridTrainer::~GridTrainer() {
    stop();
}

void GridTrainer::ingest_situation(learn::GenericSituation s) {
    // Drop on overflow rather than blocking — producers should be tolerant.
    sit_ring_.push(std::move(s));
}

void GridTrainer::ingest_episode(EpisodeSummary e) {
    ep_ring_.push(std::move(e));
}

void GridTrainer::warmup_with(const std::vector<learn::GenericSituation>& sits) {
    for (const auto& s : sits) buffer_.push(s);
}

void GridTrainer::start() {
    if (running_.load()) return;
    stop_flag_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    loop_thread_ = std::thread([this] { run_loop(); });
}

void GridTrainer::stop() {
    if (!running_.load()) return;
    stop_flag_.store(true, std::memory_order_release);
    if (loop_thread_.joinable()) loop_thread_.join();
    running_.store(false, std::memory_order_release);
}

void GridTrainer::run_loop() {
    while (!stop_flag_.load(std::memory_order_acquire)) {
        int drained = drain_situations(cfg_.ingest_burst);
        drain_episodes();
        bool did_work = drained > 0;
        for (int i = 0; i < cfg_.steps_per_tick; ++i) {
            if (buffer_.size() == 0) break;
            if (train_step()) did_work = true;
        }
        if (!did_work) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    // Final drain so producers' last messages aren't lost.
    drain_situations(static_cast<int>(kSituationRingCap));
    drain_episodes();
}

void GridTrainer::step_sync(int sgd_steps) {
    drain_situations(static_cast<int>(kSituationRingCap));
    drain_episodes();
    for (int i = 0; i < sgd_steps; ++i) {
        if (buffer_.size() == 0) break;
        train_step();
    }
}

int GridTrainer::drain_situations(int max) {
    int n = 0;
    learn::GenericSituation s;
    while (n < max && sit_ring_.pop(s)) {
        buffer_.push(std::move(s));
        ++n;
    }
    return n;
}

int GridTrainer::drain_episodes() {
    int n = 0;
    EpisodeSummary e;
    while (ep_ring_.pop(e)) {
        ++n;

        // BestCrop ingest: only successful episodes with a snapshot.
        if (!e.failed && e.start_snapshot.has_value()) {
            best_.push(e.start_snapshot, e.action_prefix, e.total_return, e.depth);
        }

        // FailureTape: only failed episodes with a tail.
        if (e.failed && !e.failure_tail.empty()) {
            tape_.record_failure(e.failure_tail);
        }

        // Trailing-mean tracker.
        if (!ret_window_.empty()) {
            ret_window_[static_cast<size_t>(ret_write_idx_)] = e.total_return;
            ret_write_idx_ = (ret_write_idx_ + 1) % static_cast<int>(ret_window_.size());
            if (ret_filled_ < static_cast<int>(ret_window_.size())) ++ret_filled_;
        }

        ++episodes_seen_;
        float mean = 0.0f;
        if (ret_filled_ > 0) {
            for (int i = 0; i < ret_filled_; ++i) mean += ret_window_[static_cast<size_t>(i)];
            mean /= static_cast<float>(ret_filled_);
        }
        GridEvent ev;
        ev.kind          = GridEvent::Kind::EpisodeIngested;
        ev.episode_count = episodes_seen_;
        ev.mean_return   = mean;
        emit(std::move(ev));
    }
    return n;
}

bool GridTrainer::train_step() {
    int prev_publishes = trainer_.total_publishes();
    auto step = trainer_.step();
    (void)step;
    bool published = trainer_.total_publishes() > prev_publishes;
    if (published) {
        GridEvent ev;
        ev.kind        = GridEvent::Kind::WeightsUpdated;
        ev.version     = handle_.version();
        ev.total_steps = trainer_.total_steps();
        emit(std::move(ev));

        // Checkpoint ring + best rotation.
        if (!cfg_.ckpt_dir.empty()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "ckpt_%04d.bin", ckpt_ring_idx_);
            std::string slot_path = cfg_.ckpt_dir + "/" + buf;
            ckpt_ring_idx_ = (ckpt_ring_idx_ + 1) %
                std::max(1, cfg_.ckpt_ring_size);

            auto blob = net_.save();
            write_blob_to_file(slot_path, blob);

            // Best rotation: use trailing-mean return across observed
            // episodes. Only rotates after the window is filled.
            if (ret_filled_ > 0 &&
                ret_filled_ >= static_cast<int>(ret_window_.size())) {
                float mean = 0.0f;
                for (float v : ret_window_) mean += v;
                mean /= static_cast<float>(ret_window_.size());
                if (!best_initialized_ || mean > best_mean_) {
                    best_initialized_ = true;
                    best_mean_        = mean;
                    std::string best_path = cfg_.ckpt_dir + "/best.bin";
                    copy_file(slot_path, best_path);
                    GridEvent be;
                    be.kind        = GridEvent::Kind::BestRotated;
                    be.path        = best_path;
                    be.mean_return = mean;
                    emit(std::move(be));
                }
            }
        }
    }
    return published;
}

void GridTrainer::emit(GridEvent e) {
    if (ev_cb_) ev_cb_(e);
    // Spin briefly to acquire the events lock; the only contention is
    // between this thread and a polling caller, both very short.
    bool expected = false;
    while (!events_lock_.compare_exchange_weak(expected, true,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
        expected = false;
    }
    events_.push_back(std::move(e));
    events_lock_.store(false, std::memory_order_release);
}

std::vector<GridEvent> GridTrainer::poll_events() {
    std::vector<GridEvent> out;
    bool expected = false;
    while (!events_lock_.compare_exchange_weak(expected, true,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
        expected = false;
    }
    out.swap(events_);
    events_lock_.store(false, std::memory_order_release);
    return out;
}

GridTrainerStats GridTrainer::stats() const {
    GridTrainerStats s;
    s.total_steps         = trainer_.total_steps();
    s.total_publishes     = trainer_.total_publishes();
    s.episodes_ingested   = episodes_seen_;
    s.buffer_size         = static_cast<int>(buffer_.size());
    s.running             = running_.load(std::memory_order_acquire);
    if (ret_filled_ > 0) {
        float m = 0.0f;
        for (int i = 0; i < ret_filled_; ++i) m += ret_window_[static_cast<size_t>(i)];
        s.trailing_mean_return = m / static_cast<float>(ret_filled_);
    }
    s.best_mean_return = best_mean_;
    return s;
}

} // namespace brogameagent::grid
