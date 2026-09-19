// bro.ai.game.grid — the recording/training half: GenericRecorder,
// GenericReplayReader and GridTrainer.
//
// Ported from ai_grid_bindings.cpp.

#include "host_ai_grid_shared.h"

#ifdef BROGAMEAGENT_HAS_NN

#include <variant>

namespace brogameagent::api {

HostClass g_gridRecorderClass;
HostClass g_gridReaderClass;
HostClass g_gridTrainerClass;

namespace {

std::vector<grid::FieldDef> readSchema(Value arr) {
    std::vector<grid::FieldDef> out;
    if (!ev::isObject(arr)) return out;
    ev::Persistent root(arr);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return out;
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Value e = ev::getElement(root.get(), i);
        if (!ev::isObject(e)) continue;
        ev::Persistent eo(e);
        grid::FieldDef fd;
        fd.name = readStringPropOr(eo.get(), "name", "");
        const std::string t = readStringPropOr(eo.get(), "type", "f32");
        if (t == "i32") fd.type = grid::FieldType::I32;
        else if (t == "i64") fd.type = grid::FieldType::I64;
        else if (t == "f64") fd.type = grid::FieldType::F64;
        else fd.type = grid::FieldType::F32;
        out.push_back(std::move(fd));
    }
    return out;
}

grid::Row rowFromValue(Value arr, const std::vector<grid::FieldDef>& schema) {
    grid::Row row;
    if (!ev::isObject(arr)) return row;
    ev::Persistent root(arr);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return row;
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
    row.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Value e = ev::getElement(root.get(), i);
        const grid::FieldType t =
            i < schema.size() ? schema[i].type : grid::FieldType::F32;
        const double d = (ev::isUndefined(e) || ev::isObject(e)) ? 0.0 : ev::toDouble(e);
        switch (t) {
            case grid::FieldType::I32: row.push_back(static_cast<int32_t>(d)); break;
            case grid::FieldType::I64: row.push_back(static_cast<int64_t>(ev::isBigInt(e)
                                                        ? static_cast<int64_t>(ev::toInt64(e))
                                                        : static_cast<int64_t>(d)));
                                       break;
            case grid::FieldType::F32: row.push_back(static_cast<float>(d)); break;
            case grid::FieldType::F64: row.push_back(d); break;
        }
    }
    return row;
}

Value fieldValueToJs(const grid::FieldValue& v) {
    if (auto* p = std::get_if<int32_t>(&v)) return ev::fromDouble(*p);
    if (auto* p = std::get_if<int64_t>(&v)) return makeBigIntValue(static_cast<uint64_t>(*p));
    if (auto* p = std::get_if<float>(&v)) return ev::fromDouble(*p);
    return ev::fromDouble(std::get<double>(v));
}

Value rowToJs(const grid::Row& row) {
    return hostArrayOf(row.size(), [&](size_t i) { return fieldValueToJs(row[i]); });
}

std::vector<grid::Row> rowsFromValue(Value arr, const std::vector<grid::FieldDef>& schema) {
    std::vector<grid::Row> out;
    if (!ev::isObject(arr)) return out;
    ev::Persistent root(arr);
    Value lenV = ev::getProperty(root.get(), "length");
    if (ev::isUndefined(lenV) || ev::isObject(lenV)) return out;
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        out.push_back(rowFromValue(ev::getElement(root.get(), i), schema));
    }
    return out;
}

// ── GenericRecorder ───────────────────────────────────────────────────────

void decorateGridRecorder(ObjectBuilder& b) {
    b.accessor("frameCount", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridRecorder(self);
        return ev::fromDouble(d && d->rec ? static_cast<double>(d->rec->frame_count()) : 0.0);
    });
    b.def("open", 5, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridRecorder(self);
        if (!d || !d->rec || a.size() < 5) return ev::fromBool(false);
        ev::Persistent schemas(a[4]);
        d->roster = readSchema(ev::getProperty(schemas.get(), "roster"));
        d->frame = readSchema(ev::getProperty(schemas.get(), "frame"));
        d->events = readSchema(ev::getProperty(schemas.get(), "events"));
        const bool ok = d->rec->open(strAt(a, 0), readSeedArg(a[1], 0), readSeedArg(a[2], 0),
                                     static_cast<float>(numAt(a, 3)), d->roster, d->frame,
                                     d->events);
        return ev::fromBool(ok);
    });
    b.def("isOpen", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridRecorder(self);
        return ev::fromBool(d && d->rec && d->rec->is_open());
    });
    b.def("writeRoster", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridRecorder(self);
        if (!d || !d->rec || a.empty()) return ev::undefined();
        d->rec->write_roster(rowsFromValue(a[0], d->roster));
        return ev::undefined();
    });
    b.def("recordFrame", 4, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridRecorder(self);
        if (!d || !d->rec || a.size() < 3) return ev::undefined();
        auto rows = rowsFromValue(a[2], d->frame);
        std::vector<grid::Row> events;
        if (a.size() >= 4) events = rowsFromValue(a[3], d->events);
        d->rec->record_frame(readSeedArg(a[0], 0), static_cast<float>(numAt(a, 1)), rows, events);
        return ev::undefined();
    });
    b.def("close", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridRecorder(self);
        return ev::fromBool(d && d->rec ? d->rec->close() : false);
    });
}

// ── GenericReplayReader ───────────────────────────────────────────────────

void decorateGridReader(ObjectBuilder& b) {
    b.accessor("frameCount", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridReader(self);
        return ev::fromDouble(d && d->rr ? static_cast<double>(d->rr->frame_count()) : 0.0);
    });
    b.accessor("errorMessage", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridReader(self);
        return ev::fromUtf8(d && d->rr ? d->rr->error_message() : std::string());
    });
    b.def("open", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridReader(self);
        if (!d || !d->rr || a.empty()) return ev::fromBool(false);
        return ev::fromBool(d->rr->open(strAt(a, 0)));
    });
    b.def("frame", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridReader(self);
        ObjectBuilder o;
        if (!d || !d->rr) return o.get();
        grid::GenericFrame fr = d->rr->frame(static_cast<size_t>(i32At(a, 0)));
        o.set("stepIdx", makeBigIntValue(fr.step_idx));
        o.set("elapsed", static_cast<double>(fr.elapsed));
        o.set("rows", hostArrayOf(fr.rows.size(), [&](size_t i) { return rowToJs(fr.rows[i]); }));
        o.set("events",
              hostArrayOf(fr.events.size(), [&](size_t i) { return rowToJs(fr.events[i]); }));
        return o.get();
    });
    b.def("trajectory", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridReader(self);
        if (!d || !d->rr || a.size() < 2) {
            return hostArrayOf(0, [](size_t) { return ev::undefined(); });
        }
        auto vals = d->rr->trajectory(static_cast<size_t>(i32At(a, 0)), strAt(a, 1));
        return hostArrayOf(vals.size(), [&](size_t i) { return fieldValueToJs(vals[i]); });
    });
}

// ── GridTrainer ───────────────────────────────────────────────────────────

void decorateGridTrainer(ObjectBuilder& b) {
    b.accessor("running", [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridTrainer(self);
        return ev::fromBool(d && d->tr ? d->tr->running() : false);
    });
    b.def("ingestSituation", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (!d || !d->tr || a.empty()) return ev::undefined();
        d->tr->ingest_situation(situationFromValue(a[0]));
        return ev::undefined();
    });
    b.def("ingestEpisode", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (!d || !d->tr || a.empty() || !ev::isObject(a[0])) return ev::undefined();
        ev::Persistent o(a[0]);
        grid::EpisodeSummary e;
        e.total_return = static_cast<float>(getDoubleProperty(o.get(), "totalReturn", 0.0));
        e.depth = getIntProp(o.get(), "depth", 0);
        e.failed = getBoolProperty(o.get(), "failed", false);
        Value snap = ev::getProperty(o.get(), "snapshot");
        if (!ev::isUndefined(snap) && !ev::isNull(snap)) {
            e.start_snapshot = std::any{JsSnapshot(snap)};
        }
        e.action_prefix = readIntArrayValue(ev::getProperty(o.get(), "prefix"));
        e.failure_tail = readFailureTail(ev::getProperty(o.get(), "tail"));
        d->tr->ingest_episode(std::move(e));
        return ev::undefined();
    });
    b.def("warmupWith", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (!d || !d->tr || a.empty() || !ev::isObject(a[0])) return ev::undefined();
        ev::Persistent arr(a[0]);
        Value lenV = ev::getProperty(arr.get(), "length");
        if (ev::isUndefined(lenV) || ev::isObject(lenV)) return ev::undefined();
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(lenV));
        std::vector<learn::GenericSituation> sits;
        sits.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            sits.push_back(situationFromValue(ev::getElement(arr.get(), i)));
        }
        d->tr->warmup_with(sits);
        return ev::undefined();
    });
    b.def("start", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (d && d->tr) d->tr->start();
        return ev::undefined();
    });
    b.def("stop", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (d && d->tr) d->tr->stop();
        return ev::undefined();
    });
    b.def("stepSync", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (d && d->tr) d->tr->step_sync(i32At(a, 0));
        return ev::undefined();
    });
    b.def("stats", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridTrainer(self);
        ObjectBuilder o;
        if (!d || !d->tr) return o.get();
        grid::GridTrainerStats s = d->tr->stats();
        o.set("totalSteps", static_cast<double>(s.total_steps));
        o.set("totalPublishes", static_cast<double>(s.total_publishes));
        o.set("episodesIngested", static_cast<double>(s.episodes_ingested));
        o.set("trailingMeanReturn", static_cast<double>(s.trailing_mean_return));
        o.set("bestMeanReturn", static_cast<double>(s.best_mean_return));
        o.set("bufferSize", static_cast<double>(s.buffer_size));
        o.set("running", s.running);
        return o.get();
    });
    b.def("pollEvents", 0, [](Value self, std::span<const Value>) -> Value {
        auto* d = unwrapGridTrainer(self);
        if (!d || !d->tr) return hostArrayOf(0, [](size_t) { return ev::undefined(); });
        std::vector<grid::GridEvent> evs = d->tr->poll_events();
        return hostArrayOf(evs.size(), [&](size_t i) {
            const auto& e = evs[i];
            const char* kind = "?";
            switch (e.kind) {
                case grid::GridEvent::Kind::WeightsUpdated: kind = "weightsUpdated"; break;
                case grid::GridEvent::Kind::BestRotated: kind = "bestRotated"; break;
                case grid::GridEvent::Kind::EpisodeIngested: kind = "episodeIngested"; break;
            }
            ObjectBuilder o;
            o.set("kind", kind);
            o.set("version", makeBigIntValue(e.version));
            o.set("totalSteps", static_cast<double>(e.total_steps));
            o.set("episodeCount", static_cast<double>(e.episode_count));
            o.set("meanReturn", static_cast<double>(e.mean_return));
            o.set("path", e.path);
            return o.get();
        });
    });
}

Value createGridTrainer(std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0])) {
        return ev::throwTypeError("createGridTrainer(opts): expected object");
    }
    ev::Persistent opts(a[0]);

    grid::GridTrainerConfig cfg;
    Value netV = ev::getProperty(opts.get(), "net");
    ev::Persistent nv(ev::isObject(netV) ? netV : opts.get());
    cfg.net.in_dim = getIntProp(nv.get(), "inDim", 0);
    cfg.net.value_hidden = getIntProp(nv.get(), "valueHidden", cfg.net.value_hidden);
    cfg.net.num_actions = getIntProp(nv.get(), "numActions", 0);
    cfg.net.seed = getU64Property(nv.get(), "seed", cfg.net.seed);
    {
        std::vector<int> hidden = readIntArrayValue(ev::getProperty(nv.get(), "hidden"));
        if (!hidden.empty()) cfg.net.hidden = std::move(hidden);
    }

    Value bufV = ev::getProperty(opts.get(), "buffer");
    if (ev::isObject(bufV)) {
        cfg.buffer_capacity = getIntProp(bufV, "capacity", cfg.buffer_capacity);
    }

    Value trV = ev::getProperty(opts.get(), "trainer");
    if (ev::isObject(trV)) {
        ev::Persistent tr(trV);
        cfg.trainer.lr = static_cast<float>(getDoubleProperty(tr.get(), "lr", cfg.trainer.lr));
        cfg.trainer.momentum =
            static_cast<float>(getDoubleProperty(tr.get(), "momentum", cfg.trainer.momentum));
        cfg.trainer.batch = getIntProp(tr.get(), "batch", cfg.trainer.batch);
        cfg.trainer.policy_weight = static_cast<float>(
            getDoubleProperty(tr.get(), "policyWeight", cfg.trainer.policy_weight));
        cfg.trainer.value_weight = static_cast<float>(
            getDoubleProperty(tr.get(), "valueWeight", cfg.trainer.value_weight));
        cfg.trainer.publish_every = getIntProp(tr.get(), "publishEvery", cfg.trainer.publish_every);
        cfg.trainer.rng_seed = getU64Property(tr.get(), "rngSeed", cfg.trainer.rng_seed);
    }

    Value ckptV = ev::getProperty(opts.get(), "ckpt");
    if (ev::isObject(ckptV)) {
        ev::Persistent ck(ckptV);
        cfg.ckpt_dir = readStringPropOr(ck.get(), "dir", cfg.ckpt_dir.c_str());
        cfg.ckpt_ring_size = getIntProp(ck.get(), "ringSize", cfg.ckpt_ring_size);
        cfg.best_window = getIntProp(ck.get(), "bestWindow", cfg.best_window);
    }

    cfg.ingest_burst = getIntProp(opts.get(), "ingestBurst", cfg.ingest_burst);
    cfg.steps_per_tick = getIntProp(opts.get(), "stepsPerTick", cfg.steps_per_tick);

    if (cfg.net.in_dim <= 0 || cfg.net.num_actions <= 0) {
        return ev::throwTypeError("createGridTrainer: net.inDim and net.numActions must be > 0");
    }

    auto cell = std::make_unique<HostGridTrainer>();
    try {
        cell->tr = std::make_unique<grid::GridTrainer>(std::move(cfg));
    } catch (const std::exception& e) {
        return ev::throwError(e.what());
    }
    return g_gridTrainerClass.createInstance(std::move(cell));
}

} // namespace

void ensureAIGridRecordingClassesInstalled() {
    static thread_local bool installed = false;
    if (installed) return;
    installed = true;

    g_gridRecorderClass.init("AIGridGenericRecorder", decorateGridRecorder);
    g_gridReaderClass.init("AIGridGenericReplayReader", decorateGridReader);
    g_gridTrainerClass.init("AIGridTrainer", decorateGridTrainer);
}

void installAIGridRecording(ObjectBuilder& gridNs) {
    ensureAIGridRecordingClassesInstalled();

    gridNs.def("createGenericRecorder", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostGridRecorder>();
        cell->rec = std::make_unique<grid::GenericRecorder>();
        return g_gridRecorderClass.createInstance(std::move(cell));
    });

    gridNs.def("createGenericReplayReader", 0, [](Value, std::span<const Value>) -> Value {
        auto cell = std::make_unique<HostGridReader>();
        cell->rr = std::make_unique<grid::GenericReplayReader>();
        return g_gridReaderClass.createInstance(std::move(cell));
    });

    gridNs.def("createGridTrainer", 1, [](Value, std::span<const Value> a) -> Value {
        return createGridTrainer(a);
    });
}

} // namespace brogameagent::api

#endif  // BROGAMEAGENT_HAS_NN
