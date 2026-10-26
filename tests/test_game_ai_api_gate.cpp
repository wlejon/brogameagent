// bro.ai.game's replay files under the host's access gate (embed/embed_access.h;
// src/api/api.h, "Access"): the recorders' open asks Write and the replay
// readers' open Read about the path, a refusal is an Error carrying the gate's
// sentence, code EACCES and the absolute path, nothing is created or read, and
// with the gate removed everything behaves as before.

#include "api/api.h"
#include "embed/embed.h"
#include "embed/embed_access.h"
#include "eval/eval.h"

#ifdef BROGAMEAGENT_HAS_NN
#include "brotensor/api.h"
#endif

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;
namespace ev = bronze::embed;

#define GCHECK(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK failed: " #cond " (" << __FILE__ << ":"        \
                      << __LINE__ << "): " << (msg) << std::endl;              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The path as the gate is handed it.
std::string gatePath(const fs::path& p) {
    return utf8(fs::absolute(p).lexically_normal());
}

std::string jsStr(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string evalString(const std::string& src) {
    auto r = bronze::eval::evalScript(src);
    if (r.thrown) return "<thrown: " + ev::toUtf8(r.value) + ">";
    return ev::isString(r.value) ? ev::toUtf8(r.value) : "<not a string>";
}

// Runs `stmt`; "ok" or the refusal as one comparable string. `stmt` may
// `return` a string of its own (a failed check).
std::string syncJs(const std::string& stmt) {
    return evalString("(function() { const G = bro.ai.game; try { " + stmt +
                      "; return 'ok'; } catch (e) {\n"
                      "  return 'denied|' + e.message + '|' + e.code + '|' + e.path; } })()");
}

std::string refusal(const fs::path& p) {
    const std::string path = gatePath(p);
    return "denied|no files access: " + path + "|EACCES|" + path;
}

const char* kSchemas =
    "{roster: [{name: 'id', type: 'i32'}], frame: [{name: 'x', type: 'f32'}], events: [{name: 'k', type: 'i32'}]}";

} // namespace

int main() {
    std::cout << "Running brogameagent access-gate test..." << std::endl;
    // Per-process: the test and its gcstress twin run at once.
    const fs::path root = fs::temp_directory_path() / ("brogameagent_gate_" + std::to_string(std::random_device{}()));
    const fs::path ok = root / "ok";
    const fs::path hidden = root / "hidden";
    std::error_code ec;
    fs::create_directories(ok);
    fs::create_directories(hidden);

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
#ifdef BROGAMEAGENT_HAS_NN
        brotensor::api::installTensor();
#endif
        brogameagent::api::installGameAi();

        // A replay in each directory, recorded before any gate is installed.
        for (const fs::path& dir : {ok, hidden}) {
            const std::string r = syncJs("const rec = G.createRecorder(); if (!rec.open(" + jsStr(dir / "r.bin") +
                                         ", 1, 2, 1 / 60)) return 'open'; rec.close()");
            GCHECK(r == "ok", r);
        }

        const std::string hiddenPrefix = gatePath(hidden);
        std::atomic<int> calls{0};
        ev::setAccessGate([&](const ev::AccessRequest& q) -> std::string {
            if (q.kind != ev::AccessKind::Read && q.kind != ev::AccessKind::Write) return "";
            ++calls;
            const std::string path(q.subject);
            return path.compare(0, hiddenPrefix.size(), hiddenPrefix) == 0 ? "no files access: " + path : "";
        });

        std::string r;
        r = syncJs("G.createRecorder().open(" + jsStr(hidden / "new.bin") + ", 1, 2, 1 / 60)");
        GCHECK(r == refusal(hidden / "new.bin"), r);
        GCHECK(!fs::exists(hidden / "new.bin"), "a refused recorder created its file");
        r = syncJs("G.createReplayReader().open(" + jsStr(hidden / "r.bin") + ")");
        GCHECK(r == refusal(hidden / "r.bin"), r);
        // Allowed: the reader opens it (an empty replay, so open's own answer
        // is not the point; it does not throw).
        r = syncJs("G.createReplayReader().open(" + jsStr(ok / "r.bin") + ")");
        GCHECK(r == "ok", r);
        r = syncJs("const rec = G.createRecorder(); if (!rec.open(" + jsStr(ok / "new.bin") +
                   ", 1, 2, 1 / 60)) return 'open'; rec.close()");
        GCHECK(r == "ok", r);
        GCHECK(fs::exists(ok / "new.bin"), "an allowed recorder did not write");

        // The grid recorder, reader and trainer checkpoints (NN builds).
        const bool hasGrid = evalString("typeof bro.ai.game.grid === 'object' && typeof "
                                        "bro.ai.game.grid.createGenericRecorder === 'function' ? 'yes' : 'no'") == "yes";
#ifdef BROGAMEAGENT_HAS_NN
        GCHECK(hasGrid, "an NN build without bro.ai.game.grid's recorders");
#endif
        if (hasGrid) {
            r = syncJs("G.grid.createGenericRecorder().open(" + jsStr(hidden / "g.bin") + ", 1, 2, 1 / 60, " +
                       kSchemas + ")");
            GCHECK(r == refusal(hidden / "g.bin"), r);
            GCHECK(!fs::exists(hidden / "g.bin"), "a refused grid recorder created its file");
            r = syncJs("G.grid.createGenericReplayReader().open(" + jsStr(hidden / "r.bin") + ")");
            GCHECK(r == refusal(hidden / "r.bin"), r);
            r = syncJs("G.grid.createGridTrainer({net: {inDim: 4, numActions: 2}, ckpt: {dir: " +
                       jsStr(hidden / "ckpt") + "}})");
            GCHECK(r == refusal(hidden / "ckpt"), r);
            r = syncJs("const rec = G.grid.createGenericRecorder(); if (!rec.open(" + jsStr(ok / "g.bin") +
                       ", 1, 2, 1 / 60, " + kSchemas + ")) return 'open'; rec.close()");
            GCHECK(r == "ok", r);
            GCHECK(fs::exists(ok / "g.bin"), "an allowed grid recorder did not write");
        }

        // With the gate removed, nothing asks and everything works as before.
        ev::setAccessGate(nullptr);
        const int before = calls.load();
        r = syncJs("G.createReplayReader().open(" + jsStr(hidden / "r.bin") + ")");
        GCHECK(r == "ok", r);
        r = syncJs("const rec = G.createRecorder(); if (!rec.open(" + jsStr(hidden / "new.bin") +
                   ", 1, 2, 1 / 60)) return 'open'; rec.close()");
        GCHECK(r == "ok", r);
        GCHECK(fs::exists(hidden / "new.bin"), "an ungated recorder did not write");
        GCHECK(calls.load() == before, "a removed gate was still asked");
    }
    ev::destroyRealm(realm);
    fs::remove_all(root, ec);

    std::cout << "brogameagent access-gate test passed" << std::endl;
    return 0;
}
