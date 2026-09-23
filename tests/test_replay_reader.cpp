// Replay readers against corrupt files: every count and offset a .bgar or a
// generic grid replay carries is checked against the bytes actually present,
// so a truncated or hostile file fails open() (or yields an empty frame)
// instead of allocating from a garbage count or reading past the data.
// Core-only: builds with BROGAMEAGENT_WITH_NN=OFF.

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <brogameagent/brogameagent.h>
#include <brogameagent/grid/generic_recorder.h>
#include <brogameagent/replay_format.h>
#include <brogameagent/replay_reader.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace brogameagent;

struct TestEntry {
    const char* name;
    void (*fn)();
};

static std::vector<TestEntry>& registry() {
    static std::vector<TestEntry> r;
    return r;
}

#define TEST(name) \
    static void test_##name(); \
    struct Register_##name { Register_##name() { registry().push_back({#name, test_##name}); } } reg_##name; \
    static void test_##name()

static void check(bool cond, const char* msg, int line) {
    if (!cond) {
        printf("    assertion failed at line %d: %s\n", line, msg);
        throw 0;
    }
}

#define CHECK(cond) check(cond, #cond, __LINE__)

// ─── byte helpers ───────────────────────────────────────────────────────────

static std::vector<uint8_t> readAll(const std::string& path) {
    std::vector<uint8_t> out;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::rewind(f);
    out.resize(n > 0 ? static_cast<size_t>(n) : 0);
    if (!out.empty() && std::fread(out.data(), 1, out.size(), f) != out.size()) out.clear();
    std::fclose(f);
    return out;
}

static void writeAll(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    if (!bytes.empty()) std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
}

template <typename T>
static void poke(std::vector<uint8_t>& b, size_t off, T v) {
    check(off + sizeof(T) <= b.size(), "poke in range", __LINE__);
    std::memcpy(b.data() + off, &v, sizeof(T));
}

template <typename T>
static T peek(const std::vector<uint8_t>& b, size_t off) {
    T v{};
    check(off + sizeof(T) <= b.size(), "peek in range", __LINE__);
    std::memcpy(&v, b.data() + off, sizeof(T));
    return v;
}

// ─── .bgar (ReplayReader) ───────────────────────────────────────────────────

static const char* kBgar = "test_replay_reader_src.bgar";
static const char* kBgarBad = "test_replay_reader_bad.bgar";

// Two agents, three frames, one damage event; returns the file's bytes.
static std::vector<uint8_t> writeValidBgar() {
    World world;
    world.seed(3);
    Agent a, b;
    a.unit().id = 1; a.unit().teamId = 0; a.unit().hp = 100; a.unit().maxHp = 100;
    a.setPosition(1.0f, 2.0f);
    b.unit().id = 2; b.unit().teamId = 1; b.unit().hp = 50; b.unit().maxHp = 50;
    b.setPosition(-1.0f, 0.0f);
    world.addAgent(&a);
    world.addAgent(&b);

    Recorder rec;
    CHECK(rec.open(kBgar, 9, 3, 0.05f));
    rec.writeRoster(world.agents());
    for (uint32_t s = 0; s < 3; ++s) rec.recordFrame(s, 0.05f * static_cast<float>(s), world);
    CHECK(rec.close());
    return readAll(kBgar);
}

static bool openPatched(const std::vector<uint8_t>& bytes, ReplayReader& r) {
    writeAll(kBgarBad, bytes);
    return r.open(kBgarBad);
}

using replay::AgentStatic;
using replay::FileHeader;
using replay::Footer;
using replay::FrameHeader;
using replay::IndexEntry;

static constexpr size_t kRosterCountAt = sizeof(FileHeader);

TEST(bgar_valid_file_reads) {
    auto bytes = writeValidBgar();
    ReplayReader r;
    CHECK(openPatched(bytes, r));
    CHECK(r.roster().size() == 2);
    CHECK(r.frameCount() == 3);
    auto f = r.frame(2);
    CHECK(f.header.stepIdx == 2);
    CHECK(f.agents.size() == 2);
}

TEST(bgar_huge_roster_count_rejected) {
    auto bytes = writeValidBgar();
    poke<uint32_t>(bytes, kRosterCountAt, 0xFFFFFFFFu);
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
    CHECK(r.errorMessage().find("roster") != std::string::npos);
    CHECK(r.roster().empty());
    CHECK(r.frameCount() == 0);
}

TEST(bgar_roster_count_past_file_rejected) {
    // Within kMaxRoster, but more records than the file could hold.
    auto bytes = writeValidBgar();
    poke<uint32_t>(bytes, kRosterCountAt, 50000u);
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
}

TEST(bgar_index_offset_overflow_rejected) {
    // indexOffset + indexCount * 16 used to wrap past 2^64 and pass.
    auto bytes = writeValidBgar();
    const size_t footerAt = bytes.size() - sizeof(Footer);
    poke<uint64_t>(bytes, footerAt, UINT64_MAX - 8);
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
}

TEST(bgar_index_count_huge_rejected) {
    auto bytes = writeValidBgar();
    const size_t footerAt = bytes.size() - sizeof(Footer);
    poke<uint32_t>(bytes, footerAt + 8, 0x7FFFFFFFu);
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
}

TEST(bgar_index_inside_header_rejected) {
    auto bytes = writeValidBgar();
    const size_t footerAt = bytes.size() - sizeof(Footer);
    // Point the index at the file header with a count that still ends at
    // the footer: the index must lie after the roster.
    const uint64_t off = 8;
    poke<uint64_t>(bytes, footerAt, off);
    poke<uint32_t>(bytes, footerAt + 8,
                   static_cast<uint32_t>((footerAt - off) / sizeof(IndexEntry)));
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
}

TEST(bgar_frame_offset_out_of_range_rejected) {
    auto bytes = writeValidBgar();
    const size_t footerAt = bytes.size() - sizeof(Footer);
    const uint64_t indexAt = peek<uint64_t>(bytes, footerAt);
    // IndexEntry.offset of entry 1 -> far past the end of the file.
    poke<uint64_t>(bytes, static_cast<size_t>(indexAt) + sizeof(IndexEntry) + 8,
                   uint64_t{1} << 40);
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
}

TEST(bgar_overstated_frame_counts_yield_empty_frame) {
    auto bytes = writeValidBgar();
    const size_t footerAt = bytes.size() - sizeof(Footer);
    const uint64_t indexAt = peek<uint64_t>(bytes, footerAt);
    // The last frame's header: its body would run into the index table.
    const uint64_t lastFrame =
        peek<uint64_t>(bytes, static_cast<size_t>(indexAt) + 2 * sizeof(IndexEntry) + 8);
    poke<uint16_t>(bytes, static_cast<size_t>(lastFrame) + 8, 60000);  // liveCount
    ReplayReader r;
    CHECK(openPatched(bytes, r));
    auto bad = r.frame(2);
    CHECK(bad.agents.empty());
    CHECK(bad.header.liveCount == 0);
    // The other frames are unaffected, and the query helpers skip the bad one.
    CHECK(r.frame(1).agents.size() == 2);
    CHECK(r.trajectory(1).size() == 2);
}

TEST(bgar_truncated_file_rejected) {
    auto bytes = writeValidBgar();
    bytes.resize(bytes.size() / 2);
    ReplayReader r;
    CHECK(!openPatched(bytes, r));
}

// ─── generic grid replay (GenericReplayReader) ──────────────────────────────

static const char* kGen = "test_replay_reader_src.bggr";
static const char* kGenBad = "test_replay_reader_bad.bggr";

// Header before the first schema: magic 8, version 4, episode 8, seed 8, dt 4.
static constexpr size_t kSchemaAt = 32;
// Schemas below: roster {"id" I32} = 4 + (2+2+1); frame {"x" F32, "y" F32} =
// 4 + 2*(2+1+1); events {"hit" I32} = 4 + (2+3+1).
static constexpr size_t kRosterSchemaBytes = 4 + 5;
static constexpr size_t kFrameSchemaBytes  = 4 + 8;
static constexpr size_t kEventSchemaBytes  = 4 + 6;
static constexpr size_t kFrameSchemaAt = kSchemaAt + kRosterSchemaBytes;
static constexpr size_t kRosterCountAt2 =
    kSchemaAt + kRosterSchemaBytes + kFrameSchemaBytes + kEventSchemaBytes;

static std::vector<uint8_t> writeValidGeneric() {
    using namespace grid;
    GenericRecorder rec;
    CHECK(rec.open(kGen, 5, 6, 0.1f,
                   { { "id", FieldType::I32 } },
                   { { "x", FieldType::F32 }, { "y", FieldType::F32 } },
                   { { "hit", FieldType::I32 } }));
    rec.write_roster({ { int32_t{7} }, { int32_t{8} } });
    for (uint64_t s = 0; s < 3; ++s) {
        rec.record_frame(s, 0.1f * static_cast<float>(s),
                         { { 1.0f, 2.0f }, { 3.0f, 4.0f } },
                         { { int32_t{1} } });
    }
    CHECK(rec.close());
    return readAll(kGen);
}

static bool openGenPatched(const std::vector<uint8_t>& bytes, grid::GenericReplayReader& r) {
    writeAll(kGenBad, bytes);
    return r.open(kGenBad);
}

// footer_off (the u32 frame count, then the u64 frame offsets).
static size_t genFooterAt(const std::vector<uint8_t>& b) {
    return static_cast<size_t>(peek<uint64_t>(b, b.size() - 16));
}

TEST(generic_valid_file_reads) {
    auto bytes = writeValidGeneric();
    CHECK(peek<uint32_t>(bytes, kRosterCountAt2) == 2);  // layout sanity
    grid::GenericReplayReader r;
    CHECK(openGenPatched(bytes, r));
    CHECK(r.roster().size() == 2);
    CHECK(r.frame_count() == 3);
    auto f = r.frame(1);
    CHECK(f.step_idx == 1);
    CHECK(f.rows.size() == 2);
    CHECK(f.events.size() == 1);
}

TEST(generic_huge_schema_count_rejected) {
    auto bytes = writeValidGeneric();
    poke<uint32_t>(bytes, kFrameSchemaAt, 0xFFFFFFF0u);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
    CHECK(!r.is_open());
}

TEST(generic_unknown_field_type_rejected) {
    auto bytes = writeValidGeneric();
    // Type byte of the roster schema's only field: count 4 + len 2 + "id" 2.
    bytes[kSchemaAt + 8] = 0x7F;
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
}

TEST(generic_huge_roster_count_rejected) {
    auto bytes = writeValidGeneric();
    poke<uint32_t>(bytes, kRosterCountAt2, 0xFFFFFFFFu);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
    CHECK(r.roster().empty());
}

TEST(generic_roster_row_size_mismatch_rejected) {
    auto bytes = writeValidGeneric();
    poke<uint32_t>(bytes, kRosterCountAt2 + 4, 4096u);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
}

TEST(generic_footer_offset_out_of_range_rejected) {
    auto bytes = writeValidGeneric();
    poke<uint64_t>(bytes, bytes.size() - 16, uint64_t{1} << 62);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
}

TEST(generic_huge_frame_count_rejected) {
    auto bytes = writeValidGeneric();
    poke<uint32_t>(bytes, genFooterAt(bytes), 0x40000000u);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
    CHECK(r.frame_count() == 0);
}

TEST(generic_frame_offset_out_of_range_rejected) {
    auto bytes = writeValidGeneric();
    poke<uint64_t>(bytes, genFooterAt(bytes) + 4 + 8, uint64_t{1} << 50);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
}

TEST(generic_overstated_row_count_yields_empty_frame) {
    auto bytes = writeValidGeneric();
    const size_t frame0 = static_cast<size_t>(peek<uint64_t>(bytes, genFooterAt(bytes) + 4));
    // Row count sits after step_idx u64 + elapsed f32.
    poke<uint32_t>(bytes, frame0 + 12, 0x00FFFFFFu);
    grid::GenericReplayReader r;
    CHECK(openGenPatched(bytes, r));
    auto f = r.frame(0);
    CHECK(f.rows.empty());
    CHECK(f.events.empty());
    CHECK(r.frame(1).rows.size() == 2);
}

TEST(generic_overstated_event_count_yields_empty_frame) {
    auto bytes = writeValidGeneric();
    const size_t frame2 =
        static_cast<size_t>(peek<uint64_t>(bytes, genFooterAt(bytes) + 4 + 16));
    // Event count follows two 8-byte rows.
    poke<uint32_t>(bytes, frame2 + 16 + 16, 100000u);
    grid::GenericReplayReader r;
    CHECK(openGenPatched(bytes, r));
    CHECK(r.frame(2).events.empty());
    CHECK(r.frame(0).events.size() == 1);
}

TEST(generic_truncated_file_rejected) {
    auto bytes = writeValidGeneric();
    bytes.resize(bytes.size() - 20);
    grid::GenericReplayReader r;
    CHECK(!openGenPatched(bytes, r));
}

int main() {
    printf("brogameagent replay reader tests\n");
    printf("================================\n");

    int passed = 0;
    for (const auto& t : registry()) {
        try {
            t.fn();
            passed++;
            printf("  PASS  %s\n", t.name);
        } catch (...) {
            printf("  FAIL  %s\n", t.name);
        }
    }
    std::remove(kBgar);
    std::remove(kBgarBad);
    std::remove(kGen);
    std::remove(kGenBad);

    int total = static_cast<int>(registry().size());
    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
