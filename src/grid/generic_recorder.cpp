#include "brogameagent/grid/generic_recorder.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace brogameagent::grid {

namespace {

constexpr char     MAGIC[8]     = { 'B','G','A','G','R','I','D','\0' };
constexpr char     MAGIC_END[8] = { 'B','G','A','G','E','N','D','\0' };
constexpr uint32_t VERSION = 1;

uint32_t field_size(FieldType t) {
    switch (t) {
        case FieldType::I32: return 4;
        case FieldType::I64: return 8;
        case FieldType::F32: return 4;
        case FieldType::F64: return 8;
    }
    return 0;
}

uint32_t schema_row_bytes(const std::vector<FieldDef>& s) {
    uint32_t total = 0;
    for (const auto& f : s) total += field_size(f.type);
    return total;
}

template <typename T>
bool write_raw(std::FILE* f, const T& v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
bool read_raw(std::FILE* f, T& v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

bool write_schema(std::FILE* f, const std::vector<FieldDef>& s) {
    uint32_t n = static_cast<uint32_t>(s.size());
    if (!write_raw(f, n)) return false;
    for (const auto& fd : s) {
        uint16_t name_len = static_cast<uint16_t>(fd.name.size());
        if (!write_raw(f, name_len)) return false;
        if (name_len && std::fwrite(fd.name.data(), 1, name_len, f) != name_len) return false;
        uint8_t t = static_cast<uint8_t>(fd.type);
        if (!write_raw(f, t)) return false;
    }
    return true;
}

// 64-bit file positioning: `long` is 32 bits on Windows, so fseek/ftell
// cannot address a replay past 2 GiB.
bool seek_to(std::FILE* f, uint64_t off) {
    if (off > static_cast<uint64_t>(INT64_MAX)) return false;
#ifdef _MSC_VER
    return _fseeki64(f, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

bool tell_at(std::FILE* f, uint64_t& off) {
#ifdef _MSC_VER
    const __int64 p = _ftelli64(f);
#else
    const off_t p = ftello(f);
#endif
    if (p < 0) return false;
    off = static_cast<uint64_t>(p);
    return true;
}

bool file_size_of(std::FILE* f, uint64_t& size) {
    uint64_t here = 0;
    if (!tell_at(f, here)) return false;
#ifdef _MSC_VER
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
#endif
    if (!tell_at(f, size)) return false;
    return seek_to(f, here);
}

// Every count in the file is checked against these and against the bytes
// left in the file before it sizes an allocation.
constexpr uint32_t kMaxFields = 4096;       // per schema
constexpr uint32_t kMaxRows   = 1u << 20;   // roster, per-frame rows / events

// `count` records of `row_bytes` each fit in `avail` bytes, within kMaxRows.
bool rows_fit(uint32_t count, uint32_t row_bytes, uint64_t avail) {
    return count <= kMaxRows &&
           static_cast<uint64_t>(count) * row_bytes <= avail;
}

bool read_schema(std::FILE* f, uint64_t limit, std::vector<FieldDef>& out) {
    uint32_t n = 0;
    if (!read_raw(f, n)) return false;
    uint64_t pos = 0;
    if (!tell_at(f, pos) || pos > limit) return false;
    // Smallest field on disk: u16 name length + u8 type.
    constexpr uint64_t kMinField = sizeof(uint16_t) + sizeof(uint8_t);
    if (n > kMaxFields || static_cast<uint64_t>(n) * kMinField > limit - pos) return false;
    out.clear();
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t name_len = 0;
        if (!read_raw(f, name_len)) return false;
        std::string name(name_len, '\0');
        if (name_len && std::fread(name.data(), 1, name_len, f) != name_len) return false;
        uint8_t t = 0;
        if (!read_raw(f, t)) return false;
        if (t < static_cast<uint8_t>(FieldType::I32) ||
            t > static_cast<uint8_t>(FieldType::F64)) return false;
        out.push_back({ std::move(name), static_cast<FieldType>(t) });
    }
    return true;
}

void write_row(std::FILE* f, const std::vector<FieldDef>& schema, const Row& row) {
    for (size_t i = 0; i < schema.size(); ++i) {
        FieldType t = schema[i].type;
        if (i < row.size()) {
            const auto& v = row[i];
            switch (t) {
                case FieldType::I32: {
                    int32_t x = 0;
                    if (auto* pi32 = std::get_if<int32_t>(&v))      x = *pi32;
                    else if (auto* pi64 = std::get_if<int64_t>(&v)) x = static_cast<int32_t>(*pi64);
                    else if (auto* pf   = std::get_if<float>(&v))   x = static_cast<int32_t>(*pf);
                    else if (auto* pd   = std::get_if<double>(&v))  x = static_cast<int32_t>(*pd);
                    write_raw(f, x);
                    break;
                }
                case FieldType::I64: {
                    int64_t x = 0;
                    if (auto* pi64 = std::get_if<int64_t>(&v))      x = *pi64;
                    else if (auto* pi32 = std::get_if<int32_t>(&v)) x = static_cast<int64_t>(*pi32);
                    else if (auto* pf   = std::get_if<float>(&v))   x = static_cast<int64_t>(*pf);
                    else if (auto* pd   = std::get_if<double>(&v))  x = static_cast<int64_t>(*pd);
                    write_raw(f, x);
                    break;
                }
                case FieldType::F32: {
                    float x = 0.0f;
                    if (auto* pf32 = std::get_if<float>(&v))        x = *pf32;
                    else if (auto* pd   = std::get_if<double>(&v))  x = static_cast<float>(*pd);
                    else if (auto* pi32 = std::get_if<int32_t>(&v)) x = static_cast<float>(*pi32);
                    else if (auto* pi64 = std::get_if<int64_t>(&v)) x = static_cast<float>(*pi64);
                    write_raw(f, x);
                    break;
                }
                case FieldType::F64: {
                    double x = 0.0;
                    if (auto* pd64 = std::get_if<double>(&v))       x = *pd64;
                    else if (auto* pf   = std::get_if<float>(&v))   x = static_cast<double>(*pf);
                    else if (auto* pi32 = std::get_if<int32_t>(&v)) x = static_cast<double>(*pi32);
                    else if (auto* pi64 = std::get_if<int64_t>(&v)) x = static_cast<double>(*pi64);
                    write_raw(f, x);
                    break;
                }
            }
        } else {
            // Field missing in row — emit zero-valued slot of correct size.
            uint8_t zeros[8] = {0};
            std::fwrite(zeros, 1, field_size(t), f);
        }
    }
}

bool read_row(std::FILE* f, const std::vector<FieldDef>& schema, Row& row) {
    row.clear();
    row.reserve(schema.size());
    for (const auto& fd : schema) {
        bool ok = false;
        switch (fd.type) {
            case FieldType::I32: { int32_t x = 0; ok = read_raw(f, x); row.push_back(x); break; }
            case FieldType::I64: { int64_t x = 0; ok = read_raw(f, x); row.push_back(x); break; }
            case FieldType::F32: { float x = 0;   ok = read_raw(f, x); row.push_back(x); break; }
            case FieldType::F64: { double x = 0;  ok = read_raw(f, x); row.push_back(x); break; }
        }
        if (!ok) return false;
    }
    return true;
}

bool read_rows(std::FILE* f, const std::vector<FieldDef>& schema, uint32_t n,
               std::vector<Row>& out) {
    out.clear();
    out.reserve(n);
    for (uint32_t k = 0; k < n; ++k) {
        Row r;
        if (!read_row(f, schema, r)) return false;
        out.push_back(std::move(r));
    }
    return true;
}

} // namespace

// ─── Recorder ──────────────────────────────────────────────────────────────

GenericRecorder::~GenericRecorder() {
    if (file_) close();
}

bool GenericRecorder::open(const std::string& path,
                           uint64_t episode_id, uint64_t seed, float dt,
                           std::vector<FieldDef> roster_schema,
                           std::vector<FieldDef> frame_schema,
                           std::vector<FieldDef> event_schema) {
    if (file_) return false;
#ifdef _MSC_VER
    fopen_s(&file_, path.c_str(), "wb");
#else
    file_ = std::fopen(path.c_str(), "wb");
#endif
    if (!file_) return false;

    roster_schema_ = std::move(roster_schema);
    frame_schema_  = std::move(frame_schema);
    event_schema_  = std::move(event_schema);
    roster_row_bytes_ = schema_row_bytes(roster_schema_);
    frame_row_bytes_  = schema_row_bytes(frame_schema_);
    event_row_bytes_  = schema_row_bytes(event_schema_);

    if (std::fwrite(MAGIC, 1, 8, file_) != 8) return false;
    if (!write_raw(file_, VERSION))           return false;
    if (!write_raw(file_, episode_id))        return false;
    if (!write_raw(file_, seed))              return false;
    if (!write_raw(file_, dt))                return false;

    if (!write_schema(file_, roster_schema_)) return false;
    if (!write_schema(file_, frame_schema_))  return false;
    if (!write_schema(file_, event_schema_))  return false;

    // Roster placeholder (count = 0). Filled by write_roster().
    uint32_t placeholder = 0;
    if (!write_raw(file_, placeholder))       return false;
    if (!write_raw(file_, roster_row_bytes_)) return false;

    return true;
}

void GenericRecorder::write_roster(const std::vector<Row>& rows) {
    if (!file_ || roster_written_) return;
    // Backpatch the count placeholder we wrote during open(). We need the
    // header section size: 8 magic + 4 ver + 8 episode + 8 seed + 4 dt
    // + roster schema + frame schema + event schema + 4 placeholder + 4
    // roster_row_bytes — but easiest is ftell back and forth.
    long here = std::ftell(file_);
    long count_offset = here - 8;   // placeholder was 8 bytes back: u32 count + u32 row_bytes
    uint32_t n = static_cast<uint32_t>(rows.size());
    std::fseek(file_, count_offset, SEEK_SET);
    write_raw(file_, n);
    std::fseek(file_, here, SEEK_SET);
    for (const auto& r : rows) write_row(file_, roster_schema_, r);
    roster_written_ = true;
}

void GenericRecorder::record_frame(uint64_t step_idx, float elapsed,
                                   const std::vector<Row>& rows,
                                   const std::vector<Row>& events) {
    if (!file_) return;
    // If write_roster was never called, fix the placeholder to 0 and skip
    // forward — we'll write nothing for the roster body.
    if (!roster_written_) {
        roster_written_ = true; // count is already 0; nothing to backpatch.
    }
    uint64_t off = static_cast<uint64_t>(std::ftell(file_));
    frame_offsets_.push_back(off);
    write_raw(file_, step_idx);
    write_raw(file_, elapsed);
    uint32_t nr = static_cast<uint32_t>(rows.size());
    write_raw(file_, nr);
    for (const auto& r : rows) write_row(file_, frame_schema_, r);
    uint32_t ne = static_cast<uint32_t>(events.size());
    write_raw(file_, ne);
    for (const auto& e : events) write_row(file_, event_schema_, e);
}

bool GenericRecorder::close() {
    if (!file_) return false;
    uint64_t footer_off = static_cast<uint64_t>(std::ftell(file_));
    uint32_t n = static_cast<uint32_t>(frame_offsets_.size());
    write_raw(file_, n);
    for (auto o : frame_offsets_) write_raw(file_, o);
    write_raw(file_, footer_off);
    std::fwrite(MAGIC_END, 1, 8, file_);
    int rc = std::fclose(file_);
    file_ = nullptr;
    return rc == 0;
}

// ─── Reader ────────────────────────────────────────────────────────────────

GenericReplayReader::~GenericReplayReader() {
    if (file_) std::fclose(file_);
}

bool GenericReplayReader::open(const std::string& path) {
    if (file_) std::fclose(file_);
    file_ = nullptr;
    err_.clear();
#ifdef _MSC_VER
    fopen_s(&file_, path.c_str(), "rb");
#else
    file_ = std::fopen(path.c_str(), "rb");
#endif
    roster_.clear();
    frame_offsets_.clear();
    data_end_ = 0;
    if (!file_) { err_ = "open failed"; return false; }

    // Every count and offset read below is checked against the bytes the
    // file actually holds before it sizes an allocation or a seek: a corrupt
    // or hostile replay fails to open instead of allocating gigabytes.
    auto fail = [this](const char* why) {
        err_ = why;
        std::fclose(file_);
        file_ = nullptr;
        roster_.clear();
        frame_offsets_.clear();
        data_end_ = 0;
        return false;
    };

    // Trailer: footer_off u64 + MAGIC_END[8], the last 16 bytes.
    constexpr uint64_t kTrailer = sizeof(uint64_t) + 8;
    uint64_t file_size = 0;
    if (!file_size_of(file_, file_size)) return fail("file size");

    char magic[8] = {0};
    if (std::fread(magic, 1, 8, file_) != 8 || std::memcmp(magic, MAGIC, 8) != 0) {
        return fail("bad magic");
    }
    uint32_t ver = 0;
    if (!read_raw(file_, ver) || ver != VERSION) return fail("version mismatch");
    if (!read_raw(file_, episode_id_) ||
        !read_raw(file_, seed_) ||
        !read_raw(file_, dt_)) {
        return fail("header read");
    }
    if (file_size < kTrailer) return fail("file too small");
    const uint64_t body_limit = file_size - kTrailer;
    if (!read_schema(file_, body_limit, roster_schema_) ||
        !read_schema(file_, body_limit, frame_schema_)  ||
        !read_schema(file_, body_limit, event_schema_)) {
        return fail("schema read");
    }
    const uint32_t roster_bytes = schema_row_bytes(roster_schema_);
    frame_row_bytes_ = schema_row_bytes(frame_schema_);
    event_row_bytes_ = schema_row_bytes(event_schema_);

    uint32_t roster_n = 0, roster_row_bytes = 0;
    if (!read_raw(file_, roster_n) || !read_raw(file_, roster_row_bytes)) {
        return fail("roster header");
    }
    if (roster_row_bytes != roster_bytes) return fail("roster row size mismatch");
    uint64_t pos = 0;
    if (!tell_at(file_, pos) || pos > body_limit) return fail("roster header");
    if (!rows_fit(roster_n, roster_bytes, body_limit - pos)) {
        return fail("roster count out of range");
    }
    if (!read_rows(file_, roster_schema_, roster_n, roster_)) return fail("truncated roster");
    uint64_t frames_begin = 0;
    if (!tell_at(file_, frames_begin)) return fail("roster read");

    // Walk to the footer to grab frame offsets.
    if (!seek_to(file_, body_limit)) return fail("seek footer");
    uint64_t footer_off = 0;
    if (!read_raw(file_, footer_off)) return fail("read footer offset");
    char tail[8] = {0};
    if (std::fread(tail, 1, 8, file_) != 8 || std::memcmp(tail, MAGIC_END, 8) != 0) {
        return fail("bad footer magic");
    }
    // The footer (u32 count + u64 offsets) sits between the last frame and
    // the trailer, and the writer leaves nothing else there.
    if (footer_off < frames_begin || footer_off > body_limit ||
        body_limit - footer_off < sizeof(uint32_t)) {
        return fail("footer offset out of range");
    }
    if (!seek_to(file_, footer_off)) return fail("seek to footer");
    uint32_t nframes = 0;
    if (!read_raw(file_, nframes)) return fail("frame count");
    if (static_cast<uint64_t>(nframes) * sizeof(uint64_t)
        != body_limit - footer_off - sizeof(uint32_t)) {
        return fail("frame count out of range");
    }
    // Smallest frame: step_idx u64 + elapsed f32 + row count u32 + event count u32.
    constexpr uint64_t kMinFrame = sizeof(uint64_t) + sizeof(float) + 2 * sizeof(uint32_t);
    frame_offsets_.resize(nframes);
    for (uint32_t i = 0; i < nframes; ++i) {
        if (!read_raw(file_, frame_offsets_[i])) return fail("frame offsets");
        const uint64_t off = frame_offsets_[i];
        if (off < frames_begin || off > footer_off || footer_off - off < kMinFrame) {
            return fail("frame offset out of range");
        }
    }
    data_end_ = footer_off;
    return true;
}

GenericFrame GenericReplayReader::frame(size_t i) const {
    // open() validated the offset; each count is checked against the bytes
    // left before the footer, so an overstated count yields an empty frame
    // rather than rows read out of the footer (or a huge reservation).
    GenericFrame fr;
    if (!file_ || i >= frame_offsets_.size()) return fr;
    if (!seek_to(file_, frame_offsets_[i])) return fr;
    if (!read_raw(file_, fr.step_idx)) return {};
    if (!read_raw(file_, fr.elapsed))  return {};
    auto remaining = [this](uint64_t& avail) {
        uint64_t pos = 0;
        if (!tell_at(file_, pos) || pos > data_end_) return false;
        avail = data_end_ - pos;
        return true;
    };
    uint64_t avail = 0;
    uint32_t nr = 0;
    if (!read_raw(file_, nr) || !remaining(avail)) return {};
    if (!rows_fit(nr, frame_row_bytes_, avail)) return {};
    if (!read_rows(file_, frame_schema_, nr, fr.rows)) return {};
    uint32_t ne = 0;
    if (!read_raw(file_, ne) || !remaining(avail)) return {};
    if (!rows_fit(ne, event_row_bytes_, avail)) return {};
    if (!read_rows(file_, event_schema_, ne, fr.events)) return {};
    return fr;
}

std::vector<FieldValue> GenericReplayReader::trajectory(size_t row_index,
                                                        const std::string& field_name) const {
    std::vector<FieldValue> out;
    int field_idx = -1;
    for (size_t i = 0; i < frame_schema_.size(); ++i) {
        if (frame_schema_[i].name == field_name) { field_idx = static_cast<int>(i); break; }
    }
    if (field_idx < 0) return out;
    out.reserve(frame_offsets_.size());
    for (size_t i = 0; i < frame_offsets_.size(); ++i) {
        auto fr = frame(i);
        if (row_index < fr.rows.size() && static_cast<size_t>(field_idx) < fr.rows[row_index].size()) {
            out.push_back(fr.rows[row_index][static_cast<size_t>(field_idx)]);
        } else {
            // Default-zero of the field type.
            switch (frame_schema_[static_cast<size_t>(field_idx)].type) {
                case FieldType::I32: out.push_back(int32_t{0}); break;
                case FieldType::I64: out.push_back(int64_t{0}); break;
                case FieldType::F32: out.push_back(float{0}); break;
                case FieldType::F64: out.push_back(double{0}); break;
            }
        }
    }
    return out;
}

} // namespace brogameagent::grid
