#include "brogameagent/grid/generic_recorder.h"

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

bool read_schema(std::FILE* f, std::vector<FieldDef>& out) {
    uint32_t n = 0;
    if (!read_raw(f, n)) return false;
    out.clear();
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t name_len = 0;
        if (!read_raw(f, name_len)) return false;
        std::string name(name_len, '\0');
        if (name_len && std::fread(name.data(), 1, name_len, f) != name_len) return false;
        uint8_t t = 0;
        if (!read_raw(f, t)) return false;
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

Row read_row(std::FILE* f, const std::vector<FieldDef>& schema) {
    Row row;
    row.reserve(schema.size());
    for (const auto& fd : schema) {
        switch (fd.type) {
            case FieldType::I32: { int32_t x = 0; read_raw(f, x); row.push_back(x); break; }
            case FieldType::I64: { int64_t x = 0; read_raw(f, x); row.push_back(x); break; }
            case FieldType::F32: { float x = 0;   read_raw(f, x); row.push_back(x); break; }
            case FieldType::F64: { double x = 0;  read_raw(f, x); row.push_back(x); break; }
        }
    }
    return row;
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
    if (!file_) { err_ = "open failed"; return false; }

    char magic[8] = {0};
    if (std::fread(magic, 1, 8, file_) != 8 || std::memcmp(magic, MAGIC, 8) != 0) {
        err_ = "bad magic"; std::fclose(file_); file_ = nullptr; return false;
    }
    uint32_t ver = 0;
    if (!read_raw(file_, ver) || ver != VERSION) {
        err_ = "version mismatch"; std::fclose(file_); file_ = nullptr; return false;
    }
    if (!read_raw(file_, episode_id_) ||
        !read_raw(file_, seed_) ||
        !read_raw(file_, dt_)) {
        err_ = "header read"; std::fclose(file_); file_ = nullptr; return false;
    }
    if (!read_schema(file_, roster_schema_) ||
        !read_schema(file_, frame_schema_)  ||
        !read_schema(file_, event_schema_)) {
        err_ = "schema read"; std::fclose(file_); file_ = nullptr; return false;
    }
    frame_row_bytes_ = schema_row_bytes(frame_schema_);
    event_row_bytes_ = schema_row_bytes(event_schema_);

    uint32_t roster_n = 0, roster_row_bytes = 0;
    if (!read_raw(file_, roster_n) || !read_raw(file_, roster_row_bytes)) {
        err_ = "roster header"; std::fclose(file_); file_ = nullptr; return false;
    }
    roster_.clear();
    roster_.reserve(roster_n);
    for (uint32_t i = 0; i < roster_n; ++i) roster_.push_back(read_row(file_, roster_schema_));

    // Walk to footer to grab frame offsets. The footer trailer is
    // (footer_off u64 + magic_end[8]) at the end of the file.
    if (std::fseek(file_, -16, SEEK_END) != 0) {
        err_ = "seek footer"; std::fclose(file_); file_ = nullptr; return false;
    }
    uint64_t footer_off = 0;
    if (!read_raw(file_, footer_off)) {
        err_ = "read footer offset"; std::fclose(file_); file_ = nullptr; return false;
    }
    char tail[8] = {0};
    std::fread(tail, 1, 8, file_);
    if (std::memcmp(tail, MAGIC_END, 8) != 0) {
        err_ = "bad footer magic"; std::fclose(file_); file_ = nullptr; return false;
    }
    if (std::fseek(file_, static_cast<long>(footer_off), SEEK_SET) != 0) {
        err_ = "seek to footer"; std::fclose(file_); file_ = nullptr; return false;
    }
    uint32_t nframes = 0;
    if (!read_raw(file_, nframes)) {
        err_ = "frame count"; std::fclose(file_); file_ = nullptr; return false;
    }
    frame_offsets_.resize(nframes);
    for (uint32_t i = 0; i < nframes; ++i) {
        if (!read_raw(file_, frame_offsets_[i])) {
            err_ = "frame offsets"; std::fclose(file_); file_ = nullptr; return false;
        }
    }
    return true;
}

GenericFrame GenericReplayReader::frame(size_t i) const {
    GenericFrame fr;
    if (!file_ || i >= frame_offsets_.size()) return fr;
    if (std::fseek(file_, static_cast<long>(frame_offsets_[i]), SEEK_SET) != 0) return fr;
    if (!read_raw(file_, fr.step_idx)) return {};
    if (!read_raw(file_, fr.elapsed))  return {};
    uint32_t nr = 0;
    read_raw(file_, nr);
    fr.rows.reserve(nr);
    for (uint32_t k = 0; k < nr; ++k) fr.rows.push_back(read_row(file_, frame_schema_));
    uint32_t ne = 0;
    read_raw(file_, ne);
    fr.events.reserve(ne);
    for (uint32_t k = 0; k < ne; ++k) fr.events.push_back(read_row(file_, event_schema_));
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
