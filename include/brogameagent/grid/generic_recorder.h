#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

namespace brogameagent::grid {

// ─── Schema-driven replay (.bgargrid) ─────────────────────────────────────
//
// A binary record format keyed on caller-supplied schemas: one schema for
// the static roster (per-entity rows written once at start), one schema
// for per-frame entity rows, one schema for per-frame events. Lets you
// record episodes from envs whose entity model doesn't match the bundled
// MOBA Unit shape (gridworld tiles, platformer mobs, puzzle pieces, …).
//
// Differences from the MOBA `.bgar`:
//   - Not byte-compatible. Distinct magic ("BGAGRID\0").
//   - No fixed roster/frame structs — fields are described at open time.
//   - Field types are numeric-only (i32/i64/f32/f64). Strings can be
//     hashed to ints by the caller, or split into per-character u8 fields.
//   - Frame-row count is variable per frame (you can spawn / despawn).
//
// Wire layout (little-endian implicit on x86 + ARM, no portability claim):
//
//     magic[8]               "BGAGRID\0"
//     version u32            == 1
//     episode_id u64
//     seed u64
//     dt f32
//     // schemas
//     roster_schema { num_fields u32, field defs }
//     frame_schema  { num_fields u32, field defs }
//     event_schema  { num_fields u32, field defs }
//     // each field def = { name_len u16, name bytes, type u8 }
//     // roster block
//     num_roster_rows u32
//     roster_row_bytes u32     // == sum of field sizes
//     packed roster rows
//     // frames (streamed)
//     ... per frame ...
//     // footer
//     num_frames u32
//     u64 frame_offsets[num_frames]
//     u64 footer_offset        // file pos where num_frames starts
//     magic_footer[8]          "BGAGEND\0"
//
// Per-frame block:
//     step_idx u64
//     elapsed f32
//     num_rows u32, packed rows
//     num_events u32, packed events

enum class FieldType : uint8_t {
    I32 = 1,
    I64 = 2,
    F32 = 3,
    F64 = 4,
};

struct FieldDef {
    std::string name;
    FieldType   type;
};

// One numeric value per field. The caller sets the alternative matching
// the schema — mismatch is zero-padded on write.
using FieldValue = std::variant<int32_t, int64_t, float, double>;

// One row = one value per field, in schema order.
using Row = std::vector<FieldValue>;

class GenericRecorder {
public:
    GenericRecorder() = default;
    ~GenericRecorder();
    GenericRecorder(const GenericRecorder&) = delete;
    GenericRecorder& operator=(const GenericRecorder&) = delete;

    // Open a file for writing. The schemas are immutable for the lifetime
    // of the file. event_schema may be empty if the format has no events.
    bool open(const std::string& path,
              uint64_t episode_id, uint64_t seed, float dt,
              std::vector<FieldDef> roster_schema,
              std::vector<FieldDef> frame_schema,
              std::vector<FieldDef> event_schema = {});

    bool is_open() const { return file_ != nullptr; }

    // Call once, before any frames. Mismatched-arity rows are skipped.
    void write_roster(const std::vector<Row>& rows);

    // Record one frame. `events` defaults to none.
    void record_frame(uint64_t step_idx, float elapsed,
                      const std::vector<Row>& rows,
                      const std::vector<Row>& events = {});

    bool close();

    size_t frame_count() const { return frame_offsets_.size(); }

private:
    std::FILE*               file_ = nullptr;
    std::vector<FieldDef>    roster_schema_;
    std::vector<FieldDef>    frame_schema_;
    std::vector<FieldDef>    event_schema_;
    uint32_t                 roster_row_bytes_ = 0;
    uint32_t                 frame_row_bytes_  = 0;
    uint32_t                 event_row_bytes_  = 0;
    bool                     roster_written_ = false;
    std::vector<uint64_t>    frame_offsets_;
};

// ─── Reader ───────────────────────────────────────────────────────────────

struct GenericFrame {
    uint64_t          step_idx = 0;
    float             elapsed  = 0.0f;
    std::vector<Row>  rows;
    std::vector<Row>  events;
};

class GenericReplayReader {
public:
    GenericReplayReader() = default;
    ~GenericReplayReader();
    GenericReplayReader(const GenericReplayReader&) = delete;
    GenericReplayReader& operator=(const GenericReplayReader&) = delete;

    // Returns false on I/O error or format mismatch. error_message() has
    // a human-readable explanation in that case.
    bool open(const std::string& path);
    bool is_open() const { return file_ != nullptr; }
    const std::string& error_message() const { return err_; }

    uint64_t episode_id() const { return episode_id_; }
    uint64_t seed()       const { return seed_; }
    float    dt()         const { return dt_; }

    const std::vector<FieldDef>& roster_schema() const { return roster_schema_; }
    const std::vector<FieldDef>& frame_schema()  const { return frame_schema_;  }
    const std::vector<FieldDef>& event_schema()  const { return event_schema_;  }

    const std::vector<Row>& roster() const { return roster_; }

    size_t frame_count() const { return frame_offsets_.size(); }

    // Read frame `i` (0-based). Returns an empty frame on out-of-range.
    GenericFrame frame(size_t i) const;

    // Pull one named field's value across all frames for entities that
    // appear in every frame at a stable row index. The caller picks
    // `row_index` (typically 0 if the schema records a single tracked
    // entity, or computed from a roster lookup). Frames where row_index
    // is out of range yield a default-zero value of the field's type.
    std::vector<FieldValue> trajectory(size_t row_index,
                                       const std::string& field_name) const;

private:
    mutable std::FILE*       file_ = nullptr;
    std::string              err_;
    uint64_t                 episode_id_ = 0;
    uint64_t                 seed_       = 0;
    float                    dt_         = 0.0f;
    std::vector<FieldDef>    roster_schema_;
    std::vector<FieldDef>    frame_schema_;
    std::vector<FieldDef>    event_schema_;
    uint32_t                 frame_row_bytes_ = 0;
    uint32_t                 event_row_bytes_ = 0;
    std::vector<Row>         roster_;
    std::vector<uint64_t>    frame_offsets_;
};

} // namespace brogameagent::grid
