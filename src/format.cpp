#include "pfc/format.hpp"

#include <cstring>
#include <fstream>
#include <string>

#include "pfc/crc32.hpp"

namespace pfc {
namespace {

void put16(std::uint8_t* p, std::uint16_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
}
void put32(std::uint8_t* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void put64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
std::uint16_t get16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
std::uint32_t get32(const std::uint8_t* p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
    return v;
}
std::uint64_t get64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

[[noreturn]] void fail(const std::string& message) { throw FormatError(message); }

void check_payload_size(const IndexEntry& e, std::size_t chunk) {
    const std::string where = "chunk " + std::to_string(chunk) + ": ";
    if (e.codec == Codec::raw) {
        if (e.payload_size != e.raw_size) fail(where + "raw payload size differs from chunk size");
        return;
    }

    if (e.payload_size == 0 || e.payload_size % 2 != 0) fail(where + "malformed RLE payload size");
    if (e.payload_size > 2ull * e.raw_size) fail(where + "RLE payload larger than any valid encoding");
    if (e.raw_size > 255ull * (e.payload_size / 2)) fail(where + "RLE payload too short for the chunk size");
}

}

const char* codec_name(Codec codec) {
    switch (codec) {
        case Codec::raw: return "raw";
        case Codec::rle: return "rle";
        case Codec::delta_rle: return "delta+rle";
    }
    return "unknown";
}

bool codec_is_known(std::uint8_t value) { return value <= static_cast<std::uint8_t>(Codec::delta_rle); }

std::uint32_t chunk_count_for(std::uint64_t original_size, std::uint32_t chunk_size) {
    if (chunk_size == 0) return 0;
    return static_cast<std::uint32_t>((original_size + chunk_size - 1) / chunk_size);
}

std::uint32_t expected_raw_size(const Header& header, std::uint32_t chunk_index) {
    const std::uint64_t start = static_cast<std::uint64_t>(chunk_index) * header.chunk_size;
    const std::uint64_t left = header.original_size - start;
    return static_cast<std::uint32_t>(left < header.chunk_size ? left : header.chunk_size);
}

std::array<std::uint8_t, kHeaderSize> encode_header(const Header& header) {
    std::array<std::uint8_t, kHeaderSize> out{};
    std::memcpy(out.data(), kMagic, 4);
    put16(&out[4], kVersion);
    put16(&out[6], static_cast<std::uint16_t>(kHeaderSize));
    put32(&out[8], header.chunk_size);
    put32(&out[12], header.chunk_count);
    put64(&out[16], header.original_size);
    put64(&out[24], header.index_offset);
    put32(&out[32], header.index_crc);
    put32(&out[36], crc32(out.data(), 36));
    return out;
}

std::vector<std::uint8_t> encode_index(const std::vector<IndexEntry>& index) {
    std::vector<std::uint8_t> out(index.size() * kIndexEntrySize, 0);
    for (std::size_t i = 0; i < index.size(); ++i) {
        std::uint8_t* p = &out[i * kIndexEntrySize];
        p[0] = static_cast<std::uint8_t>(index[i].codec);
        put32(p + 4, index[i].raw_size);
        put64(p + 8, index[i].offset);
        put32(p + 16, index[i].payload_size);
        put32(p + 20, index[i].crc);
    }
    return out;
}

Header decode_header(const std::uint8_t* bytes) {
    if (std::memcmp(bytes, kMagic, 4) != 0) fail("not a PFC archive (bad magic)");
    const std::uint16_t version = get16(bytes + 4);
    if (version != kVersion) {
        fail("unsupported archive version " + std::to_string(version) + " (this build reads version " +
             std::to_string(kVersion) + ")");
    }
    if (get16(bytes + 6) != kHeaderSize) fail("unexpected header size");
    if (crc32(bytes, 36) != get32(bytes + 36)) fail("header checksum mismatch");

    Header header;
    header.chunk_size = get32(bytes + 8);
    header.chunk_count = get32(bytes + 12);
    header.original_size = get64(bytes + 16);
    header.index_offset = get64(bytes + 24);
    header.index_crc = get32(bytes + 32);
    return header;
}

std::vector<IndexEntry> decode_index(const std::uint8_t* bytes, std::size_t entry_count) {
    std::vector<IndexEntry> index(entry_count);
    for (std::size_t i = 0; i < entry_count; ++i) {
        const std::uint8_t* p = bytes + i * kIndexEntrySize;
        if (!codec_is_known(p[0])) fail("chunk " + std::to_string(i) + ": unknown codec id");
        if (p[1] != 0 || p[2] != 0 || p[3] != 0) fail("chunk " + std::to_string(i) + ": reserved bytes are not zero");
        index[i].codec = static_cast<Codec>(p[0]);
        index[i].raw_size = get32(p + 4);
        index[i].offset = get64(p + 8);
        index[i].payload_size = get32(p + 16);
        index[i].crc = get32(p + 20);
    }
    return index;
}

void validate_layout(const Layout& layout) {
    const Header& h = layout.header;
    if (h.chunk_size == 0 || h.chunk_size > kMaxChunkSize) fail("chunk size out of range");
    if (h.chunk_count != chunk_count_for(h.original_size, h.chunk_size)) {
        fail("chunk count does not match the original size and chunk size");
    }
    if (layout.index.size() != h.chunk_count) fail("index length does not match the chunk count");
    if (h.index_offset < kHeaderSize || h.index_offset > layout.file_size) fail("index offset outside the file");
    if (h.index_offset + static_cast<std::uint64_t>(h.chunk_count) * kIndexEntrySize != layout.file_size) {
        fail("file length does not match the index (truncated or has trailing data)");
    }

    std::uint64_t next_offset = kHeaderSize;
    for (std::uint32_t i = 0; i < h.chunk_count; ++i) {
        const IndexEntry& e = layout.index[i];
        if (e.raw_size != expected_raw_size(h, i)) fail("chunk " + std::to_string(i) + ": wrong chunk size");
        if (e.offset != next_offset) fail("chunk " + std::to_string(i) + ": offset is not contiguous");
        check_payload_size(e, i);
        next_offset += e.payload_size;
        if (next_offset > h.index_offset) fail("chunk " + std::to_string(i) + ": payload runs into the index");
    }
    if (next_offset != h.index_offset) fail("gap between the last payload and the index");
}

Layout read_layout(const std::filesystem::path& archive) {
    std::error_code ec;
    const std::uint64_t file_size = std::filesystem::file_size(archive, ec);
    if (ec) throw IoError("cannot read '" + archive.string() + "': " + ec.message());
    if (file_size < kHeaderSize) fail("file is too small to be a PFC archive");

    std::ifstream in(archive, std::ios::binary);
    if (!in) throw IoError("cannot open '" + archive.string() + "'");

    std::array<std::uint8_t, kHeaderSize> header_bytes{};
    if (!in.read(reinterpret_cast<char*>(header_bytes.data()), kHeaderSize)) throw IoError("cannot read the header");

    Layout layout;
    layout.file_size = file_size;
    layout.header = decode_header(header_bytes.data());
    const Header& h = layout.header;

    if (h.index_offset < kHeaderSize || h.index_offset > file_size ||
        h.index_offset + static_cast<std::uint64_t>(h.chunk_count) * kIndexEntrySize != file_size) {
        fail("file length does not match the index (truncated or has trailing data)");
    }

    std::vector<std::uint8_t> index_bytes(static_cast<std::size_t>(h.chunk_count) * kIndexEntrySize);
    in.seekg(static_cast<std::streamoff>(h.index_offset));
    if (!index_bytes.empty() && !in.read(reinterpret_cast<char*>(index_bytes.data()), index_bytes.size())) {
        throw IoError("cannot read the chunk index");
    }
    if (crc32(index_bytes.data(), index_bytes.size()) != h.index_crc) fail("index checksum mismatch");

    layout.index = decode_index(index_bytes.data(), h.chunk_count);
    validate_layout(layout);
    return layout;
}

}
