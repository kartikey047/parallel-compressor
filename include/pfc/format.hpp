

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "pfc/common.hpp"

namespace pfc {

enum class Codec : std::uint8_t {
    raw = 0,
    rle = 1,
    delta_rle = 2,
};

const char* codec_name(Codec codec);
bool codec_is_known(std::uint8_t value);

inline constexpr char kMagic[4] = {'P', 'F', 'C', 'A'};
inline constexpr std::uint16_t kVersion = 2;
inline constexpr std::size_t kHeaderSize = 40;
inline constexpr std::size_t kIndexEntrySize = 24;

struct Header {
    std::uint32_t chunk_size = 0;
    std::uint32_t chunk_count = 0;
    std::uint64_t original_size = 0;
    std::uint64_t index_offset = 0;
    std::uint32_t index_crc = 0;
};

struct IndexEntry {
    Codec codec = Codec::raw;
    std::uint32_t raw_size = 0;
    std::uint64_t offset = 0;
    std::uint32_t payload_size = 0;
    std::uint32_t crc = 0;
};

struct Layout {
    Header header;
    std::vector<IndexEntry> index;
    std::uint64_t file_size = 0;
};

std::uint32_t chunk_count_for(std::uint64_t original_size, std::uint32_t chunk_size);

std::array<std::uint8_t, kHeaderSize> encode_header(const Header& header);
std::vector<std::uint8_t> encode_index(const std::vector<IndexEntry>& index);

Header decode_header(const std::uint8_t* bytes);

std::vector<IndexEntry> decode_index(const std::uint8_t* bytes, std::size_t entry_count);

void validate_layout(const Layout& layout);

Layout read_layout(const std::filesystem::path& archive);

std::uint32_t expected_raw_size(const Header& header, std::uint32_t chunk_index);

}
