#include "pfc/crc32.hpp"

#include <array>

namespace pfc {
namespace {

std::array<std::uint32_t, 256> make_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t value = i;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value & 1) ? (value >> 1) ^ 0xEDB88320u : value >> 1;
        }
        table[i] = value;
    }
    return table;
}

}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t previous) {
    static const std::array<std::uint32_t, 256> table = make_table();
    std::uint32_t crc = ~previous;
    for (std::size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

}
