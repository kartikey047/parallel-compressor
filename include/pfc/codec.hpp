

#pragma once

#include <cstdint>
#include <vector>

#include "pfc/common.hpp"
#include "pfc/format.hpp"

namespace pfc {

enum class CodecChoice {
    automatic,
    raw,
    rle,
    delta,
};

struct EncodedChunk {
    Codec codec = Codec::raw;
    std::vector<std::uint8_t> payload;
};

std::vector<std::uint8_t> rle_encode(ByteView input);

void rle_decode(ByteView input, MutableByteView out);

std::vector<std::uint8_t> delta_forward(ByteView input);
void delta_inverse(MutableByteView data);

inline Codec pick_codec(std::size_t raw_size, std::size_t rle_size, std::size_t delta_rle_size) {
    Codec best = Codec::raw;
    std::size_t best_size = raw_size;
    if (rle_size < best_size) {
        best = Codec::rle;
        best_size = rle_size;
    }
    if (delta_rle_size < best_size) best = Codec::delta_rle;
    return best;
}

EncodedChunk cpu_encode_chunk(ByteView raw, CodecChoice choice);

void cpu_decode_chunk(Codec codec, ByteView payload, MutableByteView out);

}
