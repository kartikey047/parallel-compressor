#include "pfc/codec.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace pfc {

std::vector<std::uint8_t> rle_encode(ByteView input) {
    std::vector<std::uint8_t> out;
    out.reserve(input.size / 4 + 16);
    std::size_t i = 0;
    while (i < input.size) {
        const std::uint8_t value = input.data[i];
        std::size_t run = 1;
        while (i + run < input.size && input.data[i + run] == value) ++run;
        i += run;
        while (run > 0) {
            const std::size_t count = std::min<std::size_t>(run, 255);
            out.push_back(static_cast<std::uint8_t>(count));
            out.push_back(value);
            run -= count;
        }
    }
    return out;
}

void rle_decode(ByteView input, MutableByteView out) {
    if (input.size % 2 != 0) throw FormatError("malformed RLE payload (odd length)");
    std::size_t written = 0;
    for (std::size_t i = 0; i < input.size; i += 2) {
        const std::size_t count = input.data[i];
        if (count == 0) throw FormatError("malformed RLE payload (zero run length)");
        if (count > out.size - written) throw FormatError("RLE payload expands beyond the chunk size");
        std::memset(out.data + written, input.data[i + 1], count);
        written += count;
    }
    if (written != out.size) throw FormatError("RLE payload is shorter than the chunk size");
}

std::vector<std::uint8_t> delta_forward(ByteView input) {
    std::vector<std::uint8_t> out(input.size);
    std::uint8_t previous = 0;
    for (std::size_t i = 0; i < input.size; ++i) {
        out[i] = static_cast<std::uint8_t>(input.data[i] - previous);
        previous = input.data[i];
    }
    return out;
}

void delta_inverse(MutableByteView data) {
    std::uint8_t running = 0;
    for (std::size_t i = 0; i < data.size; ++i) {
        running = static_cast<std::uint8_t>(running + data.data[i]);
        data.data[i] = running;
    }
}

EncodedChunk cpu_encode_chunk(ByteView raw, CodecChoice choice) {
    constexpr std::size_t kNotTried = std::numeric_limits<std::size_t>::max();
    if (choice == CodecChoice::raw || raw.size == 0) return {};

    std::vector<std::uint8_t> rle, delta_rle;
    if (choice == CodecChoice::automatic || choice == CodecChoice::rle) rle = rle_encode(raw);
    if (choice == CodecChoice::automatic || choice == CodecChoice::delta) {
        const std::vector<std::uint8_t> delta = delta_forward(raw);
        delta_rle = rle_encode({delta.data(), delta.size()});
    }

    const Codec codec = pick_codec(raw.size, rle.empty() ? kNotTried : rle.size(),
                                   delta_rle.empty() ? kNotTried : delta_rle.size());
    EncodedChunk chunk;
    chunk.codec = codec;
    if (codec == Codec::rle) chunk.payload = std::move(rle);
    if (codec == Codec::delta_rle) chunk.payload = std::move(delta_rle);
    return chunk;
}

void cpu_decode_chunk(Codec codec, ByteView payload, MutableByteView out) {
    switch (codec) {
        case Codec::raw:
            if (payload.size != out.size) throw FormatError("raw chunk has the wrong size");
            if (out.size > 0) std::memcpy(out.data, payload.data, out.size);
            return;
        case Codec::rle:
            rle_decode(payload, out);
            return;
        case Codec::delta_rle:
            rle_decode(payload, out);
            delta_inverse(out);
            return;
    }
    throw FormatError("unknown codec");
}

}
