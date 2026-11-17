

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pfc/codec.hpp"
#include "pfc/common.hpp"

namespace pfc {

struct GpuDecodeJob {
    Codec codec = Codec::rle;
    ByteView payload;
    MutableByteView out;
};

class GpuEngine {
public:
    virtual ~GpuEngine() = default;

    virtual std::vector<EncodedChunk> encode(const std::vector<ByteView>& chunks, CodecChoice choice) = 0;

    virtual void decode(const std::vector<GpuDecodeJob>& jobs) = 0;

    virtual std::string device_name() const = 0;
};

bool gpu_compiled();

bool gpu_available(int device, std::string* why_not = nullptr);

std::unique_ptr<GpuEngine> create_gpu_engine(int device, std::size_t max_chunk, int streams);

}
