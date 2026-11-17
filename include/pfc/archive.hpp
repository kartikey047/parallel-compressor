
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "pfc/codec.hpp"
#include "pfc/common.hpp"
#include "pfc/format.hpp"
#include "pfc/gpu.hpp"

namespace pfc {

enum class Backend {
    cpu,
    cuda,
    automatic,
};

const char* backend_name(Backend backend);
Backend parse_backend(const std::string& text);

struct Options {
    std::size_t chunk_size = kDefaultChunkSize;
    Backend backend = Backend::automatic;
    int threads = 0;
    int device = 0;
    int streams = 3;
    CodecChoice codec = CodecChoice::automatic;

    bool overwrite = false;
};

struct Report {
    Backend backend = Backend::cpu;
    std::string note;
    std::string gpu_device;
};

Report compress_file(const std::filesystem::path& input, const std::filesystem::path& output,
                     const Options& options = {});
Report decompress_file(const std::filesystem::path& archive, const std::filesystem::path& output,
                       const Options& options = {});

Report verify_file(const std::filesystem::path& archive, const Options& options = {});

int effective_threads(int requested);

}
