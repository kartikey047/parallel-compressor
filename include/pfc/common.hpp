
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pfc {

struct ByteView {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
};

struct MutableByteView {
    std::uint8_t* data = nullptr;
    std::size_t size = 0;
};

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class UsageError : public Error {
public:
    using Error::Error;
};

class IoError : public Error {
public:
    using Error::Error;
};

class FormatError : public Error {
public:
    using Error::Error;
};

class GpuError : public Error {
public:
    using Error::Error;
};

inline constexpr std::size_t kDefaultChunkSize = std::size_t{1} << 20;
inline constexpr std::size_t kMaxChunkSize = std::size_t{1} << 30;
inline constexpr std::size_t kGpuMaxChunkSize = std::size_t{32} << 20;

}
