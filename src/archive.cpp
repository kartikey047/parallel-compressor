#include "pfc/archive.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pfc/crc32.hpp"

namespace pfc {
namespace {

template <class Body>
void parallel_for(std::int64_t count, int threads, Body&& body) {
    std::exception_ptr failure;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads)
#endif
    for (std::int64_t i = 0; i < count; ++i) {
        try {
            body(i);
        } catch (...) {
#ifdef _OPENMP
#pragma omp critical(pfc_failure)
#endif
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
}

void check_options(const Options& options) {
    if (options.chunk_size == 0 || options.chunk_size > kMaxChunkSize) {
        throw UsageError("chunk size must be between 1 and " + std::to_string(kMaxChunkSize) + " bytes");
    }
    if (options.threads < 0) throw UsageError("threads must not be negative");
    if (options.streams < 1 || options.streams > 16) throw UsageError("streams must be between 1 and 16");
}

std::unique_ptr<GpuEngine> choose_backend(const Options& options, std::size_t max_chunk, std::string& note) {
    if (options.backend == Backend::cpu) return nullptr;

    if (options.backend == Backend::cuda) {
        if (max_chunk > kGpuMaxChunkSize) {
            throw UsageError("the CUDA backend supports chunks up to " + std::to_string(kGpuMaxChunkSize) + " bytes");
        }
        return create_gpu_engine(options.device, max_chunk, options.streams);
    }

    std::string why_not;
    if (!gpu_available(options.device, &why_not)) {
        note = "using CPU: " + why_not;
    } else if (max_chunk > kGpuMaxChunkSize) {
        note = "using CPU: chunk size is above the GPU limit";
    } else {
        try {
            return create_gpu_engine(options.device, max_chunk, options.streams);
        } catch (const GpuError& error) {
            note = std::string("using CPU: GPU setup failed (") + error.what() + ")";
        }
    }
    return nullptr;
}

std::filesystem::path partial_path(const std::filesystem::path& output) {
    std::filesystem::path partial = output;
    partial += ".partial";
    return partial;
}

void check_output_allowed(const std::filesystem::path& output, const Options& options) {
    std::error_code ec;
    if (!options.overwrite && std::filesystem::exists(output, ec)) {
        throw IoError("'" + output.string() + "' already exists (use --force to replace it)");
    }
}

std::uint64_t file_size_or_throw(const std::filesystem::path& path) {
    std::error_code ec;
    const std::uint64_t size = std::filesystem::file_size(path, ec);
    if (ec) throw IoError("cannot read '" + path.string() + "': " + ec.message());
    return size;
}

void write_bytes(std::ofstream& out, const std::uint8_t* data, std::size_t size) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) throw IoError("write failed (disk full?)");
}

void fill_gpu_report(Report& report, const GpuEngine* engine) {
    if (!engine) return;
    report.backend = Backend::cuda;
    report.gpu_device = engine->device_name();
}

void compress_with(const std::filesystem::path& input, const std::filesystem::path& partial, const Options& options,
                   GpuEngine* engine, int threads) {
    const std::uint64_t original_size = file_size_or_throw(input);
    if (original_size > 0xFFFFFFFFull * options.chunk_size) throw UsageError("file needs more than 2^32 chunks");

    std::ifstream in(input, std::ios::binary);
    if (!in) throw IoError("cannot open '" + input.string() + "'");
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out) throw IoError("cannot create '" + partial.string() + "'");

    const std::array<std::uint8_t, kHeaderSize> placeholder{};
    write_bytes(out, placeholder.data(), placeholder.size());

    const std::size_t batch = engine ? static_cast<std::size_t>(options.streams) * 2 : static_cast<std::size_t>(threads) * 2;
    std::vector<std::vector<std::uint8_t>> buffers(batch);
    std::vector<IndexEntry> index;
    std::uint64_t remaining = original_size;
    std::uint64_t payload_offset = kHeaderSize;

    while (remaining > 0) {

        std::vector<ByteView> views;
        for (std::size_t k = 0; k < batch && remaining > 0; ++k) {
            const std::size_t bytes = static_cast<std::size_t>(std::min<std::uint64_t>(options.chunk_size, remaining));
            buffers[k].resize(bytes);
            if (!in.read(reinterpret_cast<char*>(buffers[k].data()), static_cast<std::streamsize>(bytes))) {
                throw IoError("read failed on '" + input.string() + "'");
            }
            views.push_back({buffers[k].data(), bytes});
            remaining -= bytes;
        }
        const std::int64_t n = static_cast<std::int64_t>(views.size());
        std::vector<EncodedChunk> encoded(views.size());
        std::vector<std::uint32_t> crcs(views.size());
        if (engine) {
            encoded = engine->encode(views, options.codec);
            parallel_for(n, threads, [&](std::int64_t i) { crcs[i] = crc32(views[i].data, views[i].size); });
        } else {
            parallel_for(n, threads, [&](std::int64_t i) {
                encoded[i] = cpu_encode_chunk(views[i], options.codec);
                crcs[i] = crc32(views[i].data, views[i].size);
            });
        }
        for (std::size_t i = 0; i < views.size(); ++i) {
            const bool is_raw = encoded[i].codec == Codec::raw;
            const std::uint8_t* data = is_raw ? views[i].data : encoded[i].payload.data();
            const std::size_t size = is_raw ? views[i].size : encoded[i].payload.size();
            write_bytes(out, data, size);

            IndexEntry entry;
            entry.codec = encoded[i].codec;
            entry.raw_size = static_cast<std::uint32_t>(views[i].size);
            entry.offset = payload_offset;
            entry.payload_size = static_cast<std::uint32_t>(size);
            entry.crc = crcs[i];
            index.push_back(entry);
            payload_offset += size;
        }
    }

    const std::vector<std::uint8_t> index_bytes = encode_index(index);
    Header header;
    header.chunk_size = static_cast<std::uint32_t>(options.chunk_size);
    header.chunk_count = static_cast<std::uint32_t>(index.size());
    header.original_size = original_size;
    header.index_offset = payload_offset;
    header.index_crc = crc32(index_bytes.data(), index_bytes.size());
    write_bytes(out, index_bytes.data(), index_bytes.size());
    out.seekp(0);
    const auto header_bytes = encode_header(header);
    write_bytes(out, header_bytes.data(), header_bytes.size());
    out.close();
    if (!out) throw IoError("could not finish writing '" + partial.string() + "'");
}

void decode_with(const std::filesystem::path& archive, const Layout& layout, std::ofstream* out, const Options& options,
                 GpuEngine* engine, int threads) {
    std::ifstream in(archive, std::ios::binary);
    if (!in) throw IoError("cannot open '" + archive.string() + "'");

    const std::size_t batch = engine ? static_cast<std::size_t>(options.streams) * 2 : static_cast<std::size_t>(threads) * 2;
    std::vector<std::vector<std::uint8_t>> payloads(batch), restored(batch);
    const std::uint32_t total = layout.header.chunk_count;
    in.seekg(static_cast<std::streamoff>(kHeaderSize));

    for (std::uint32_t first = 0; first < total; first += static_cast<std::uint32_t>(batch)) {
        const std::size_t n = std::min<std::size_t>(batch, total - first);

        for (std::size_t k = 0; k < n; ++k) {
            const IndexEntry& e = layout.index[first + k];
            payloads[k].resize(e.payload_size);
            restored[k].resize(e.raw_size);
            if (!in.read(reinterpret_cast<char*>(payloads[k].data()), e.payload_size)) {
                throw FormatError("chunk " + std::to_string(first + k) + ": payload is truncated");
            }
        }
        auto decode_cpu = [&](std::int64_t k) {
            const IndexEntry& e = layout.index[first + k];
            cpu_decode_chunk(e.codec, {payloads[k].data(), payloads[k].size()}, {restored[k].data(), restored[k].size()});
        };
        if (engine) {
            std::vector<GpuDecodeJob> jobs;
            std::vector<std::size_t> raw_chunks;
            for (std::size_t k = 0; k < n; ++k) {
                const IndexEntry& e = layout.index[first + k];
                if (e.codec == Codec::raw) {
                    raw_chunks.push_back(k);
                } else {
                    jobs.push_back({e.codec, {payloads[k].data(), payloads[k].size()}, {restored[k].data(), restored[k].size()}});
                }
            }
            for (std::size_t k : raw_chunks) decode_cpu(static_cast<std::int64_t>(k));
            if (!jobs.empty()) engine->decode(jobs);
        } else {
            parallel_for(static_cast<std::int64_t>(n), threads, decode_cpu);
        }
        parallel_for(static_cast<std::int64_t>(n), threads, [&](std::int64_t k) {
            const IndexEntry& e = layout.index[first + k];
            if (crc32(restored[k].data(), restored[k].size()) != e.crc) {
                throw FormatError("chunk " + std::to_string(first + k) + ": checksum mismatch (data is corrupt)");
            }
        });
        for (std::size_t k = 0; k < n; ++k) {
            if (out) write_bytes(*out, restored[k].data(), restored[k].size());
        }
    }

}

Report run_decode(const std::filesystem::path& archive, const std::filesystem::path* output, const Options& options) {
    check_options(options);
    const Layout layout = read_layout(archive);

    Report report;
    const int threads = effective_threads(options.threads);
    report.backend = Backend::cpu;

    std::unique_ptr<GpuEngine> engine = choose_backend(options, layout.header.chunk_size, report.note);

    std::filesystem::path partial;
    std::ofstream out_stream;
    if (output) {
        check_output_allowed(*output, options);
        partial = partial_path(*output);
        out_stream.open(partial, std::ios::binary | std::ios::trunc);
        if (!out_stream) throw IoError("cannot create '" + partial.string() + "'");
    }

    auto cleanup = [&] {
        if (output) {
            out_stream.close();
            std::error_code ec;
            std::filesystem::remove(partial, ec);
        }
    };

    try {
        try {
            decode_with(archive, layout, output ? &out_stream : nullptr, options, engine.get(), threads);
        } catch (const GpuError& error) {
            if (options.backend != Backend::automatic) throw;

            report.note = std::string("GPU failed (") + error.what() + "); redone on the CPU";
            engine.reset();
            report = Report{};
            report.note = "GPU failed; redone on the CPU";
            if (output) {
                out_stream.close();
                out_stream.open(partial, std::ios::binary | std::ios::trunc);
            }
            decode_with(archive, layout, output ? &out_stream : nullptr, options, nullptr, threads);
        }
        if (output) {
            out_stream.close();
            if (!out_stream) throw IoError("could not finish writing '" + partial.string() + "'");
            std::filesystem::rename(partial, *output);
        }
    } catch (...) {
        cleanup();
        throw;
    }

    fill_gpu_report(report, engine.get());
    return report;
}

}

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::cpu: return "cpu";
        case Backend::cuda: return "cuda";
        case Backend::automatic: return "auto";
    }
    return "?";
}

Backend parse_backend(const std::string& text) {
    if (text == "cpu") return Backend::cpu;
    if (text == "cuda") return Backend::cuda;
    if (text == "auto") return Backend::automatic;
    throw UsageError("backend must be cpu, cuda or auto (got '" + text + "')");
}

int effective_threads(int requested) {
    if (requested > 0) return requested;
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

Report compress_file(const std::filesystem::path& input, const std::filesystem::path& output, const Options& options) {
    check_options(options);
    check_output_allowed(output, options);

    Report report;
    const int threads = effective_threads(options.threads);
    std::unique_ptr<GpuEngine> engine =
        choose_backend(options, options.chunk_size, report.note);
    const std::filesystem::path partial = partial_path(output);
    std::string note = report.note;

    auto attempt = [&](GpuEngine* gpu) {
        report = Report{};
        compress_with(input, partial, options, gpu, threads);
    };

    try {
        try {
            attempt(engine.get());
        } catch (const GpuError& error) {
            if (options.backend != Backend::automatic) throw;
            engine.reset();
            attempt(nullptr);
            note = std::string("GPU failed (") + error.what() + "); redone on the CPU";
        }
        std::filesystem::rename(partial, output);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(partial, ec);
        throw;
    }

    report.note = note;
    report.backend = Backend::cpu;
    fill_gpu_report(report, engine.get());
    return report;
}

Report decompress_file(const std::filesystem::path& archive, const std::filesystem::path& output, const Options& options) {
    return run_decode(archive, &output, options);
}

Report verify_file(const std::filesystem::path& archive, const Options& options) {
    return run_decode(archive, nullptr, options);
}

}
