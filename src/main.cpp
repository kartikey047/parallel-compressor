
#include <iostream>
#include <map>
#include <string>

#include "pfc/archive.hpp"
#include "pfc/format.hpp"
#include "pfc/gpu.hpp"

namespace {

const char* kUsage = R"(usage:
  compressor compress   INPUT  [-o OUTPUT] [options]
  compressor decompress ARCHIVE [-o OUTPUT] [options]
  compressor verify     ARCHIVE [options]
  compressor info       ARCHIVE
  compressor devices

options:
  -o, --output FILE     output file (default: INPUT.pfc, or ARCHIVE without .pfc)
  --backend cpu|cuda|auto   where the codec runs (default auto)
  --threads N           CPU worker threads (default: OpenMP default)
  --chunk-size BYTES    uncompressed bytes per chunk, K/M/G suffixes allowed (default 1M)
  --codec auto|raw|rle|delta   which codec to use per chunk (default auto: pick the smallest)
  --device N            CUDA device index (default 0)
  --streams N           concurrent GPU pipelines (default 3)
  -f, --force           replace the output file if it exists
  -q, --quiet           print nothing on success
  -h, --help            this text
)";

struct Args {
    std::string command;
    std::string input;
    std::string output;
    pfc::Options options;
    bool quiet = false;

};

std::size_t parse_size(const std::string& text) {
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw pfc::UsageError("not a number: '" + text + "'");
    }
    std::string suffix = text.substr(consumed);
    unsigned long long scale = 1;
    if (suffix == "K" || suffix == "k") scale = 1ull << 10;
    else if (suffix == "M" || suffix == "m") scale = 1ull << 20;
    else if (suffix == "G" || suffix == "g") scale = 1ull << 30;
    else if (!suffix.empty()) throw pfc::UsageError("bad size suffix in '" + text + "'");
    return static_cast<std::size_t>(value * scale);
}

int parse_int(const std::string& text) {
    try {
        return std::stoi(text);
    } catch (const std::exception&) {
        throw pfc::UsageError("not a number: '" + text + "'");
    }
}

pfc::CodecChoice parse_codec(const std::string& text) {
    if (text == "auto") return pfc::CodecChoice::automatic;
    if (text == "raw") return pfc::CodecChoice::raw;
    if (text == "rle") return pfc::CodecChoice::rle;
    if (text == "delta") return pfc::CodecChoice::delta;
    throw pfc::UsageError("codec must be auto, raw, rle or delta (got '" + text + "')");
}

Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2) throw pfc::UsageError("missing command");
    args.command = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw pfc::UsageError("option " + a + " needs a value");
            return argv[++i];
        };
        if (a == "-o" || a == "--output") args.output = value();
        else if (a == "--backend") args.options.backend = pfc::parse_backend(value());
        else if (a == "--threads") args.options.threads = parse_int(value());
        else if (a == "--chunk-size") args.options.chunk_size = parse_size(value());
        else if (a == "--codec") args.options.codec = parse_codec(value());
        else if (a == "--device") args.options.device = parse_int(value());
        else if (a == "--streams") args.options.streams = parse_int(value());
        else if (a == "-f" || a == "--force") args.options.overwrite = true;
        else if (a == "-q" || a == "--quiet") args.quiet = true;
        else if (a == "-h" || a == "--help") { std::cout << kUsage; std::exit(0); }
        else if (!a.empty() && a[0] == '-') throw pfc::UsageError("unknown option " + a);
        else if (args.input.empty()) args.input = a;
        else throw pfc::UsageError("unexpected argument '" + a + "'");
    }
    return args;
}

std::string default_output(const Args& args) {
    if (args.command == "compress") return args.input + ".pfc";
    const std::string suffix = ".pfc";
    if (args.input.size() > suffix.size() && args.input.compare(args.input.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return args.input.substr(0, args.input.size() - suffix.size());
    }
    return args.input + ".out";
}

void print_report(const char* verb, const std::string& input, const pfc::Report& r) {
    std::cout << verb << " " << input << " with " << pfc::backend_name(r.backend) << "\n";
    if (!r.note.empty()) std::cout << "  note: " << r.note << "\n";
}

int run(const Args& args) {
    if (args.command == "-h" || args.command == "--help") {
        std::cout << kUsage;
        return 0;
    }

    if (args.command == "devices") {
        std::string why_not;
        std::cout << "CUDA support compiled in: " << (pfc::gpu_compiled() ? "yes" : "no") << "\n";
        if (pfc::gpu_available(args.options.device, &why_not)) {
            std::cout << "device " << args.options.device << ": " << pfc::create_gpu_engine(args.options.device, 1 << 20, 1)->device_name() << "\n";
        } else {
            std::cout << "device " << args.options.device << ": not available (" << why_not << ")\n";
        }
        return 0;
    }

    if (args.input.empty()) throw pfc::UsageError("missing input file");

    if (args.command == "compress") {
        const pfc::Report r = pfc::compress_file(args.input, args.output.empty() ? default_output(args) : args.output, args.options);
        if (!args.quiet) print_report("compressed", args.input, r);
    } else if (args.command == "decompress") {
        const pfc::Report r = pfc::decompress_file(args.input, args.output.empty() ? default_output(args) : args.output, args.options);
        if (!args.quiet) print_report("restored", args.input, r);
    } else if (args.command == "verify") {
        const pfc::Report r = pfc::verify_file(args.input, args.options);
        if (!args.quiet) print_report("verified OK", args.input, r);
    } else if (args.command == "info") {
        const pfc::Layout layout = pfc::read_layout(args.input);
        std::map<pfc::Codec, std::uint64_t> stored;
        std::map<pfc::Codec, std::uint32_t> count;
        for (const pfc::IndexEntry& e : layout.index) {
            stored[e.codec] += e.payload_size;
            ++count[e.codec];
        }
        std::cout << "format version " << pfc::kVersion << ", " << layout.file_size << " bytes in the archive\n"
                  << "original size " << layout.header.original_size << " bytes, " << layout.header.chunk_count
                  << " chunks of " << layout.header.chunk_size << " bytes\n";
        for (auto [codec, n] : count) {
            std::cout << "  " << pfc::codec_name(codec) << ": " << n << " chunks, " << stored[codec] << " bytes stored\n";
        }
    } else {
        throw pfc::UsageError("unknown command '" + args.command + "'");
    }
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const pfc::UsageError& error) {
        std::cerr << "error: " << error.what() << "\n\n" << kUsage;
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
