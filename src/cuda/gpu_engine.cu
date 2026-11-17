

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.cuh"
#include "pfc/gpu.hpp"

namespace pfc {
namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw GpuError(std::string(what) + ": " + cudaGetErrorString(status));
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) { check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) { o.ptr_ = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(count_, o.count_);
        return *this;
    }
    T* get() const { return ptr_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

template <class T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count) : count_(count) { check(cudaMallocHost(&ptr_, count * sizeof(T)), "cudaMallocHost"); }
    ~PinnedBuffer() { if (ptr_) cudaFreeHost(ptr_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) { o.ptr_ = nullptr; }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(count_, o.count_);
        return *this;
    }
    T* get() const { return ptr_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() { check(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

struct Workspace {
    explicit Workspace(std::size_t capacity, std::size_t scan_bytes)
        : in(capacity), aux(capacity), pay(2 * capacity), out0(2 * capacity), out1(2 * capacity),
          scratch(4 * capacity), meta(8), scan(scan_bytes),
          h_in(capacity), h_out(2 * capacity), h_pay(2 * capacity), h_res(capacity), h_meta(8) {}

    Stream stream;
    DeviceBuffer<std::uint8_t> in;
    DeviceBuffer<std::uint8_t> aux;
    DeviceBuffer<std::uint8_t> pay;
    DeviceBuffer<std::uint8_t> out0;
    DeviceBuffer<std::uint8_t> out1;
    DeviceBuffer<std::uint32_t> scratch;
    DeviceBuffer<std::uint64_t> meta;
    DeviceBuffer<std::uint8_t> scan;

    PinnedBuffer<std::uint8_t> h_in, h_out, h_pay, h_res;
    PinnedBuffer<std::uint64_t> h_meta;

};

struct EncodeTicket {
    bool skipped = false;
    bool want_rle = false;
    bool want_delta = false;
};

class CudaEngine final : public GpuEngine {
public:
    CudaEngine(int device, std::size_t max_chunk, int streams) : device_(device), max_chunk_(max_chunk) {
        check(cudaSetDevice(device), "cudaSetDevice");
        capacity_ = (max_chunk + 7) & ~std::size_t{7};
        const std::size_t scan_bytes = gpu::scan_scratch_bytes(static_cast<std::uint32_t>(capacity_));
        for (int i = 0; i < streams; ++i) workspaces_.push_back(std::make_unique<Workspace>(capacity_, scan_bytes));
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
        name_ = props.name;
    }

    std::string device_name() const override { return name_; }

    std::vector<EncodedChunk> encode(const std::vector<ByteView>& chunks, CodecChoice choice) override {
        select_device();
        std::vector<EncodedChunk> results(chunks.size());
        const std::size_t streams = workspaces_.size();
        for (std::size_t base = 0; base < chunks.size(); base += streams) {
            const std::size_t count = std::min(streams, chunks.size() - base);
            std::vector<EncodeTicket> tickets(count);
            for (std::size_t k = 0; k < count; ++k) tickets[k] = launch_encode(*workspaces_[k], chunks[base + k], choice);
            for (std::size_t k = 0; k < count; ++k) results[base + k] = collect_encode(*workspaces_[k], chunks[base + k], tickets[k]);
        }
        return results;
    }

    void decode(const std::vector<GpuDecodeJob>& jobs) override {
        select_device();
        const std::size_t streams = workspaces_.size();
        for (std::size_t base = 0; base < jobs.size(); base += streams) {
            const std::size_t count = std::min(streams, jobs.size() - base);
            for (std::size_t k = 0; k < count; ++k) launch_decode(*workspaces_[k], jobs[base + k]);
            for (std::size_t k = 0; k < count; ++k) collect_decode(*workspaces_[k], jobs[base + k]);
        }
    }

private:
    void select_device() { check(cudaSetDevice(device_), "cudaSetDevice"); }

    void run_rle_pipeline(Workspace& ws, const std::uint8_t* src, std::uint32_t n, std::uint8_t* out, std::uint64_t* meta) {
        cudaStream_t s = ws.stream.get();
        std::uint32_t* flags = ws.scratch.get();
        std::uint32_t* run_index = flags + capacity_;
        std::uint32_t* starts = run_index + capacity_;
        std::uint32_t* token_offset = starts + capacity_;
        void* scan = ws.scan.get();
        const std::size_t scan_bytes = ws.scan.bytes();

        check(gpu::mark_run_starts(src, flags, n, s), "mark_run_starts");
        check(gpu::exclusive_scan(scan, scan_bytes, flags, run_index, n, s), "run index scan");
        check(gpu::finalize_runs(flags, run_index, n, meta, s), "finalize_runs");
        check(gpu::gather_run_starts(flags, run_index, starts, n, s), "gather_run_starts");
        check(gpu::count_tokens(starts, meta, n, flags, s), "count_tokens");
        check(gpu::exclusive_scan(scan, scan_bytes, flags, token_offset, n, s), "token offset scan");
        check(gpu::finalize_tokens(flags, token_offset, n, meta, s), "finalize_tokens");
        check(gpu::emit_tokens(src, starts, token_offset, meta, n, out, s), "emit_tokens");
    }

    EncodeTicket launch_encode(Workspace& ws, ByteView chunk, CodecChoice choice) {
        EncodeTicket ticket;
        if (chunk.size > max_chunk_) throw GpuError("chunk is larger than the GPU engine's chunk size");
        if (chunk.size == 0 || choice == CodecChoice::raw) {
            ticket.skipped = true;
            return ticket;
        }
        ticket.want_rle = choice == CodecChoice::automatic || choice == CodecChoice::rle;
        ticket.want_delta = choice == CodecChoice::automatic || choice == CodecChoice::delta;

        const std::uint32_t n = static_cast<std::uint32_t>(chunk.size);
        cudaStream_t s = ws.stream.get();
        std::memcpy(ws.h_in.get(), chunk.data, chunk.size);

        check(cudaMemcpyAsync(ws.in.get(), ws.h_in.get(), n, cudaMemcpyHostToDevice, s), "upload chunk");

        check(cudaMemsetAsync(ws.meta.get(), 0, 4 * sizeof(std::uint64_t), s), "clear meta");
        if (ticket.want_rle) run_rle_pipeline(ws, ws.in.get(), n, ws.out0.get(), ws.meta.get());
        if (ticket.want_delta) {
            check(gpu::delta_forward(ws.in.get(), ws.aux.get(), n, s), "delta_forward");
            run_rle_pipeline(ws, ws.aux.get(), n, ws.out1.get(), ws.meta.get() + 2);
        }
        check(cudaMemcpyAsync(ws.h_meta.get(), ws.meta.get(), 4 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, s), "download sizes");
        return ticket;
    }

    EncodedChunk collect_encode(Workspace& ws, ByteView chunk, const EncodeTicket& ticket) {
        if (ticket.skipped) return {};
        constexpr std::size_t kNotTried = std::numeric_limits<std::size_t>::max();
        check(cudaStreamSynchronize(ws.stream.get()), "wait for sizes");

        const std::uint64_t* meta = ws.h_meta.get();
        const std::size_t rle_size = ticket.want_rle ? 2 * static_cast<std::size_t>(meta[1]) : kNotTried;
        const std::size_t delta_size = ticket.want_delta ? 2 * static_cast<std::size_t>(meta[3]) : kNotTried;
        const Codec codec = pick_codec(chunk.size, rle_size, delta_size);

        EncodedChunk result;
        result.codec = codec;
        if (codec == Codec::raw) return result;

        cudaStream_t s = ws.stream.get();
        const std::size_t size = codec == Codec::rle ? rle_size : delta_size;
        const std::uint8_t* device_tokens = codec == Codec::rle ? ws.out0.get() : ws.out1.get();
        check(cudaMemcpyAsync(ws.h_out.get(), device_tokens, size, cudaMemcpyDeviceToHost, s), "download payload");
        check(cudaStreamSynchronize(s), "wait for payload");

        result.payload.assign(ws.h_out.get(), ws.h_out.get() + size);
        return result;
    }

    void launch_decode(Workspace& ws, const GpuDecodeJob& job) {
        if (job.codec != Codec::rle && job.codec != Codec::delta_rle) throw GpuError("the GPU decoder only handles RLE chunks");
        if (job.out.size == 0 || job.out.size > max_chunk_) throw GpuError("chunk size is outside what the GPU engine was built for");
        if (job.payload.size == 0 || job.payload.size % 2 != 0 || job.payload.size > 2 * max_chunk_) {
            throw FormatError("malformed RLE payload size");
        }

        const std::uint32_t tokens = static_cast<std::uint32_t>(job.payload.size / 2);
        const std::uint32_t raw_size = static_cast<std::uint32_t>(job.out.size);
        cudaStream_t s = ws.stream.get();
        std::uint64_t* counts = reinterpret_cast<std::uint64_t*>(ws.scratch.get());
        std::uint64_t* offsets = counts + capacity_;
        std::memcpy(ws.h_pay.get(), job.payload.data, job.payload.size);

        check(cudaMemcpyAsync(ws.pay.get(), ws.h_pay.get(), job.payload.size, cudaMemcpyHostToDevice, s), "upload payload");

        check(cudaMemsetAsync(ws.meta.get(), 0, 2 * sizeof(std::uint64_t), s), "clear meta");
        check(gpu::read_counts(ws.pay.get(), tokens, counts, ws.meta.get(), s), "read_counts");
        check(gpu::exclusive_scan(ws.scan.get(), ws.scan.bytes(), counts, offsets, tokens, s), "offset scan");
        check(gpu::finalize_decode(counts, offsets, tokens, ws.meta.get(), s), "finalize_decode");
        check(gpu::fill_output(ws.pay.get(), offsets, ws.meta.get(), tokens, raw_size, ws.aux.get(), s), "fill_output");

        const std::uint8_t* result = ws.aux.get();
        if (job.codec == Codec::delta_rle) {
            check(gpu::inclusive_scan_mod256(ws.scan.get(), ws.scan.bytes(), ws.aux.get(), ws.in.get(), raw_size, s), "inverse delta scan");
            result = ws.in.get();
        }
        check(cudaMemcpyAsync(ws.h_meta.get(), ws.meta.get(), 2 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, s), "download status");
        check(cudaMemcpyAsync(ws.h_res.get(), result, raw_size, cudaMemcpyDeviceToHost, s), "download result");
    }

    void collect_decode(Workspace& ws, const GpuDecodeJob& job) {
        check(cudaStreamSynchronize(ws.stream.get()), "wait for decode");

        const std::uint64_t decoded_length = ws.h_meta.get()[0];
        const std::uint64_t zero_length_tokens = ws.h_meta.get()[1];
        if (zero_length_tokens != 0) throw FormatError("malformed RLE payload (zero run length)");
        if (decoded_length != job.out.size) throw FormatError("RLE payload does not decode to the chunk size");
        std::memcpy(job.out.data, ws.h_res.get(), job.out.size);
    }

    int device_;
    std::size_t max_chunk_;
    std::size_t capacity_ = 0;
    std::string name_;
    std::vector<std::unique_ptr<Workspace>> workspaces_;
};

}

bool gpu_compiled() { return true; }

bool gpu_available(int device, std::string* why_not) {
    auto fail = [&](const std::string& reason) {
        if (why_not) *why_not = reason;
        return false;
    };
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) return fail(std::string("CUDA runtime error: ") + cudaGetErrorString(status));
    if (count == 0) return fail("no CUDA device found");
    if (device < 0 || device >= count) return fail("device " + std::to_string(device) + " does not exist (found " + std::to_string(count) + ")");
    if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        return fail(std::string("cannot create a context on device ") + std::to_string(device) + ": " + cudaGetErrorString(cudaGetLastError()));
    }
    return true;
}

std::unique_ptr<GpuEngine> create_gpu_engine(int device, std::size_t max_chunk, int streams) {
    std::string why_not;
    if (!gpu_available(device, &why_not)) throw GpuError(why_not);
    if (max_chunk == 0 || max_chunk > kGpuMaxChunkSize) throw GpuError("unsupported chunk size for the GPU engine");
    if (streams < 1) throw GpuError("need at least one stream");
    return std::make_unique<CudaEngine>(device, max_chunk, streams);
}

}
