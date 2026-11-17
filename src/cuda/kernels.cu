#include "kernels.cuh"

#include <algorithm>

#include <cub/device/device_scan.cuh>

namespace pfc::gpu {
namespace {

inline unsigned blocks_for(std::uint32_t n) { return (n + kThreads - 1) / kThreads; }

struct AddMod256 {
    __host__ __device__ std::uint8_t operator()(std::uint8_t a, std::uint8_t b) const {
        return static_cast<std::uint8_t>(a + b);
    }
};

__global__ void delta_forward_kernel(const std::uint8_t* in, std::uint8_t* out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i == 0 ? in[0] : static_cast<std::uint8_t>(in[i] - in[i - 1]);
}

__global__ void mark_run_starts_kernel(const std::uint8_t* in, std::uint32_t* flags, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || in[i] != in[i - 1]) ? 1u : 0u;
}

__global__ void finalize_runs_kernel(const std::uint32_t* flags, const std::uint32_t* run_index, std::uint32_t n,
                                     std::uint64_t* meta) {
    meta[0] = static_cast<std::uint64_t>(run_index[n - 1]) + flags[n - 1];
}

__global__ void gather_run_starts_kernel(const std::uint32_t* flags, const std::uint32_t* run_index,
                                         std::uint32_t* starts, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i]) starts[run_index[i]] = i;
}

__global__ void count_tokens_kernel(const std::uint32_t* starts, const std::uint64_t* meta, std::uint32_t n,
                                    std::uint32_t* tokens) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const std::uint32_t runs = static_cast<std::uint32_t>(meta[0]);
    if (r >= runs) {
        tokens[r] = 0;
        return;
    }
    const std::uint32_t end = (r + 1 < runs) ? starts[r + 1] : n;
    tokens[r] = (end - starts[r] + 254) / 255;
}

__global__ void finalize_tokens_kernel(const std::uint32_t* tokens, const std::uint32_t* token_offset, std::uint32_t n,
                                       std::uint64_t* meta) {
    meta[1] = static_cast<std::uint64_t>(token_offset[n - 1]) + tokens[n - 1];
}

__global__ void emit_tokens_kernel(const std::uint8_t* in, const std::uint32_t* starts,
                                   const std::uint32_t* token_offset, const std::uint64_t* meta, std::uint32_t n,
                                   std::uint8_t* out) {
    const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t runs = static_cast<std::uint32_t>(meta[0]);
    const std::uint32_t total_tokens = static_cast<std::uint32_t>(meta[1]);
    if (t >= total_tokens) return;

    std::uint32_t lo = 0, hi = runs - 1;
    while (lo < hi) {
        const std::uint32_t mid = (lo + hi + 1) / 2;
        if (token_offset[mid] <= t) lo = mid; else hi = mid - 1;
    }
    const std::uint32_t r = lo;
    const std::uint32_t start = starts[r];
    const std::uint32_t end = (r + 1 < runs) ? starts[r + 1] : n;
    const std::uint32_t k = t - token_offset[r];
    const std::uint32_t remaining = (end - start) - 255 * k;
    out[2ull * t] = static_cast<std::uint8_t>(remaining < 255 ? remaining : 255);
    out[2ull * t + 1] = in[start];
}

__global__ void read_counts_kernel(const std::uint8_t* payload, std::uint32_t token_count, std::uint64_t* counts,
                                   std::uint64_t* meta) {
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= token_count) return;
    const std::uint8_t count = payload[2ull * j];
    counts[j] = count;
    if (count == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&meta[1]), 1ull);
}

__global__ void finalize_decode_kernel(const std::uint64_t* counts, const std::uint64_t* offsets,
                                       std::uint32_t token_count, std::uint64_t* meta) {
    meta[0] = offsets[token_count - 1] + counts[token_count - 1];
}

__global__ void fill_output_kernel(const std::uint8_t* payload, const std::uint64_t* offsets, const std::uint64_t* meta,
                                   std::uint32_t token_count, std::uint32_t raw_size, std::uint8_t* out) {
    const std::uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= raw_size) return;
    if (p >= meta[0]) {
        out[p] = 0;
        return;
    }

    std::uint32_t lo = 0, hi = token_count - 1;
    while (lo < hi) {
        const std::uint32_t mid = (lo + hi + 1) / 2;
        if (offsets[mid] <= p) lo = mid; else hi = mid - 1;
    }
    out[p] = payload[2ull * lo + 1];
}

}

std::size_t scan_scratch_bytes(std::uint32_t n) {
    std::size_t a = 0, b = 0, c = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, a, static_cast<const std::uint32_t*>(nullptr), static_cast<std::uint32_t*>(nullptr), static_cast<int>(n));
    cub::DeviceScan::ExclusiveSum(nullptr, b, static_cast<const std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr), static_cast<int>(n));
    cub::DeviceScan::InclusiveScan(nullptr, c, static_cast<const std::uint8_t*>(nullptr), static_cast<std::uint8_t*>(nullptr), AddMod256{}, static_cast<int>(n));
    return std::max({a, b, c});
}

cudaError_t exclusive_scan(void* scratch, std::size_t bytes, const std::uint32_t* in, std::uint32_t* out,
                           std::uint32_t n, cudaStream_t stream) {
    return cub::DeviceScan::ExclusiveSum(scratch, bytes, in, out, static_cast<int>(n), stream);
}

cudaError_t exclusive_scan(void* scratch, std::size_t bytes, const std::uint64_t* in, std::uint64_t* out,
                           std::uint32_t n, cudaStream_t stream) {
    return cub::DeviceScan::ExclusiveSum(scratch, bytes, in, out, static_cast<int>(n), stream);
}

cudaError_t inclusive_scan_mod256(void* scratch, std::size_t bytes, const std::uint8_t* in, std::uint8_t* out,
                                  std::uint32_t n, cudaStream_t stream) {
    return cub::DeviceScan::InclusiveScan(scratch, bytes, in, out, AddMod256{}, static_cast<int>(n), stream);
}

cudaError_t delta_forward(const std::uint8_t* in, std::uint8_t* out, std::uint32_t n, cudaStream_t stream) {
    delta_forward_kernel<<<blocks_for(n), kThreads, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

cudaError_t mark_run_starts(const std::uint8_t* in, std::uint32_t* flags, std::uint32_t n, cudaStream_t stream) {
    mark_run_starts_kernel<<<blocks_for(n), kThreads, 0, stream>>>(in, flags, n);
    return cudaGetLastError();
}

cudaError_t finalize_runs(const std::uint32_t* flags, const std::uint32_t* run_index, std::uint32_t n,
                          std::uint64_t* meta, cudaStream_t stream) {
    finalize_runs_kernel<<<1, 1, 0, stream>>>(flags, run_index, n, meta);
    return cudaGetLastError();
}

cudaError_t gather_run_starts(const std::uint32_t* flags, const std::uint32_t* run_index, std::uint32_t* starts,
                              std::uint32_t n, cudaStream_t stream) {
    gather_run_starts_kernel<<<blocks_for(n), kThreads, 0, stream>>>(flags, run_index, starts, n);
    return cudaGetLastError();
}

cudaError_t count_tokens(const std::uint32_t* starts, const std::uint64_t* meta, std::uint32_t n,
                         std::uint32_t* tokens, cudaStream_t stream) {
    count_tokens_kernel<<<blocks_for(n), kThreads, 0, stream>>>(starts, meta, n, tokens);
    return cudaGetLastError();
}

cudaError_t finalize_tokens(const std::uint32_t* tokens, const std::uint32_t* token_offset, std::uint32_t n,
                            std::uint64_t* meta, cudaStream_t stream) {
    finalize_tokens_kernel<<<1, 1, 0, stream>>>(tokens, token_offset, n, meta);
    return cudaGetLastError();
}

cudaError_t emit_tokens(const std::uint8_t* in, const std::uint32_t* starts, const std::uint32_t* token_offset,
                        const std::uint64_t* meta, std::uint32_t n, std::uint8_t* out, cudaStream_t stream) {

    emit_tokens_kernel<<<blocks_for(n), kThreads, 0, stream>>>(in, starts, token_offset, meta, n, out);
    return cudaGetLastError();
}

cudaError_t read_counts(const std::uint8_t* payload, std::uint32_t token_count, std::uint64_t* counts,
                        std::uint64_t* meta, cudaStream_t stream) {
    read_counts_kernel<<<blocks_for(token_count), kThreads, 0, stream>>>(payload, token_count, counts, meta);
    return cudaGetLastError();
}

cudaError_t finalize_decode(const std::uint64_t* counts, const std::uint64_t* offsets, std::uint32_t token_count,
                            std::uint64_t* meta, cudaStream_t stream) {
    finalize_decode_kernel<<<1, 1, 0, stream>>>(counts, offsets, token_count, meta);
    return cudaGetLastError();
}

cudaError_t fill_output(const std::uint8_t* payload, const std::uint64_t* offsets, const std::uint64_t* meta,
                        std::uint32_t token_count, std::uint32_t raw_size, std::uint8_t* out, cudaStream_t stream) {
    fill_output_kernel<<<blocks_for(raw_size), kThreads, 0, stream>>>(payload, offsets, meta, token_count, raw_size, out);
    return cudaGetLastError();
}

}
