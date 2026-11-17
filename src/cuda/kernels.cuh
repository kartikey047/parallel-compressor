

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pfc::gpu {

constexpr int kThreads = 256;

std::size_t scan_scratch_bytes(std::uint32_t n);

cudaError_t exclusive_scan(void* scratch, std::size_t scratch_bytes, const std::uint32_t* in, std::uint32_t* out,
                           std::uint32_t n, cudaStream_t stream);
cudaError_t exclusive_scan(void* scratch, std::size_t scratch_bytes, const std::uint64_t* in, std::uint64_t* out,
                           std::uint32_t n, cudaStream_t stream);

cudaError_t inclusive_scan_mod256(void* scratch, std::size_t scratch_bytes, const std::uint8_t* in, std::uint8_t* out,
                                  std::uint32_t n, cudaStream_t stream);

cudaError_t delta_forward(const std::uint8_t* in, std::uint8_t* out, std::uint32_t n, cudaStream_t stream);
cudaError_t mark_run_starts(const std::uint8_t* in, std::uint32_t* flags, std::uint32_t n, cudaStream_t stream);
cudaError_t finalize_runs(const std::uint32_t* flags, const std::uint32_t* run_index, std::uint32_t n,
                          std::uint64_t* meta, cudaStream_t stream);
cudaError_t gather_run_starts(const std::uint32_t* flags, const std::uint32_t* run_index, std::uint32_t* starts,
                              std::uint32_t n, cudaStream_t stream);
cudaError_t count_tokens(const std::uint32_t* starts, const std::uint64_t* meta, std::uint32_t n,
                         std::uint32_t* tokens, cudaStream_t stream);
cudaError_t finalize_tokens(const std::uint32_t* tokens, const std::uint32_t* token_offset, std::uint32_t n,
                            std::uint64_t* meta, cudaStream_t stream);
cudaError_t emit_tokens(const std::uint8_t* in, const std::uint32_t* starts, const std::uint32_t* token_offset,
                        const std::uint64_t* meta, std::uint32_t n, std::uint8_t* out, cudaStream_t stream);

cudaError_t read_counts(const std::uint8_t* payload, std::uint32_t token_count, std::uint64_t* counts,
                        std::uint64_t* meta, cudaStream_t stream);
cudaError_t finalize_decode(const std::uint64_t* counts, const std::uint64_t* offsets, std::uint32_t token_count,
                            std::uint64_t* meta, cudaStream_t stream);
cudaError_t fill_output(const std::uint8_t* payload, const std::uint64_t* offsets, const std::uint64_t* meta,
                        std::uint32_t token_count, std::uint32_t raw_size, std::uint8_t* out, cudaStream_t stream);

}
