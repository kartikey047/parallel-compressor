# PFC archive format, version 2

A PFC archive stores one file as a sequence of independent chunks. This document is the specification; the
reference implementation is `src/format.cpp`.

All integers are **little endian**. CRC-32 is the IEEE 802.3 CRC (polynomial `0xEDB88320`, initial value and
final XOR `0xFFFFFFFF`), the same as zlib and PNG; the CRC-32 of the ASCII string `123456789` is `0xCBF43926`.

## Layout

```
offset 0        header                  40 bytes
offset 40       chunk payloads          back to back, in chunk order
index_offset    chunk index             24 bytes per chunk, to the end of the file
```

The index is at the end so a writer can stream chunks out in one pass and write the header last.

## Header (40 bytes)

| offset | size | field | meaning |
|---:|---:|---|---|
| 0 | 4 | magic | the bytes `P F C A` |
| 4 | 2 | version | `2` |
| 6 | 2 | header_size | `40` |
| 8 | 4 | chunk_size | uncompressed bytes per chunk (every chunk except the last), `1 .. 2^30` |
| 12 | 4 | chunk_count | number of chunks; `ceil(original_size / chunk_size)`, so `0` for an empty file |
| 16 | 8 | original_size | size of the uncompressed file in bytes |
| 24 | 8 | index_offset | file offset of the chunk index |
| 32 | 4 | index_crc | CRC-32 of the index bytes (the CRC of zero bytes, `0`, when there are no chunks) |
| 36 | 4 | header_crc | CRC-32 of header bytes 0 to 35 |

## Chunk index entry (24 bytes)

| offset | size | field | meaning |
|---:|---:|---|---|
| 0 | 1 | codec | `0` raw, `1` RLE, `2` delta+RLE |
| 1 | 3 | reserved | must be zero |
| 4 | 4 | raw_size | uncompressed size of this chunk |
| 8 | 8 | offset | file offset of the payload |
| 16 | 4 | payload_size | stored size of the payload |
| 20 | 4 | crc | CRC-32 of the chunk's **uncompressed** bytes |

## Codecs

**Raw (0).** The payload is the chunk itself: `payload_size == raw_size`. It is the escape for data that does
not compress, so an archive never grows by more than its header and index
(`40 + 24 * chunk_count` bytes).

**RLE (1).** The payload is a list of tokens, each two bytes: `count` (1 to 255) then `value`. A token stands
for `count` copies of `value`. The encoding is *canonical*, so every conforming encoder produces the same
bytes: a run of `L` equal bytes is written as `floor(L / 255)` tokens with count 255 followed by one token
for the remainder (if it is not zero), and runs are maximal (two neighbouring tokens with the same value
never come from one run unless the run was longer than 255).

**Delta + RLE (2).** The chunk is first transformed with a byte-wise delta, `d[0] = x[0]` and
`d[i] = (x[i] - x[i-1]) mod 256`, and `d` is stored as an RLE payload (as above). Decoding is RLE decoding
followed by the running sum `x[i] = (x[i-1] + d[i]) mod 256`. It suits ramps, counters and slowly varying
samples; for those, differences repeat even though the values do not.

An encoder picks, per chunk, the smallest of raw, RLE and delta+RLE (raw wins ties and whenever the other
codecs are not strictly smaller).

## What a reader must check

A reader rejects the archive (without crashing and without allocating memory from unchecked size fields) if:

1. the file is shorter than 40 bytes, or the magic, version or `header_size` differ, or `header_crc` does not match;
2. `chunk_size` is 0 or above `2^30`, or `chunk_count != ceil(original_size / chunk_size)`;
3. `index_offset + 24 * chunk_count` is not exactly the file size (this catches truncation and trailing data,
   and is done **before** the index is read);
4. `index_crc` does not match the index bytes;
5. for any entry: the codec is unknown, reserved bytes are not zero, `raw_size` is not `chunk_size` (or the
   remainder for the last chunk), `offset` is not the end of the previous payload (the first starts at 40),
   or the payload size is impossible for the codec:
   - raw: `payload_size != raw_size`;
   - RLE and delta+RLE: `payload_size` is 0 or odd, is more than `2 * raw_size`, or is too small to hold
     `raw_size` bytes (one token covers at most 255 bytes, so `raw_size <= 255 * payload_size / 2`);
6. the last payload does not end exactly at `index_offset`;
7. while decoding: an RLE stream has a zero count, expands to more or fewer than `raw_size` bytes, or the CRC
   of the decoded chunk does not match `crc`.

Because the check in step 5 bounds how much a small payload can expand, a few bytes of archive cannot make a
reader allocate gigabytes.

## Compatibility

* Chunks are independent, so they can be compressed and decompressed in parallel, and by different backends
  (CPU or GPU) in the same archive.
* The version number changes whenever the layout does. Version 1 was an early prototype (magic `PFC1`) and is
  not read by this version; it is reported as "not a PFC archive".
* Archives are deterministic: the same input, chunk size and codec choice give the same bytes, whatever the
  thread count.

## Limits

Chunk size up to 1 GiB (the GPU backend: up to 32 MiB), at most 2^32 - 1 chunks, so files of many terabytes.
