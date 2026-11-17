# Parallel File Compressor

A simple C++ program that compresses and uncompresses files into `.pfc` archive files. This program uses run-length encoding (RLE) for lossless compression. We accelerate file compression by splitting files into pieces and performing RLE on each piece independently on separate GPU or CPU threads (dependening on the selected device). This yields a 95% speedup in 4-threaded GPU compression over 1-threaded compression and a 250% speedup in 4-threaded CPU compression over 1-threaded compression on a sample set of 10GB files (GPU metrics are computed from results on RTX 4080 Laptop).

## Build

Requires CMake, a C++20 compiler, and (optionally) a CUDA toolkit.

Build the CPU version:

```powershell
cmake -S . -B out/cpu -DPFC_ENABLE_CUDA=OFF
cmake --build out/cpu
```

Build with CUDA:

```powershell
cmake -S . -B out/cuda -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build out/cuda
```

On Windows, run the CUDA commands from an x64 Visual Studio developer prompt. To use CUDA 12.6 specifically, add its compiler path to the first command:

```powershell
cmake -S . -B out/cuda -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=89
```

## Usage

The executable is `compressor` in the build folder.

Compress a file:

```powershell
.\out\cpu\compressor.exe compress input.bin -o input.pfc
```

Restore a file:

```powershell
.\out\cpu\compressor.exe decompress input.pfc -o restored.bin
```

Check an archive without creating an output file:

```powershell
.\out\cpu\compressor.exe verify input.pfc
```

Show archive details:

```powershell
.\out\cpu\compressor.exe info input.pfc
```

List CUDA devices:

```powershell
.\out\cuda\compressor.exe devices
```

Useful options:

| Option | Purpose |
|---|---|
| `-o FILE` | Choose the output file. |
| `--backend cpu` | Use the CPU. |
| `--backend cuda` | Use CUDA. |
| `--backend auto` | Choose CUDA when available, otherwise use the CPU. |
| `--codec raw\|rle\|delta\|auto` | Choose how each file piece is stored. |
| `--chunk-size SIZE` | Set the size of each file piece. `K`, `M`, and `G` suffixes are accepted. |
| `-f` | Replace an existing output file. |

Run `compressor --help` to see the full command reference.
