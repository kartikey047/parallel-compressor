param([string]$Cuda = "12.6", [string]$Arch = "89")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$cudaHome = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v$Cuda"
$commands = @(
  "`"$vcvars`"",
  "set `"CUDA_PATH=$cudaHome`"",
  "set `"PATH=$cudaHome\bin;%PATH%`"",
  "cmake -S . -B out/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$Arch -DCMAKE_CUDA_COMPILER=`"$cudaHome\bin\nvcc.exe`" -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
  "cmake --build out/cuda"
) -join " && "
cmd /c $commands
