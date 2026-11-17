
#include "pfc/gpu.hpp"

namespace pfc {

bool gpu_compiled() { return false; }

bool gpu_available(int, std::string* why_not) {
    if (why_not) *why_not = "this build has no CUDA support";
    return false;
}

std::unique_ptr<GpuEngine> create_gpu_engine(int, std::size_t, int) {
    throw GpuError("this build has no CUDA support");
}

}
