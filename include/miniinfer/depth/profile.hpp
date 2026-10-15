#pragma once

// NVTX is optional and compiled out of ordinary/CPU-only builds. It records
// host scopes for a separate profiler run, never device-completion timestamps.
#if defined(MINIINFER_DEPTH_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif
namespace miniinfer::depth {
class ProfileRange final {
public:
    explicit ProfileRange(const char* name) noexcept {
#if defined(MINIINFER_DEPTH_NVTX)
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~ProfileRange() {
#if defined(MINIINFER_DEPTH_NVTX)
        nvtxRangePop();
#endif
    }
    ProfileRange(const ProfileRange&) = delete;
    ProfileRange& operator=(const ProfileRange&) = delete;
};
} // namespace miniinfer::depth
