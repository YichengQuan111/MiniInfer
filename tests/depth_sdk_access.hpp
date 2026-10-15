#pragma once
#include "miniinfer/depth/sdk.hpp"
#include <utility>

namespace miniinfer::depth::detail {
// Test-only friend definition. Production callers initialize through a package.
struct SdkTestAccess {
    static std::unique_ptr<DepthSdk> create(SdkConfig config,
        std::function<DepthMap(const RgbImage&)> backend, std::function<void()> before_delivery = {},
        std::function<bool(const std::exception_ptr&)> fatal_error = {}) {
        return std::unique_ptr<DepthSdk>(new DepthSdk(std::move(config), std::move(backend), std::move(before_delivery), std::move(fatal_error)));
    }
    static void wait(DepthSdk& sdk, const std::function<bool(const SdkSnapshot&)>& condition) {
        sdk.wait_for_test(condition, std::chrono::seconds(5));
    }
    static std::unique_ptr<DepthSdk> create_factory(SdkConfig config,
        std::function<std::function<DepthMap(const RgbImage&)>(std::size_t)> factory,
        bool worker_local = true, std::function<void()> before_delivery = {}) {
        return std::unique_ptr<DepthSdk>(new DepthSdk(std::move(config), std::move(factory),
                                                    worker_local, std::move(before_delivery)));
    }
};
}  // namespace miniinfer::depth::detail
