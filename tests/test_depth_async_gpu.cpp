#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/preprocess.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace miniinfer::depth;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: test_depth_async_gpu PACKAGE IMAGE_A IMAGE_B");
        const auto a = read_ppm(argv[2]), b = read_ppm(argv[3]);
        require(a.pixels() != b.pixels(), "mixed-image test requires different input pixels");
        std::vector<DepthMap> expected;
        SdkConfig config; config.package_dir = argv[1];
        {
            DepthSdk sdk(config);
            expected.push_back(sdk.run(a)); expected.push_back(sdk.run(b));
            require(sdk.snapshot().backend == "trt", "GPU test requires a TensorRT package");
        }
        const auto bytes = DepthSdk::reservation_bytes(a.width(), a.height())
                         + DepthSdk::reservation_bytes(b.width(), b.height());
        for (std::size_t slots : {1, 2}) for (bool host_reuse : {false, true}) for (bool gpu_reuse : {false, true}) {
            config.gpu_slots = slots;
            config.reuse_gpu_buffers = gpu_reuse;
            config.reuse_host_workspace = host_reuse;
            DepthRequest retained_a, retained_b;
            {
                DepthSdk sdk(config);
                std::size_t calls = 0;
                for (int round = 0; round < 3; ++round) {
                    // Swap submission order each round to exercise worker/slot reuse.
                    auto first = sdk.submit(round % 2 == 0 ? a : b);
                    auto second = sdk.submit(round % 2 == 0 ? b : a);
                    require(first.id() != second.id(), "request IDs collide");
                    first.wait(); second.wait(); calls += 2;
                    const auto state = sdk.snapshot();
                    require(state.ready == 2 && state.inflight == 2 && state.reserved_bytes == bytes,
                            "ready results lost their admission reservation");
                    require(state.device_allocation_calls == (gpu_reuse ? 2 * slots : 2 * calls)
                            && state.pinned_allocation_calls == state.device_allocation_calls,
                            "GPU IO allocation/reuse ledger mismatch");
                    require(state.host_workspace_allocation_calls == (host_reuse ? config.worker_count * CpuWorkspace::allocation_count : 0)
                            && (state.host_workspace_bytes > 0) == host_reuse,
                            "CPU workspaces must follow worker count independently of GPU slots");
                    DepthTiming timing;
                    if (round == 2) {
                        retained_a = std::move(first); retained_b = std::move(second);
                    } else {
                        const auto out_second = second.get(&timing), out_first = first.get();
                        require(out_first.values() == expected[round % 2].values()
                                && out_second.values() == expected[1 - round % 2].values(),
                                "mixed-image results changed across slot reuse");
                        require(timing.accepted && timing.h2d_bytes > 0
                                && timing.ready_at != std::chrono::steady_clock::time_point{},
                                "successful GPU result has no transfer/publication timing");
                    }
                }
                sdk.shutdown();
                const auto state = sdk.snapshot();
                require(state.shutdown_complete && state.ready == 2 && state.inflight == 2
                        && state.executing == 0 && state.queued == 0 && state.reserved_bytes == bytes,
                        "shutdown must drain execution independently of retained handles");
                require(state.io_device_bytes == 0 && state.io_pinned_bytes == 0 && state.host_workspace_bytes == 0,
                        "shutdown retained backend allocations");
                require(state.context_required_bytes > 0, "missing context memory requirement");
            }
            const auto out_b = retained_b.get(), out_a = retained_a.get();
            require(out_b.values() == expected[1].values() && out_a.values() == expected[0].values(),
                    "two-image full-map outputs did not outlive SDK shutdown/destruction");
        }
        std::cout << "Independent CPU workers/GPU slots, mixed-image reuse, allocation and owning handles passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
