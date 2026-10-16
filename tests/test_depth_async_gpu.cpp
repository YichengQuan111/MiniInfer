#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/preprocess.hpp"
#include "miniinfer/depth/artifact.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace miniinfer::depth;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) throw std::invalid_argument("usage: test_depth_async_gpu PACKAGE IMAGE_A IMAGE_B [MODE]");
        const auto a = read_ppm(argv[2]), b = read_ppm(argv[3]);
        require(a.pixels() != b.pixels(), "mixed-image test requires different input pixels");
        std::vector<DepthMap> expected;
        SdkConfig config; config.package_dir = argv[1];
        config.preprocess_mode = argc == 5 ? parse_preprocess_mode(argv[4]) : PreprocessMode::Cpu;
        std::vector<std::size_t> expected_h2d;
        {
            DepthSdk sdk(config);
            DepthTiming timing;
            expected.push_back(sdk.run(a, &timing)); expected_h2d.push_back(timing.h2d_bytes);
            expected.push_back(sdk.run(b, &timing)); expected_h2d.push_back(timing.h2d_bytes);
            require(sdk.snapshot().backend == "trt", "GPU test requires a TensorRT package");
        }
        const bool full = config.preprocess_mode == PreprocessMode::CudaFullBasic || config.preprocess_mode == PreprocessMode::CudaFullFused;
        const std::size_t device_per_slot = config.preprocess_mode == PreprocessMode::Cpu ? 2 : 3;
        const std::size_t pinned_per_slot = full ? 3 : 2;
        const auto size = contract_model_size(require_field(read_manifest(config.package_dir / "manifest.txt"), "contract"));
        const auto base_io_bytes = 4 * size * size * sizeof(float);
        const auto device_capacity = base_io_bytes + (full ? 3 * max_pixels :
            (config.preprocess_mode == PreprocessMode::CudaPartial ? 3 * size * size * sizeof(float) : 0));
        const auto pinned_capacity = base_io_bytes + (full ? 3 * max_pixels : 0);
        require(!full || (expected_h2d[0] == a.pixels().size() && expected_h2d[1] == b.pixels().size()),
                "full CUDA paths must upload actual RGB8 bytes");
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
                    require(state.device_allocation_calls == device_per_slot * (gpu_reuse ? slots : calls)
                            && state.pinned_allocation_calls == pinned_per_slot * (gpu_reuse ? slots : calls),
                            "GPU IO allocation/reuse ledger mismatch");
                    require(state.io_device_bytes == (gpu_reuse ? slots * device_capacity : 0)
                            && state.io_pinned_bytes == (gpu_reuse ? slots * pinned_capacity : 0),
                            "GPU scratch exceeds or disagrees with fixed capacity ledger");
                    require(state.host_workspace_allocation_calls == (host_reuse ? config.worker_count * CpuWorkspace::allocation_count : 0)
                            && (state.host_workspace_bytes > 0) == host_reuse,
                            "CPU workspaces must follow worker count independently of GPU slots");
                    require(state.engine_serialized_bytes > 0 && state.context_required_bytes > 0,
                            "engine plan/context resource indicators missing");
                    DepthTiming timing;
                    if (round == 2) {
                        retained_a = std::move(first); retained_b = std::move(second);
                    } else {
                        const auto out_second = second.get(&timing), out_first = first.get();
                        require(out_first.values() == expected[round % 2].values()
                                && out_second.values() == expected[1 - round % 2].values(),
                                "mixed-image results changed across slot reuse");
                        require(timing.accepted && timing.h2d_bytes == expected_h2d[1 - round % 2]
                                && timing.model_ms > 0 && timing.slot_wait_ms >= 0
                                && timing.pinned_copy_ms >= 0 && timing.synchronize_ms >= 0
                                && (config.preprocess_mode == PreprocessMode::Cpu || timing.gpu_preprocess_ms > 0)
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
                require(state.context_required_bytes > 0 && state.engine_serialized_bytes > 0, "missing historical context/engine requirement");
            }
            const auto out_b = retained_b.get(), out_a = retained_a.get();
            require(out_b.values() == expected[1].values() && out_a.values() == expected[0].values(),
                    "two-image full-map outputs did not outlive SDK shutdown/destruction");
        }
        std::cout << "Independent CPU workers/GPU slots, mixed-image reuse, allocation and owning handles passed: " << preprocess_mode_name(config.preprocess_mode) << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
