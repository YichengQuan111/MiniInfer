#pragma once

#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/thread_pool.hpp"
#include "miniinfer/workspace.hpp"
#include <cstddef>
#include <future>
#include <vector>

namespace miniinfer {

struct RuntimeConfig {
    std::size_t workers = 1;
    std::size_t queue_capacity = 64;  // Waiting tasks only.
    std::size_t max_batch = 256;
};

// Owns model, worker scratch storage and the pool, in lifetime dependency order.
// External run/submit/shutdown calls may overlap; the caller owns object lifetime.
class Runtime final {
public:
    explicit Runtime(MlpModel model, RuntimeConfig config = {});
    ~Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    std::future<MlpResult> submit(MlpInput input, std::shared_ptr<TaskTiming> timing = {});
    MlpResult run(MlpInput input);
    void shutdown();
    const RuntimeConfig& config() const noexcept { return config_; }

private:
    MlpModel model_;
    RuntimeConfig config_;
    std::vector<Workspace> workspaces_;
    // Destroyed first: drains/joins before model and Workspace destruction.
    ThreadPool pool_;
};

}  // namespace miniinfer
