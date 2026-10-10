#include "miniinfer/runtime.hpp"
#include "miniinfer/checked_size.hpp"
#include "miniinfer/cpu_executor.hpp"
#include <stdexcept>
#include <utility>

namespace miniinfer {
namespace {

std::vector<Workspace> prepare_workspaces(const RuntimeConfig& config) {
    if (config.workers == 0 || config.queue_capacity == 0 || config.max_batch == 0) {
        throw std::invalid_argument("Runtime workers, queue_capacity and max_batch must be positive");
    }
    const auto elements = checked_multiply(config.max_batch, MlpModel::hidden_features);
    (void)checked_multiply(checked_multiply(config.workers, elements), sizeof(float));
    std::vector<Workspace> workspaces;
    workspaces.reserve(config.workers);
    for (std::size_t worker = 0; worker < config.workers; ++worker) {
        workspaces.emplace_back(config.max_batch);
    }
    return workspaces;
}

}  // namespace

Runtime::Runtime(MlpModel model, RuntimeConfig config)
    : model_(std::move(model)), config_(config), workspaces_(prepare_workspaces(config)),
      pool_(config.workers, config.queue_capacity) {}

std::future<MlpResult> Runtime::submit(MlpInput input, std::shared_ptr<TaskTiming> timing) {
    input.validate(config_.max_batch);
    return pool_.submit([this, input = std::move(input)](std::size_t worker_id) {
        return CpuExecutor{}.run(model_, input, workspaces_.at(worker_id));
    }, std::move(timing));
}

MlpResult Runtime::run(MlpInput input) {
    return submit(std::move(input)).get();
}

void Runtime::shutdown() {
    pool_.shutdown();
}

}  // namespace miniinfer
