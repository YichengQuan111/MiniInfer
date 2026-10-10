#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/data_io.hpp"
#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/runtime.hpp"
#include "miniinfer/thread_pool.hpp"
#include "miniinfer/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using miniinfer::Buffer;
using miniinfer::CpuExecutor;
using miniinfer::MlpInput;
using miniinfer::MlpModel;
using miniinfer::MlpResult;
using miniinfer::PoolClosed;
using miniinfer::Runtime;
using miniinfer::RuntimeConfig;
using miniinfer::Workspace;

static_assert(!std::is_copy_constructible_v<Runtime>, "Runtime must not be copy constructible");
static_assert(!std::is_copy_assignable_v<Runtime>, "Runtime must not be copy assignable");
static_assert(!std::is_move_constructible_v<Runtime>, "Runtime must not be move constructible");
static_assert(!std::is_move_assignable_v<Runtime>, "Runtime must not be move assignable");

constexpr std::size_t kReferenceSamples = 64;
constexpr std::size_t kInputFeatures = 64;
constexpr std::size_t kOutputFeatures = 10;
constexpr float kAtol = 1.0e-6F;
constexpr float kRtol = 1.0e-6F;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable, std::string_view message) {
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(message) + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(std::string(message) + ": expected exception was not thrown");
}

template <typename Exception, typename T>
void expect_future_throws(std::future<T>& future, std::string_view message) {
    try {
        (void)future.get();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(message) + ": wrong future exception: " +
                                 error.what());
    }
    throw std::runtime_error(std::string(message) + ": future completed without an exception");
}

struct ReferenceInputs {
    Buffer values;
};

ReferenceInputs load_reference_inputs(const fs::path& directory) {
    return {miniinfer::load_f32_file(directory / "inputs.f32",
                                     kReferenceSamples * kInputFeatures)};
}

std::vector<float> make_input_values(const ReferenceInputs& reference, std::size_t first_sample,
                                     std::size_t batch) {
    if (first_sample > kReferenceSamples || batch > kReferenceSamples - first_sample) {
        throw std::invalid_argument("test input range exceeds the 64 official reference rows");
    }
    const std::size_t count = batch * kInputFeatures;
    const float* const first = reference.values.data() + first_sample * kInputFeatures;
    return std::vector<float>(first, first + count);
}

struct OutputSnapshot {
    std::size_t batch = 0;
    std::vector<float> logits;
    std::vector<std::int32_t> labels;
};

OutputSnapshot snapshot(const MlpResult& result) {
    const auto logits = result.logits();
    return {result.batch(), std::vector<float>(logits.data(), logits.data() + logits.size()),
            result.labels()};
}

OutputSnapshot run_direct(const MlpModel& model, const MlpInput& input) {
    Workspace workspace(256);
    CpuExecutor executor;
    const MlpResult result = executor.run(model, input, workspace);
    return snapshot(result);
}

void compare_result(const MlpResult& actual, const OutputSnapshot& expected,
                    std::size_t first_sample, std::string_view context) {
    expect(actual.batch() == expected.batch, "runtime and direct result batch sizes must match");
    const auto logits = actual.logits();
    expect(logits.rows() == expected.batch && logits.cols() == kOutputFeatures,
           "runtime result logits must have shape [batch,10]");
    expect(actual.labels().size() == expected.batch,
           "runtime result label count must match its batch");
    expect(expected.logits.size() == expected.batch * kOutputFeatures,
           "direct result snapshot logits must have shape [batch,10]");
    expect(expected.labels.size() == expected.batch,
           "direct result snapshot label count must match its batch");

    for (std::size_t row = 0; row < expected.batch; ++row) {
        for (std::size_t column = 0; column < kOutputFeatures; ++column) {
            const float value = logits.at(row, column);
            const float baseline = expected.logits[row * kOutputFeatures + column];
            const float error = std::abs(value - baseline);
            const float tolerance = kAtol + kRtol * std::abs(baseline);
            if (!std::isfinite(value) || error > tolerance) {
                std::ostringstream message;
                message << context << ", sample=" << (first_sample + row)
                        << ", column=" << column << ", actual=" << value
                        << ", direct=" << baseline << ", abs_error=" << error
                        << ", tolerance=" << tolerance;
                throw std::runtime_error(message.str());
            }
        }
        if (actual.labels()[row] != expected.labels[row]) {
            std::ostringstream message;
            message << context << ", sample=" << (first_sample + row)
                    << ", predicted label differs: runtime=" << actual.labels()[row]
                    << ", direct=" << expected.labels[row];
            throw std::runtime_error(message.str());
        }
    }
}

void expect_snapshot_unchanged(const MlpResult& result, const OutputSnapshot& before,
                               std::size_t first_sample, std::string_view context) {
    compare_result(result, before, first_sample, context);
}

void test_async_order_ownership_and_result_lifetime(const fs::path& directory,
                                                     const ReferenceInputs& reference) {
    const MlpModel direct_model = MlpModel::load(directory);
    RuntimeConfig config;
    config.workers = 2;
    config.queue_capacity = 8;
    config.max_batch = 256;
    Runtime runtime(MlpModel::load(directory), config);
    expect(runtime.config().workers == config.workers &&
               runtime.config().queue_capacity == config.queue_capacity &&
               runtime.config().max_batch == config.max_batch,
           "runtime must retain its configuration");

    std::vector<float> caller_values = make_input_values(reference, 4, 1);
    MlpInput single_input(1, caller_values);
    std::fill(caller_values.begin(), caller_values.end(), -17.0F);
    const MlpInput single_expected_input(1, make_input_values(reference, 4, 1));
    const OutputSnapshot single_expected = run_direct(direct_model, single_expected_input);
    std::future<MlpResult> single_future = runtime.submit(single_input);
    single_input = MlpInput(1, make_input_values(reference, 5, 1));

    MlpInput batch_input(8, make_input_values(reference, 12, 8));
    const MlpInput batch_expected_input(8, make_input_values(reference, 12, 8));
    const OutputSnapshot batch_expected = run_direct(direct_model, batch_expected_input);
    std::future<MlpResult> batch_future = runtime.submit(batch_input);
    batch_input = MlpInput(1, make_input_values(reference, 31, 1));

    MlpInput moved_input(1, make_input_values(reference, 40, 1));
    const MlpInput moved_expected_input(1, make_input_values(reference, 40, 1));
    const OutputSnapshot moved_expected = run_direct(direct_model, moved_expected_input);
    std::future<MlpResult> moved_future = runtime.submit(std::move(moved_input));
    moved_input = MlpInput(1, make_input_values(reference, 41, 1));

    MlpResult single_result = single_future.get();
    MlpResult batch_result = batch_future.get();
    MlpResult moved_result = moved_future.get();
    compare_result(single_result, single_expected, 4, "lvalue submit");
    compare_result(batch_result, batch_expected, 12, "ordered batch submit");
    compare_result(moved_result, moved_expected, 40, "rvalue submit");

    const OutputSnapshot single_before = snapshot(single_result);
    const OutputSnapshot batch_before = snapshot(batch_result);
    const OutputSnapshot moved_before = snapshot(moved_result);
    expect(single_result.logits().data() != batch_result.logits().data() &&
               single_result.logits().data() != moved_result.logits().data() &&
               batch_result.logits().data() != moved_result.logits().data(),
           "separate runtime results must own separate logits storage");

    const MlpInput later_input(8, make_input_values(reference, 48, 8));
    const OutputSnapshot later_expected = run_direct(direct_model, later_input);
    const MlpResult later_result = runtime.run(MlpInput(8, make_input_values(reference, 48, 8)));
    compare_result(later_result, later_expected, 48, "later synchronous run");
    expect_snapshot_unchanged(single_result, single_before, 4, "retained batch-one result");
    expect_snapshot_unchanged(batch_result, batch_before, 12, "retained batch-eight result");
    expect_snapshot_unchanged(moved_result, moved_before, 40, "retained moved-input result");
}

void test_validation_shutdown_and_worker_recovery(const fs::path& directory,
                                                   const ReferenceInputs& reference) {
    const MlpModel direct_model = MlpModel::load(directory);
    RuntimeConfig config;
    config.workers = 2;
    config.queue_capacity = 8;
    config.max_batch = 8;
    Runtime runtime(MlpModel::load(directory), config);

    const MlpInput oversized(9, std::vector<float>(9 * kInputFeatures, 0.0F));
    expect_throws<std::invalid_argument>([&] { (void)runtime.submit(oversized); },
                                         "runtime must reject input above max_batch");

    std::vector<float> overflowing_values(kInputFeatures,
                                          std::numeric_limits<float>::max());
    const MlpInput overflowing_input(1, overflowing_values);
    expect_throws<std::overflow_error>(
        [&] { (void)run_direct(direct_model, overflowing_input); },
        "finite maximum inputs must overflow the direct FP32 executor for this fixture");
    std::future<MlpResult> failed_future = runtime.submit(overflowing_input);
    expect_future_throws<std::overflow_error>(failed_future,
                                              "executor failure must be delivered by its future");

    const MlpInput recovery_input(1, make_input_values(reference, 22, 1));
    const OutputSnapshot recovery_expected = run_direct(direct_model, recovery_input);
    std::future<MlpResult> recovery_future = runtime.submit(recovery_input);
    MlpResult recovery_result = recovery_future.get();
    compare_result(recovery_result, recovery_expected, 22,
                   "worker recovery after a failed task");

    runtime.shutdown();
    runtime.shutdown();
    const MlpInput after_close(1, make_input_values(reference, 23, 1));
    expect_throws<PoolClosed>([&] { (void)runtime.submit(after_close); },
                              "submit after shutdown must report PoolClosed");
    expect_throws<PoolClosed>([&] { (void)runtime.run(after_close); },
                              "run after shutdown must report PoolClosed");
}

void test_destructor_drains_accepted_future(const fs::path& directory,
                                           const ReferenceInputs& reference) {
    const MlpModel direct_model = MlpModel::load(directory);
    const MlpInput input(8, make_input_values(reference, 28, 8));
    const OutputSnapshot expected = run_direct(direct_model, input);
    std::future<MlpResult> pending;
    {
        RuntimeConfig config;
        config.workers = 2;
        config.queue_capacity = 4;
        config.max_batch = 8;
        Runtime runtime(MlpModel::load(directory), config);
        pending = runtime.submit(input);
    }  // Destruction must drain accepted work and join workers.
    MlpResult result = pending.get();
    compare_result(result, expected, 28, "future received after runtime destruction");
}

void test_invalid_runtime_configurations(const fs::path& directory) {
    RuntimeConfig invalid_workers;
    invalid_workers.workers = 0;
    expect_throws<std::invalid_argument>(
        [&] { Runtime runtime(MlpModel::load(directory), invalid_workers); },
        "zero runtime workers must be rejected");

    RuntimeConfig invalid_capacity;
    invalid_capacity.queue_capacity = 0;
    expect_throws<std::invalid_argument>(
        [&] { Runtime runtime(MlpModel::load(directory), invalid_capacity); },
        "zero runtime queue capacity must be rejected");

    RuntimeConfig invalid_max_batch;
    invalid_max_batch.max_batch = 0;
    expect_throws<std::invalid_argument>(
        [&] { Runtime runtime(MlpModel::load(directory), invalid_max_batch); },
        "zero runtime max_batch must be rejected");

    RuntimeConfig overflowing;
    overflowing.workers = 1;
    overflowing.queue_capacity = 1;
    overflowing.max_batch = std::numeric_limits<std::size_t>::max();
    expect_throws<std::overflow_error>(
        [&] { Runtime runtime(MlpModel::load(directory), overflowing); },
        "overflowing runtime workspace capacity must be rejected before allocation");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "test_runtime: expected exactly one reference data directory argument\n";
        return 2;
    }

    try {
        const fs::path reference_directory(argv[1]);
        const ReferenceInputs reference = load_reference_inputs(reference_directory);
        test_async_order_ownership_and_result_lifetime(reference_directory, reference);
        test_validation_shutdown_and_worker_recovery(reference_directory, reference);
        test_destructor_drains_accepted_future(reference_directory, reference);
        test_invalid_runtime_configurations(reference_directory);
        std::cout << "test_runtime: 4 groups passed; workers=2; batch sizes=1,8\n";
    } catch (const std::exception& error) {
        std::cerr << "test_runtime: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "test_runtime: unknown failure\n";
        return 1;
    }
    return 0;
}
