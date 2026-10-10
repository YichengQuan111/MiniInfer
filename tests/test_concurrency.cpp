#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/data_io.hpp"
#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/runtime.hpp"
#include "miniinfer/thread_pool.hpp"
#include "miniinfer/workspace.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using miniinfer::Buffer;
using miniinfer::CpuExecutor;
using miniinfer::MlpInput;
using miniinfer::MlpModel;
using miniinfer::MlpResult;
using miniinfer::QueueFull;
using miniinfer::Runtime;
using miniinfer::RuntimeConfig;
using miniinfer::Workspace;

constexpr std::size_t kReferenceSamples = 64;
constexpr std::size_t kInputFeatures = 64;
constexpr std::size_t kOutputFeatures = 10;
constexpr std::size_t kProducerCount = 4;
constexpr std::size_t kJobsPerProducer = 40;
constexpr std::size_t kProducerFutureWindow = 2;
constexpr std::array<std::size_t, 5> kBatchSizes = {1, 8, 32, 64, 256};
constexpr std::chrono::seconds kWaitTimeout{5};

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
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
    std::vector<float> values(batch * kInputFeatures);
    for (std::size_t row = 0; row < batch; ++row) {
        const std::size_t sample = (first_sample + row) % kReferenceSamples;
        const float* const source = reference.values.data() + sample * kInputFeatures;
        std::copy_n(source, kInputFeatures, values.data() + row * kInputFeatures);
    }
    return values;
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

struct ExpectedTask {
    std::size_t id;
    MlpInput input;
    OutputSnapshot expected;
    bool expect_overflow;
};

ExpectedTask prepare_task(std::size_t id, MlpInput input, bool expect_overflow,
                          const MlpModel& model, CpuExecutor& executor, Workspace& workspace) {
    try {
        const MlpResult direct = executor.run(model, input, workspace);
        if (expect_overflow) {
            throw std::runtime_error("maximum FP32 input did not overflow the direct executor");
        }
        return {id, std::move(input), snapshot(direct), false};
    } catch (const std::overflow_error&) {
        if (!expect_overflow) {
            throw;
        }
        return {id, std::move(input), {}, true};
    }
}

void compare_result(const MlpResult& actual, const OutputSnapshot& expected,
                    std::size_t task_id, std::string_view context) {
    expect(actual.batch() == expected.batch, "runtime and direct batch sizes differ");
    const auto logits = actual.logits();
    expect(logits.rows() == expected.batch && logits.cols() == kOutputFeatures,
           "runtime logits must have shape [batch,10]");
    expect(actual.labels().size() == expected.batch,
           "runtime labels must have one value per row");
    expect(expected.logits.size() == expected.batch * kOutputFeatures,
           "direct snapshot has an invalid logits size");
    expect(expected.labels.size() == expected.batch,
           "direct snapshot has an invalid label count");

    for (std::size_t row = 0; row < expected.batch; ++row) {
        for (std::size_t column = 0; column < kOutputFeatures; ++column) {
            const float actual_value = logits.at(row, column);
            const float direct_value = expected.logits[row * kOutputFeatures + column];
            if (!std::isfinite(actual_value) || actual_value != direct_value) {
                std::ostringstream message;
                message << context << ", task=" << task_id << ", row=" << row
                        << ", column=" << column << ", runtime=" << actual_value
                        << ", direct=" << direct_value;
                throw std::runtime_error(message.str());
            }
        }
        if (actual.labels()[row] != expected.labels[row]) {
            std::ostringstream message;
            message << context << ", task=" << task_id << ", row=" << row
                    << ", runtime label=" << actual.labels()[row]
                    << ", direct label=" << expected.labels[row];
            throw std::runtime_error(message.str());
        }
    }
}

class ProducerStartGate final {
public:
    explicit ProducerStartGate(std::size_t expected) : expected_(expected) {}

    bool arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++ready_;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_ || aborted_; });
        return !aborted_;
    }

    bool wait_until_ready() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, kWaitTimeout,
                                   [this] { return ready_ == expected_ || aborted_; }) &&
               ready_ == expected_;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        aborted_ = true;
        condition_.notify_all();
    }

private:
    const std::size_t expected_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t ready_ = 0;
    bool released_ = false;
    bool aborted_ = false;
};

class JoinThreads final {
public:
    explicit JoinThreads(std::vector<std::thread>& threads) noexcept : threads_(threads) {}
    ~JoinThreads() { join_all(); }

    JoinThreads(const JoinThreads&) = delete;
    JoinThreads& operator=(const JoinThreads&) = delete;

    void join_all() noexcept {
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                try {
                    thread.join();
                } catch (...) {
                    std::terminate();
                }
            }
        }
    }

private:
    std::vector<std::thread>& threads_;
};

struct ProducerOutcome {
    std::size_t attempted = 0;
    std::size_t accepted = 0;
    std::size_t queue_full_rejected = 0;
    std::size_t submission_failed = 0;
    std::size_t completed_successfully = 0;
    std::size_t completed_with_error = 0;
    std::size_t max_pending = 0;
    std::exception_ptr failure;
};

struct PendingFuture {
    std::size_t task_index;
    std::future<MlpResult> future;
};

void remember_current_exception(ProducerOutcome& outcome) noexcept {
    if (!outcome.failure) {
        outcome.failure = std::current_exception();
    }
}

void remember_failure(ProducerOutcome& outcome, std::string message) {
    try {
        throw std::runtime_error(std::move(message));
    } catch (...) {
        remember_current_exception(outcome);
    }
}

void consume_future(ProducerOutcome& outcome, const ExpectedTask& task,
                    PendingFuture& pending) {
    if (pending.future.wait_for(kWaitTimeout) != std::future_status::ready) {
        remember_failure(outcome, "accepted Runtime future did not become ready before timeout");
        pending.future.wait();
    }

    try {
        MlpResult result = pending.future.get();
        ++outcome.completed_successfully;
        if (task.expect_overflow) {
            remember_failure(outcome,
                             "maximum FP32 task completed without its expected overflow_error");
            return;
        }
        try {
            compare_result(result, task.expected, task.id, "concurrent Runtime result");
        } catch (...) {
            remember_current_exception(outcome);
        }
    } catch (const std::overflow_error&) {
        ++outcome.completed_with_error;
        if (!task.expect_overflow) {
            remember_failure(outcome, "ordinary finite task unexpectedly overflowed");
        }
    } catch (...) {
        ++outcome.completed_with_error;
        remember_current_exception(outcome);
    }
}

void consume_oldest(ProducerOutcome& outcome, const std::vector<ExpectedTask>& tasks,
                    std::vector<PendingFuture>& pending) {
    PendingFuture item = std::move(pending.front());
    pending.erase(pending.begin());
    consume_future(outcome, tasks[item.task_index], item);
}

void run_producer(Runtime& runtime, ProducerStartGate& start_gate,
                  const std::vector<ExpectedTask>& tasks,
                  const std::array<std::vector<std::size_t>, kProducerCount>& work,
                  std::array<ProducerOutcome, kProducerCount>& outcomes,
                  std::size_t producer_id) {
    if (!start_gate.arrive_and_wait()) {
        return;
    }

    ProducerOutcome& outcome = outcomes[producer_id];
    std::vector<PendingFuture> pending;
    try {
        pending.reserve(kProducerFutureWindow);
        for (const std::size_t task_index : work[producer_id]) {
            // Complete every planned job even under heavy rejection, rather
            // than allowing a saturated queue to turn this into a tiny test.
            const auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
            bool submitted = false;
            while (!submitted) {
                ++outcome.attempted;
                try {
                    auto future = runtime.submit(tasks[task_index].input);
                    ++outcome.accepted;
                    pending.push_back({task_index, std::move(future)});
                    outcome.max_pending = std::max(outcome.max_pending, pending.size());
                    submitted = true;
                } catch (const QueueFull&) {
                    ++outcome.queue_full_rejected;
                    if (!pending.empty()) consume_oldest(outcome, tasks, pending);
                    else std::this_thread::yield();
                } catch (...) {
                    ++outcome.submission_failed;
                    remember_current_exception(outcome);
                    submitted = true;  // Report this error after recovering outstanding futures.
                }
                if (pending.size() == kProducerFutureWindow) {
                    consume_oldest(outcome, tasks, pending);
                }
                if (!submitted && std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("producer could not submit its job before deadline");
                }
            }
        }
    } catch (...) {
        remember_current_exception(outcome);
    }

    while (!pending.empty()) {
        try {
            consume_oldest(outcome, tasks, pending);
        } catch (...) {
            remember_current_exception(outcome);
            // A valid accepted future must still be consumed before its owner exits.
            if (!pending.empty()) {
                pending.front().future.wait();
                try {
                    (void)pending.front().future.get();
                    ++outcome.completed_successfully;
                } catch (...) {
                    ++outcome.completed_with_error;
                }
                pending.erase(pending.begin());
            }
        }
    }
}

struct RetainedResult {
    std::size_t task_index;
    MlpResult result;
};

void collect_serial_result(Runtime& runtime, const std::vector<ExpectedTask>& tasks,
                           std::size_t task_index, std::size_t& attempted,
                           std::size_t& accepted, std::size_t& completed_successfully,
                           std::vector<RetainedResult>* retained) {
    const ExpectedTask& task = tasks[task_index];
    ++attempted;
    std::future<MlpResult> future;
    try {
        future = runtime.submit(task.input);
        ++accepted;
    } catch (const QueueFull&) {
        throw std::runtime_error("serial Runtime submission was unexpectedly rejected as QueueFull");
    }
    if (future.wait_for(kWaitTimeout) != std::future_status::ready) {
        future.wait();
        throw std::runtime_error("serial accepted Runtime future did not become ready before timeout");
    }

    MlpResult result = future.get();
    ++completed_successfully;
    compare_result(result, task.expected, task.id, "serial Runtime result");
    if (retained != nullptr) {
        retained->push_back({task_index, std::move(result)});
    }
}

void verify_producer_outcomes(const std::array<ProducerOutcome, kProducerCount>& outcomes,
                              std::size_t& attempted, std::size_t& accepted,
                              std::size_t& queue_full_rejected,
                              std::size_t& completed_successfully,
                              std::size_t& completed_with_error,
                              std::size_t& max_outstanding) {
    for (std::size_t producer = 0; producer < kProducerCount; ++producer) {
        const ProducerOutcome& outcome = outcomes[producer];
        attempted += outcome.attempted;
        accepted += outcome.accepted;
        queue_full_rejected += outcome.queue_full_rejected;
        completed_successfully += outcome.completed_successfully;
        completed_with_error += outcome.completed_with_error;
        max_outstanding += outcome.max_pending;

        expect(outcome.attempted == outcome.accepted + outcome.queue_full_rejected +
                                       outcome.submission_failed,
               "producer attempts must be partitioned into accepted, QueueFull, or other failure");
        expect(outcome.accepted == outcome.completed_successfully + outcome.completed_with_error,
               "every accepted producer future must be recovered exactly once");
        expect(outcome.max_pending <= kProducerFutureWindow,
               "a producer exceeded its finite future window");
        if (outcome.failure) {
            try {
                std::rethrow_exception(outcome.failure);
            } catch (const std::exception& error) {
                throw std::runtime_error("producer " + std::to_string(producer) +
                                         " failed: " + error.what());
            } catch (...) {
                throw std::runtime_error("producer " + std::to_string(producer) +
                                         " failed with an unknown exception");
            }
        }
        expect(outcome.submission_failed == 0,
               "Runtime submission failed with an exception other than QueueFull");
        expect(outcome.completed_with_error == 0,
               "ordinary concurrent Runtime task failed");
    }
    expect(max_outstanding <= kProducerCount * kProducerFutureWindow,
           "combined producer future window exceeded its fixed bound");
}

void run_worker_configuration(const fs::path& directory,
                              const std::vector<ExpectedTask>& stress_tasks,
                              const std::array<std::vector<std::size_t>, kProducerCount>& work,
                              const ExpectedTask& overflow_task,
                              const ExpectedTask& recovery_task,
                              const std::vector<ExpectedTask>& coverage_tasks,
                              std::size_t workers) {
    RuntimeConfig config;
    config.workers = workers;
    config.queue_capacity = 1;
    config.max_batch = 256;
    Runtime runtime(MlpModel::load(directory), config);

    std::array<ProducerOutcome, kProducerCount> outcomes;
    ProducerStartGate start_gate(kProducerCount);
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    JoinThreads joiner(producers);
    try {
        for (std::size_t producer = 0; producer < kProducerCount; ++producer) {
            producers.emplace_back([&, producer] {
                try {
                    run_producer(runtime, start_gate, stress_tasks, work, outcomes, producer);
                } catch (...) {
                    if (!outcomes[producer].failure) {
                        outcomes[producer].failure = std::current_exception();
                    }
                }
            });
        }
    } catch (...) {
        start_gate.abort();
        joiner.join_all();
        throw;
    }

    if (!start_gate.wait_until_ready()) {
        start_gate.abort();
        joiner.join_all();
        throw std::runtime_error("four Runtime producers did not reach the start gate");
    }
    start_gate.release();
    joiner.join_all();

    std::size_t producer_attempted = 0;
    std::size_t producer_accepted = 0;
    std::size_t producer_queue_full = 0;
    std::size_t producer_successes = 0;
    std::size_t producer_errors = 0;
    std::size_t max_outstanding = 0;
    verify_producer_outcomes(outcomes, producer_attempted, producer_accepted,
                             producer_queue_full, producer_successes, producer_errors,
                             max_outstanding);
    expect(producer_accepted == kProducerCount * kJobsPerProducer,
           "all planned producer jobs must be accepted and recovered despite queue saturation");
    expect(producer_accepted + producer_queue_full == producer_attempted,
           "QueueFull rejections must be excluded from accepted work");

    std::size_t serial_attempted = 0;
    std::size_t serial_accepted = 0;
    std::size_t serial_successes = 0;
    std::size_t expected_task_errors = 0;

    ++serial_attempted;
    std::future<MlpResult> failed_future = runtime.submit(overflow_task.input);
    ++serial_accepted;
    if (failed_future.wait_for(kWaitTimeout) != std::future_status::ready) {
        failed_future.wait();
        throw std::runtime_error("maximum FP32 task future did not become ready before timeout");
    }
    try {
        (void)failed_future.get();
        ++serial_successes;
        throw std::runtime_error("maximum FP32 overflow task completed successfully");
    } catch (const std::overflow_error&) {
        ++expected_task_errors;
    }

    collect_serial_result(runtime, std::vector<ExpectedTask>{recovery_task}, 0,
                          serial_attempted, serial_accepted, serial_successes, nullptr);

    // Keep the five results alive while later Runtime work runs.
    const std::vector<ExpectedTask>& all_tasks = coverage_tasks;
    std::vector<RetainedResult> retained;
    retained.reserve(kBatchSizes.size());
    for (std::size_t index = 0; index < all_tasks.size(); ++index) {
        collect_serial_result(runtime, all_tasks, index, serial_attempted, serial_accepted,
                              serial_successes, &retained);
    }

    expect(expected_task_errors == 1,
           "the accepted maximum FP32 task must deliver exactly one overflow_error");
    expect(serial_attempted == serial_accepted,
           "serial recovery and coverage submissions must all be accepted");
    expect(serial_accepted == serial_successes + expected_task_errors,
           "serial accepted futures must be recovered exactly once");
    expect(retained.size() == kBatchSizes.size(),
           "one result from every required batch size must be retained");

    for (std::size_t index = 0; index < retained.size(); ++index) {
        const RetainedResult& held = retained[index];
        const ExpectedTask& task = all_tasks[held.task_index];
        expect(held.result.batch() == kBatchSizes[index],
               "retained results must cover each required batch size exactly once");
        compare_result(held.result, task.expected, task.id,
                       "retained result after later Runtime executions");
        for (std::size_t other = 0; other < index; ++other) {
            expect(held.result.logits().data() != retained[other].result.logits().data(),
                   "independent Runtime results must own distinct logits storage");
            expect(held.result.labels().data() != retained[other].result.labels().data(),
                   "independent Runtime results must own distinct label storage");
        }
    }

    runtime.shutdown();
    const std::size_t total_attempted = producer_attempted + serial_attempted;
    const std::size_t total_accepted = producer_accepted + serial_accepted;
    const std::size_t total_successes = producer_successes + serial_successes;
    const std::size_t total_errors = producer_errors + expected_task_errors;
    expect(total_attempted == total_accepted + producer_queue_full,
           "attempted Runtime work must equal accepted work plus QueueFull rejections");
    expect(total_accepted == total_successes + total_errors,
           "each accepted Runtime task must have one successful or exceptional completion");

    std::cout << "  workers=" << workers << ": producers=" << kProducerCount
              << ", attempted=" << total_attempted << ", accepted=" << total_accepted
              << ", QueueFull_rejected=" << producer_queue_full
              << ", successful=" << total_successes << ", task_errors=" << total_errors
              << ", retained_batches=1/8/32/64/256, future_window<="
              << (kProducerCount * kProducerFutureWindow) << '\n';
}

void test_multi_producer_runtime(const fs::path& directory) {
    const ReferenceInputs reference = load_reference_inputs(directory);
    const MlpModel direct_model = MlpModel::load(directory);
    CpuExecutor executor;
    Workspace workspace(256);

    std::vector<ExpectedTask> stress_tasks;
    stress_tasks.reserve(kProducerCount * kJobsPerProducer);
    std::array<std::vector<std::size_t>, kProducerCount> work;
    for (std::size_t producer = 0; producer < kProducerCount; ++producer) {
        work[producer].reserve(kJobsPerProducer);
        for (std::size_t sequence = 0; sequence < kJobsPerProducer; ++sequence) {
            // Start each producer with maximum batches to create queue pressure after the
            // synchronized release; the rest of the sequence covers every required size.
            const std::size_t batch = sequence < 2
                                          ? kBatchSizes.back()
                                          : kBatchSizes[(producer + sequence - 2) %
                                                        kBatchSizes.size()];
            const std::size_t id = stress_tasks.size();
            const std::size_t first_sample = (producer * 13 + sequence * 7) % kReferenceSamples;
            stress_tasks.push_back(prepare_task(
                id, MlpInput(batch, make_input_values(reference, first_sample, batch)), false,
                direct_model, executor, workspace));
            work[producer].push_back(id);
        }
    }

    std::vector<float> maximum_values(kInputFeatures, std::numeric_limits<float>::max());
    const ExpectedTask overflow_task = prepare_task(
        stress_tasks.size(), MlpInput(1, std::move(maximum_values)), true, direct_model,
        executor, workspace);
    const ExpectedTask recovery_task = prepare_task(
        stress_tasks.size() + 1,
        MlpInput(1, make_input_values(reference, 37, 1)), false, direct_model, executor,
        workspace);

    std::vector<ExpectedTask> coverage;
    coverage.reserve(kBatchSizes.size());
    for (std::size_t index = 0; index < kBatchSizes.size(); ++index) {
        const std::size_t id = stress_tasks.size() + 2 + index;
        const std::size_t first_sample = (51 + index * 9) % kReferenceSamples;
        coverage.push_back(prepare_task(
            id, MlpInput(kBatchSizes[index],
                         make_input_values(reference, first_sample, kBatchSizes[index])),
            false, direct_model, executor, workspace));
    }

    for (const std::size_t workers : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        run_worker_configuration(directory, stress_tasks, work, overflow_task, recovery_task,
                                 coverage, workers);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "test_concurrency: expected exactly one reference data directory argument\n";
        return 2;
    }

    try {
        test_multi_producer_runtime(fs::path(argv[1]));
        std::cout << "test_concurrency: 1/2/4 Worker configurations passed; "
                     "4 producers; mixed batches=1/8/32/64/256\n";
    } catch (const std::exception& error) {
        std::cerr << "test_concurrency: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "test_concurrency: unknown failure\n";
        return 1;
    }
    return 0;
}
