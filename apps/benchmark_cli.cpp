#include "miniinfer/checked_size.hpp"
#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/data_io.hpp"
#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/runtime.hpp"
#include "miniinfer/workspace.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace {

constexpr std::size_t kMaxBatch = 256;
constexpr std::uint64_t kMaxAllocationProbeIterations = 100;

thread_local bool g_count_allocations = false;
thread_local std::size_t g_allocation_events = 0;

void* allocate_bytes(std::size_t size, std::size_t alignment = 0) {
    const std::size_t allocation_size = size == 0 ? 1 : size;
    void* allocation = nullptr;
    if (alignment == 0) {
        allocation = std::malloc(allocation_size);
    } else {
        if (posix_memalign(&allocation, alignment, allocation_size) != 0) {
            allocation = nullptr;
        }
    }
    if (allocation == nullptr) {
        throw std::bad_alloc{};
    }
    if (g_count_allocations && g_allocation_events != std::numeric_limits<std::size_t>::max()) {
        ++g_allocation_events;
    }
    return allocation;
}

class AllocationCountingScope final {
public:
    AllocationCountingScope() noexcept : previous_(g_count_allocations) {
        g_count_allocations = true;
    }
    AllocationCountingScope(const AllocationCountingScope&) = delete;
    AllocationCountingScope& operator=(const AllocationCountingScope&) = delete;
    ~AllocationCountingScope() { g_count_allocations = previous_; }

private:
    bool previous_;
};

struct Options final {
    std::filesystem::path model_directory;
    std::filesystem::path input_file;
    std::size_t batch = 1;
    std::uint64_t iterations = 1000;
    std::uint64_t warmup = 100;
    std::string mode = "direct_cpu";
    std::size_t workers = 1;
    std::size_t window = 8;
    std::size_t queue_capacity = 64;
};

void print_usage() {
    std::cout << "Usage: benchmark_cli --model DIR --input FILE [--batch N] [--iterations N] [--warmup N]\n"
                 "       benchmark_cli --help | -h\n"
                 "       [--mode direct_cpu|runtime] [--workers N] [--window N] [--queue-capacity N]\n"
                 "Run direct CPU or bounded Runtime inference and report JSON metrics.\n"
                 "Defaults: batch=1, iterations=1000, warmup=100. Batch must be in [1,256].\n"
                 "The input file must contain exactly batch*64 finite little-endian FP32 values.\n";
}

int report_error(std::string_view message, int exit_code = 2) {
    std::cerr << "error: " << message << '\n';
    return exit_code;
}

bool is_option(std::string_view token) {
    return token.size() >= 2 && token[0] == '-' && token[1] == '-';
}

bool parse_unsigned(std::string_view text, std::uint64_t& value) {
    if (text.empty()) {
        return false;
    }
    for (const char character : text) {
        if (character < '0' || character > '9') {
            return false;
        }
    }
    std::uint64_t parsed = 0;
    const auto conversion = std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
    if (conversion.ec != std::errc{} || conversion.ptr != text.data() + text.size()) {
        return false;
    }
    value = parsed;
    return true;
}

bool take_value(int argc, char* argv[], int& index, std::string_view option,
                std::string_view& value, std::string& error) {
    if (index + 1 >= argc) {
        error = "missing value for '" + std::string(option) + "'";
        return false;
    }
    const std::string_view candidate{argv[index + 1]};
    if (is_option(candidate) || candidate == "-h") {
        error = "missing value for '" + std::string(option) + "'";
        return false;
    }
    value = candidate;
    ++index;
    return true;
}

bool parse_options(int argc, char* argv[], Options& options, std::string& error) {
    bool have_model = false;
    bool have_input = false;
    bool have_batch = false;
    bool have_iterations = false;
    bool have_warmup = false;
    std::vector<std::string_view> extra_options;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help" || argument == "-h") {
            error = "'" + std::string(argument) + "' must be used without other arguments";
            return false;
        }

        if (argument == "--model") {
            if (have_model) {
                error = "duplicate option '--model'";
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            options.model_directory = std::filesystem::path(std::string(value));
            have_model = true;
        } else if (argument == "--input") {
            if (have_input) {
                error = "duplicate option '--input'";
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            options.input_file = std::filesystem::path(std::string(value));
            have_input = true;
        } else if (argument == "--batch") {
            if (have_batch) {
                error = "duplicate option '--batch'";
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            std::uint64_t parsed = 0;
            if (!parse_unsigned(value, parsed) || parsed < 1 || parsed > kMaxBatch) {
                error = "invalid value for '--batch': '" + std::string(value) +
                        "' (expected an integer in [1,256])";
                return false;
            }
            options.batch = static_cast<std::size_t>(parsed);
            have_batch = true;
        } else if (argument == "--iterations") {
            if (have_iterations) {
                error = "duplicate option '--iterations'";
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            if (!parse_unsigned(value, options.iterations) || options.iterations < 1) {
                error = "invalid value for '--iterations': '" + std::string(value) +
                        "' (expected an integer >= 1)";
                return false;
            }
            have_iterations = true;
        } else if (argument == "--warmup") {
            if (have_warmup) {
                error = "duplicate option '--warmup'";
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            if (!parse_unsigned(value, options.warmup)) {
                error = "invalid value for '--warmup': '" + std::string(value) +
                        "' (expected an integer >= 0)";
                return false;
            }
            have_warmup = true;
        } else if (argument == "--mode" || argument == "--workers" ||
                   argument == "--window" || argument == "--queue-capacity") {
            if (std::find(extra_options.begin(), extra_options.end(), argument) != extra_options.end()) {
                error = "duplicate option '" + std::string(argument) + "'";
                return false;
            }
            extra_options.push_back(argument);
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value, error)) return false;
            if (argument == "--mode") {
                if (value != "direct_cpu" && value != "runtime") {
                    error = "invalid value for '--mode'";
                    return false;
                }
                options.mode = std::string(value);
            } else {
                std::uint64_t parsed = 0;
                const std::uint64_t limit = argument == "--workers" ? 256 : 65536;
                if (!parse_unsigned(value, parsed) || parsed == 0 || parsed > limit) {
                    error = "invalid value for '" + std::string(argument) + "'";
                    return false;
                }
                if (argument == "--workers") options.workers = static_cast<std::size_t>(parsed);
                if (argument == "--window") options.window = static_cast<std::size_t>(parsed);
                if (argument == "--queue-capacity") options.queue_capacity = static_cast<std::size_t>(parsed);
            }
        } else {
            error = "unknown argument '" + std::string(argument) + "'";
            return false;
        }
    }

    if (!have_model) {
        error = "missing required parameter '--model'";
        return false;
    }
    if (!have_input) {
        error = "missing required parameter '--input'";
        return false;
    }
    return true;
}

std::uint64_t read_current_rss_bytes() {
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }

    const int descriptor = ::open("/proc/self/statm", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        throw std::runtime_error("cannot open /proc/self/statm: " +
                                 std::string(std::strerror(errno)));
    }
    char buffer[256];
    ssize_t count = 0;
    do {
        count = ::read(descriptor, buffer, sizeof(buffer));
    } while (count < 0 && errno == EINTR);
    const int read_error = count < 0 ? errno : 0;
    ::close(descriptor);
    if (count <= 0) {
        throw std::runtime_error("cannot read /proc/self/statm" +
                                 (read_error == 0 ? std::string{} :
                                                        ": " + std::string(std::strerror(read_error))));
    }

    const std::string_view contents(buffer, static_cast<std::size_t>(count));
    std::size_t position = 0;
    std::uint64_t resident_pages = 0;
    for (unsigned int field = 0; field < 2; ++field) {
        while (position < contents.size() &&
               (contents[position] == ' ' || contents[position] == '\t' ||
                contents[position] == '\n' || contents[position] == '\r')) {
            ++position;
        }
        const std::size_t start = position;
        while (position < contents.size() && contents[position] >= '0' &&
               contents[position] <= '9') {
            ++position;
        }
        if (start == position) {
            throw std::runtime_error("invalid resident page count in /proc/self/statm");
        }
        if (field == 1) {
            const auto parsed = std::from_chars(contents.data() + start,
                                                contents.data() + position, resident_pages, 10);
            if (parsed.ec != std::errc{} || parsed.ptr != contents.data() + position) {
                throw std::runtime_error("resident page count overflows in /proc/self/statm");
            }
        }
    }

    const auto page_bytes = static_cast<std::uint64_t>(page_size);
    if (resident_pages > std::numeric_limits<std::uint64_t>::max() / page_bytes) {
        throw std::runtime_error("resident byte count overflows");
    }
    return resident_pages * page_bytes;
}

std::uint64_t read_peak_rss_bytes() {
    struct rusage usage {};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage(RUSAGE_SELF) failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (usage.ru_maxrss < 0) {
        throw std::runtime_error("getrusage returned a negative ru_maxrss");
    }
    const auto peak_kib = static_cast<std::uint64_t>(usage.ru_maxrss);
    constexpr std::uint64_t kBytesPerKib = 1024;
    if (peak_kib > std::numeric_limits<std::uint64_t>::max() / kBytesPerKib) {
        throw std::runtime_error("peak RSS byte count overflows");
    }
    return peak_kib * kBytesPerKib;
}

double consume_result(const miniinfer::MlpResult& result, std::size_t expected_batch) {
    if (result.batch() != expected_batch || result.labels().size() != expected_batch) {
        throw std::runtime_error("executor returned an unexpected result shape");
    }
    const auto logits = result.logits();
    if (logits.rows() != expected_batch ||
        logits.cols() != miniinfer::MlpModel::output_features) {
        throw std::runtime_error("executor returned an unexpected logits shape");
    }

    double checksum = 0.0;
    for (std::size_t row = 0; row < expected_batch; ++row) {
        const std::int32_t label = result.labels()[row];
        if (label < 0 || label >= static_cast<std::int32_t>(miniinfer::MlpModel::output_features)) {
            throw std::runtime_error("executor returned a label outside the model output range");
        }
        checksum += static_cast<double>(label) * (1.0 + static_cast<double>(row) / 100.0);
        for (std::size_t column = 0; column < miniinfer::MlpModel::output_features; ++column) {
            const float value = logits.at(row, column);
            if (!std::isfinite(value)) {
                throw std::runtime_error("executor returned a non-finite logit");
            }
            const std::size_t flat_index = row * miniinfer::MlpModel::output_features + column;
            checksum += static_cast<double>(value) * (1.0 + static_cast<double>(flat_index) / 100.0);
        }
    }
    return checksum;
}

std::uint64_t nearest_rank_index(std::size_t sample_count, unsigned int percentile) {
    const std::uint64_t count = static_cast<std::uint64_t>(sample_count);
    const std::uint64_t whole = (count / 100U) * percentile;
    const std::uint64_t remainder = (count % 100U) * percentile;
    const std::uint64_t rank = whole + (remainder + 99U) / 100U;
    return rank - 1U;
}

double percentile_ms(const std::vector<double>& sorted_latencies, unsigned int percentile) {
    return sorted_latencies.at(static_cast<std::size_t>(
        nearest_rank_index(sorted_latencies.size(), percentile)));
}

void append_json_string(std::ostream& output, std::string_view value) {
    constexpr char kHex[] = "0123456789abcdef";
    output.put('"');
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u00" << kHex[character >> 4U] << kHex[character & 0x0fU];
                } else {
                    output.put(static_cast<char>(character));
                }
        }
    }
    output.put('"');
}

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Statistics {
    std::vector<double> values;
    void reserve(std::size_t count) { values.reserve(count); }
    void add(double value) {
        if (!std::isfinite(value) || value < 0) throw std::runtime_error("invalid timing sample");
        values.push_back(value);
    }
    void emit(std::ostream& json, std::string_view prefix) {
        std::sort(values.begin(), values.end());
        double sum = 0;
        for (double value : values) sum += value;
        const double mean = values.empty() ? 0 : sum / static_cast<double>(values.size());
        json << ",\"" << prefix << "mean_ms\":" << mean;
        for (unsigned int p : {50U, 95U, 99U}) {
            json << ",\"" << prefix << 'p' << p << "_ms\":"
                 << (values.empty() ? 0 : percentile_ms(values, p));
        }
    }
};

struct Measurements {
    std::uint64_t attempts = 0, accepted = 0, successes = 0, failures = 0;
    std::uint64_t queue_full = 0, closed = 0;
    std::size_t max_inflight = 0;
    double checksum = 0, wall_ms = 0;
    Statistics latency, queue, execution, completion, collection, submit;
    void reserve(std::size_t count) {
        for (auto* metric : {&latency, &queue, &execution, &completion, &collection, &submit}) {
            metric->reserve(count);
        }
    }
    void record(Clock::time_point before, Clock::time_point submitted,
                Clock::time_point obtained, const miniinfer::TaskTiming& timing) {
        latency.add(milliseconds(before, obtained));
        queue.add(milliseconds(timing.enqueued, timing.started));
        execution.add(milliseconds(timing.started, timing.finished));
        completion.add(milliseconds(before, timing.finished));
        collection.add(milliseconds(timing.finished, obtained));
        submit.add(milliseconds(before, submitted));
    }
};

struct Pending {
    std::future<miniinfer::MlpResult> future;
    std::shared_ptr<miniinfer::TaskTiming> timing;
    Clock::time_point before, submitted;
};

// A bounded closed-loop producer: scan every ready future before replenishment;
// only block on a future when the window is full or all attempts were submitted.
Measurements measure(const Options& options, const miniinfer::MlpInput& input,
                     const miniinfer::MlpModel& model, miniinfer::Workspace* workspace,
                     miniinfer::Runtime* runtime, std::uint64_t count, bool record) {
    Measurements metrics;
    if (record) metrics.reserve(static_cast<std::size_t>(count));
    std::vector<std::optional<Pending>> slots(runtime ? options.window : 0);
    std::size_t inflight = 0;
    const auto total_start = Clock::now();
    auto collect = [&](std::optional<Pending>& slot) {
        try {
            const auto result = slot->future.get();
            const auto obtained = Clock::now();
            const double checksum = consume_result(result, options.batch);
            metrics.checksum += checksum;
            if (record) metrics.record(slot->before, slot->submitted, obtained, *slot->timing);
            ++metrics.successes;
        } catch (const std::exception&) {
            ++metrics.failures;
        }
        slot.reset();
        --inflight;
    };
    if (!runtime) {
        for (std::uint64_t index = 0; index < count; ++index) {
            ++metrics.attempts;
            ++metrics.accepted;
            metrics.max_inflight = 1;
            const auto before = Clock::now();
            try {
                // Match Runtime's owned request copy, including it in call latency.
                const miniinfer::MlpInput owned(input);
                const auto submitted = Clock::now();
                miniinfer::TaskTiming timing;
                timing.enqueued = timing.started = Clock::now();
                const auto result = miniinfer::CpuExecutor{}.run(model, owned, *workspace);
                timing.finished = Clock::now();
                const auto obtained = timing.finished;
                metrics.checksum += consume_result(result, options.batch);
                if (record) metrics.record(before, submitted, obtained, timing);
                ++metrics.successes;
            } catch (const std::exception&) {
                ++metrics.failures;
            }
        }
    } else {
        while (metrics.attempts < count || inflight != 0) {
            for (auto& slot : slots) {
                if (slot && slot->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    collect(slot);
                }
            }
            if (metrics.attempts < count && inflight < options.window) {
                auto timing = std::make_shared<miniinfer::TaskTiming>();
                const auto before = Clock::now();
                ++metrics.attempts;
                try {
                    auto future = runtime->submit(input, timing);
                    const auto submitted = Clock::now();
                    auto empty = std::find_if(slots.begin(), slots.end(), [](const auto& slot) { return !slot; });
                    empty->emplace(Pending{std::move(future), std::move(timing), before, submitted});
                    ++metrics.accepted;
                    ++inflight;
                    metrics.max_inflight = std::max(metrics.max_inflight, inflight);
                } catch (const miniinfer::QueueFull&) {
                    ++metrics.queue_full;
                } catch (const miniinfer::PoolClosed&) {
                    ++metrics.closed;
                }
            } else if (inflight != 0) {
                auto pending = std::find_if(slots.begin(), slots.end(), [](const auto& slot) { return bool(slot); });
                (*pending)->future.wait();
            }
        }
    }
    metrics.wall_ms = milliseconds(total_start, Clock::now());
    if (metrics.attempts != metrics.accepted + metrics.queue_full + metrics.closed ||
        metrics.accepted != metrics.successes + metrics.failures || !std::isfinite(metrics.checksum)) {
        throw std::runtime_error("benchmark accounting mismatch");
    }
    return metrics;
}

int run_benchmark(const Options& options) {
    try {
        if (options.iterations > static_cast<std::uint64_t>(std::vector<double>{}.max_size())) {
            throw std::length_error("iterations exceed the latency sample vector limit");
        }
        const auto input_elements = miniinfer::checked_multiply(options.batch, miniinfer::MlpModel::input_features);
        auto input_buffer = miniinfer::load_f32_file(options.input_file, input_elements);
        const miniinfer::MlpInput input(options.batch,
            std::vector<float>(input_buffer.data(), input_buffer.data() + input_buffer.size()));
        auto model = miniinfer::MlpModel::load(options.model_directory);
        std::unique_ptr<miniinfer::Workspace> workspace;
        std::unique_ptr<miniinfer::Runtime> runtime;
        if (options.mode == "runtime") {
            runtime = std::make_unique<miniinfer::Runtime>(
                std::move(model),
                miniinfer::RuntimeConfig{options.workers, options.queue_capacity, kMaxBatch});
        } else {
            workspace = std::make_unique<miniinfer::Workspace>(kMaxBatch);
        }
        const auto warmup = measure(options, input, model, workspace.get(), runtime.get(), options.warmup, false);
        if (warmup.failures != 0) throw std::runtime_error("warmup task failed");
        const auto rss_after_warmup = read_current_rss_bytes();

        // The D8 allocation probe remains a separate direct-executor probe.
        // Runtime instrumentation/worker allocations are deliberately not counted.
        const auto probe_iterations = runtime ? 0 : std::min(options.iterations, kMaxAllocationProbeIterations);
        std::uint64_t probe_failures = 0;
        g_allocation_events = 0;
        {
            AllocationCountingScope scope;
            for (std::uint64_t index = 0; index < probe_iterations; ++index) {
                try {
                    const auto result = miniinfer::CpuExecutor{}.run(model, input, *workspace);
                    (void)consume_result(result, options.batch);
                } catch (const std::exception&) {
                    ++probe_failures;
                }
            }
        }
        const auto probe_events = g_allocation_events;
        auto metrics = measure(options, input, model, workspace.get(), runtime.get(), options.iterations, true);
        const auto rss_after_timing = read_current_rss_bytes();
        if (runtime) runtime->shutdown();
        struct utsname system_info {};
        if (::uname(&system_info) != 0) throw std::runtime_error("uname failed");
        const double rate = metrics.wall_ms > 0 ? static_cast<double>(metrics.successes) * 1000 / metrics.wall_ms : 0;
        if (!std::isfinite(rate) || !std::isfinite(rate * options.batch)) throw std::runtime_error("non-finite throughput");
        std::ostringstream json;
        json.imbue(std::locale::classic());
        json << std::setprecision(std::numeric_limits<double>::max_digits10);
        json << "{\"mode\":";
        append_json_string(json, options.mode);
        json << ",\"workers\":" << (runtime ? options.workers : 0)
             << ",\"batch\":" << options.batch << ",\"iterations\":" << options.iterations
             << ",\"warmup\":" << options.warmup << ",\"window\":" << options.window
             << ",\"queue_capacity\":" << options.queue_capacity << ",\"max_inflight\":" << metrics.max_inflight
             << ",\"attempts\":" << metrics.attempts << ",\"accepted\":" << metrics.accepted
             << ",\"successes\":" << metrics.successes << ",\"failures\":" << metrics.failures
             << ",\"rejected\":" << metrics.queue_full + metrics.closed
             << ",\"queue_full_rejected\":" << metrics.queue_full << ",\"closed_rejected\":" << metrics.closed
             << ",\"wall_ms\":" << metrics.wall_ms << ",\"req_per_sec\":" << rate
             << ",\"samples_per_sec\":" << rate * options.batch << ",\"checksum\":" << metrics.checksum;
        metrics.latency.emit(json, "");
        metrics.queue.emit(json, "queue_");
        metrics.execution.emit(json, "execution_");
        metrics.completion.emit(json, "completion_");
        metrics.collection.emit(json, "collection_");
        metrics.submit.emit(json, "submit_");
        json << ",\"latency_samples\":" << metrics.latency.values.size()
             << ",\"allocation_probe_iterations\":" << probe_iterations
             << ",\"allocation_probe_new_events\":" << probe_events
             << ",\"allocation_probe_failures\":" << probe_failures
             << ",\"allocation_probe_available\":" << (runtime ? "false" : "true")
             << ",\"allocation_probe_scope\":";
        append_json_string(json, "Separate direct CpuExecutor probe counts replaceable global C++ operator new/new[] and aligned forms using a thread-local counter; excludes malloc/mmap, input copies, timing records, pool setup and worker-thread allocations. Runtime mode has no allocation probe.");
        json << ",\"rss_after_warmup_bytes\":" << rss_after_warmup
             << ",\"rss_after_timing_bytes\":" << rss_after_timing
             << ",\"peak_rss_bytes\":" << read_peak_rss_bytes() << ",\"rss_scope\":";
        append_json_string(json, "Resident pages from /proc/self/statm times page size; sampled with Runtime still alive after warmup and after timing. Includes model, workspaces, threads, bounded pending futures, latency vectors and allocator retention; results consumed and destroyed promptly.");
        json << ",\"peak_rss_scope\":";
        append_json_string(json, "getrusage(RUSAGE_SELF) ru_maxrss process high-water mark, Linux KiB converted to bytes; includes startup, preparation, warmup, timing and reporting, not executor-only peak.");
        json << ",\"timing_scope\":";
        append_json_string(json, "Formal wall time includes input copies, submissions, rejected attempts, ready-future scans, waits, successful result consumption/destruction and metric recording; setup, warmup, allocation probe, final shutdown and reporting excluded.");
        json << ",\"latency_scope\":";
        append_json_string(json, "Successful requests only: call-before (before owned input copy) to future.get result acquisition. queue=start-enqueued under queue lock; execution=finish-start (callable); completion=finish-call-before; collection=acquired-finish; submit=submit-return-call-before. Direct execution has zero queue/collection. All stages include instrumentation overhead.");
        json << ",\"latency_percentile_method\":\"nearest-rank: sorted sample at ceil(p*N), one-indexed\""
             << ",\"timer\":\"std::chrono::steady_clock\",\"compiler\":";
        append_json_string(json, __VERSION__);
#ifdef NDEBUG
        json << ",\"build_type\":\"Release\"";
#else
        json << ",\"build_type\":\"Debug\"";
#endif
        json << ",\"cxx_standard\":17,\"kernel\":";
        append_json_string(json, system_info.release);
        json << ",\"architecture\":";
        append_json_string(json, system_info.machine);
        json << ",\"hardware_threads\":" << std::thread::hardware_concurrency() << "}\n";
        std::cout << json.str();
        return 0;
    } catch (const std::exception& error) {
        return report_error(error.what(), 1);
    } catch (...) {
        return report_error("unknown benchmark failure", 1);
    }
}
}  // namespace

void* operator new(std::size_t size) { return allocate_bytes(size); }
void* operator new[](std::size_t size) { return allocate_bytes(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocate_bytes(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocate_bytes(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        const std::string_view first{argv[1]};
        if (first == "--help" || first == "-h") {
            if (argc != 2) {
                return report_error("unexpected extra argument '" +
                                    std::string(argv[2]) + "'");
            }
            print_usage();
            return 0;
        }
    }

    Options options;
    std::string error;
    if (!parse_options(argc, argv, options, error)) {
        return report_error(error);
    }
    return run_benchmark(options);
}
