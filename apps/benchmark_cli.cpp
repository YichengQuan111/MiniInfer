#include "miniinfer/checked_size.hpp"
#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/data_io.hpp"
#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
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
#include <new>
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
};

void print_usage() {
    std::cout << "Usage: benchmark_cli --model DIR --input FILE [--batch N] [--iterations N] [--warmup N]\n"
                 "       benchmark_cli --help | -h\n"
                 "Run synchronous direct CPU inference and report JSON timing and allocation data.\n"
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

std::string build_json_prefix(const Options& options, std::uint64_t allocation_probe_iterations,
                              std::size_t allocation_probe_new_events, std::uint64_t rss_after_warmup,
                              std::uint64_t rss_after_timing, double wall_ms,
                              double req_per_sec, double samples_per_sec, double mean_ms,
                              double p50_ms, double p95_ms, double p99_ms, double checksum,
                              unsigned int hardware_threads, std::string_view kernel,
                              std::string_view architecture) {
    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(std::numeric_limits<double>::max_digits10);
    json << "{\"mode\":\"direct_cpu\",\"batch\":" << options.batch
         << ",\"iterations\":" << options.iterations
         << ",\"warmup\":" << options.warmup
         << ",\"successes\":" << options.iterations
         << ",\"failures\":0,\"rejected\":0"
         << ",\"wall_ms\":" << wall_ms
         << ",\"req_per_sec\":" << req_per_sec
         << ",\"samples_per_sec\":" << samples_per_sec
         << ",\"mean_ms\":" << mean_ms
         << ",\"p50_ms\":" << p50_ms
         << ",\"p95_ms\":" << p95_ms
         << ",\"p99_ms\":" << p99_ms
         << ",\"checksum\":" << checksum
         << ",\"allocation_probe_iterations\":" << allocation_probe_iterations
         << ",\"allocation_probe_new_events\":" << allocation_probe_new_events
         << ",\"allocation_probe_scope\":";
    append_json_string(json,
        "Counts replaceable global C++ operator new/new[] and aligned forms via a thread-local counter; excludes malloc/mmap and allocations that bypass these operators.");
    json << ",\"rss_after_warmup_bytes\":" << rss_after_warmup
         << ",\"rss_after_timing_bytes\":" << rss_after_timing
         << ",\"peak_rss_scope\":";
    append_json_string(json, "getrusage(RUSAGE_SELF) ru_maxrss high-water mark for the process, sampled after the benchmark; Linux KiB converted to bytes.");
    json << ",\"compiler\":";
    append_json_string(json, __VERSION__);
#ifdef NDEBUG
    json << ",\"build_type\":\"Release\"";
#else
    json << ",\"build_type\":\"Debug\"";
#endif
    json << ",\"cxx_standard\":17,\"kernel\":";
    append_json_string(json, kernel);
    json << ",\"architecture\":";
    append_json_string(json, architecture);
    json << ",\"hardware_threads\":" << hardware_threads
         << ",\"hardware_threads_scope\":";
    append_json_string(json, "std::thread::hardware_concurrency logical-thread hint, not a physical-core count");
    json << ",\"timer\":\"std::chrono::steady_clock\""
         << ",\"timing_scope\":";
    append_json_string(json, "Total steady-clock interval around formal iterations, including result checks/consumption/destruction and latency recording; setup, warmup, allocation probe, and reporting are excluded.");
    json << ",\"latency_scope\":";
    append_json_string(json, "Per-iteration steady-clock interval around executor.run, result checks/consumption, and result destruction.");
    json << ",\"latency_percentile_method\":";
    append_json_string(json, "nearest-rank: sorted sample at ceil(p*N), one-indexed");
    json << ",\"rss_scope\":";
    append_json_string(json, "Resident pages from /proc/self/statm multiplied by sysconf(_SC_PAGESIZE); sampled after warmup and after formal timing.");
    return json.str();
}

int run_benchmark(const Options& options) {
    try {
        const std::size_t input_elements = miniinfer::checked_multiply(
            options.batch, miniinfer::MlpModel::input_features);
        auto input_buffer = miniinfer::load_f32_file(options.input_file, input_elements);
        std::vector<float> input_values(input_buffer.data(),
                                        input_buffer.data() + input_buffer.size());
        const miniinfer::MlpInput input(options.batch, std::move(input_values));
        const auto model = miniinfer::MlpModel::load(options.model_directory);
        miniinfer::Workspace workspace(kMaxBatch);
        const miniinfer::CpuExecutor executor;

        for (std::uint64_t iteration = 0; iteration < options.warmup; ++iteration) {
            const auto result = executor.run(model, input, workspace);
            const double warmup_checksum = consume_result(result, options.batch);
            if (!std::isfinite(warmup_checksum)) {
                throw std::runtime_error("warmup result checksum is non-finite");
            }
        }
        const std::uint64_t rss_after_warmup = read_current_rss_bytes();

        const std::uint64_t allocation_probe_iterations =
            std::min(options.iterations, kMaxAllocationProbeIterations);
        g_allocation_events = 0;
        {
            AllocationCountingScope count_allocations;
            for (std::uint64_t iteration = 0; iteration < allocation_probe_iterations; ++iteration) {
                const auto result = executor.run(model, input, workspace);
                const double probe_checksum = consume_result(result, options.batch);
                if (!std::isfinite(probe_checksum)) {
                    throw std::runtime_error("allocation probe result checksum is non-finite");
                }
            }
        }
        const std::size_t allocation_probe_new_events = g_allocation_events;

        if (options.iterations > static_cast<std::uint64_t>(
                                     std::vector<double>{}.max_size())) {
            throw std::length_error("iterations exceed the latency sample vector limit");
        }
        std::vector<double> latencies;
        latencies.reserve(static_cast<std::size_t>(options.iterations));

        double checksum = 0.0;
        const auto total_start = std::chrono::steady_clock::now();
        for (std::uint64_t iteration = 0; iteration < options.iterations; ++iteration) {
            const auto request_start = std::chrono::steady_clock::now();
            {
                const auto result = executor.run(model, input, workspace);
                checksum += consume_result(result, options.batch);
            }
            const auto request_end = std::chrono::steady_clock::now();
            const double latency_ms =
                std::chrono::duration<double, std::milli>(request_end - request_start).count();
            latencies.push_back(latency_ms);
        }
        const auto total_end = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();
        const std::uint64_t rss_after_timing = read_current_rss_bytes();

        std::sort(latencies.begin(), latencies.end());
        double latency_sum = 0.0;
        for (const double latency : latencies) {
            latency_sum += latency;
        }
        const double mean_ms = latency_sum / static_cast<double>(latencies.size());
        const double wall_seconds = wall_ms / 1000.0;
        const double req_per_sec = wall_seconds > 0.0
                                       ? static_cast<double>(options.iterations) / wall_seconds
                                       : 0.0;
        const double samples_per_sec = req_per_sec * static_cast<double>(options.batch);
        const double p50_ms = percentile_ms(latencies, 50);
        const double p95_ms = percentile_ms(latencies, 95);
        const double p99_ms = percentile_ms(latencies, 99);
        if (!std::isfinite(wall_ms) || !std::isfinite(req_per_sec) ||
            !std::isfinite(samples_per_sec) || !std::isfinite(mean_ms) ||
            !std::isfinite(p50_ms) || !std::isfinite(p95_ms) ||
            !std::isfinite(p99_ms) || !std::isfinite(checksum)) {
            throw std::runtime_error("benchmark produced a non-finite metric");
        }

        struct utsname system_info {};
        if (::uname(&system_info) != 0) {
            throw std::runtime_error("uname failed: " + std::string(std::strerror(errno)));
        }
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        std::string output = build_json_prefix(
            options, allocation_probe_iterations, allocation_probe_new_events,
            rss_after_warmup, rss_after_timing, wall_ms, req_per_sec, samples_per_sec,
            mean_ms, p50_ms, p95_ms, p99_ms, checksum, hardware_threads,
            system_info.release, system_info.machine);
        output.reserve(output.size() + 64);

        const std::uint64_t peak_rss_bytes = read_peak_rss_bytes();
        char peak_buffer[32];
        const auto peak_conversion = std::to_chars(
            peak_buffer, peak_buffer + sizeof(peak_buffer), peak_rss_bytes);
        if (peak_conversion.ec != std::errc{}) {
            throw std::runtime_error("could not format peak RSS");
        }
        output.append(",\"peak_rss_bytes\":");
        output.append(peak_buffer, peak_conversion.ptr);
        output.append("}\n");
        std::cout << output;
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
