#include "miniinfer/depth/sdk.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

namespace {
using namespace miniinfer::depth;
using Clock = std::chrono::steady_clock;
struct Args {
    std::filesystem::path package, image, csv, summary;
    std::string backend;
    std::size_t concurrency = 1, iterations = 1000, warmup = 20;
    SdkConfig config;
    Args() { config.byte_budget = 512ULL * 1024 * 1024; }
};
std::size_t number(const std::string& value, bool allow_zero = false) {
    std::size_t result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || (!allow_zero && !result))
        throw std::invalid_argument("invalid integer: " + value);
    return result;
}
Args parse(int argc, char** argv) {
    Args a;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc || !seen.insert(key).second)
            throw std::invalid_argument("missing/duplicate option: " + key);
        const std::string value = argv[i + 1];
        if (key == "--package") a.package = value;
        else if (key == "--backend") a.backend = value;
        else if (key == "--image") a.image = value;
        else if (key == "--csv") a.csv = value;
        else if (key == "--summary") a.summary = value;
        else if (key == "--concurrency") a.concurrency = number(value);
        else if (key == "--iterations") a.iterations = number(value);
        else if (key == "--warmup") a.warmup = number(value, true);
        else if (key == "--worker-count") a.config.worker_count = number(value);
        else if (key == "--queue-capacity") a.config.queue_capacity = number(value);
        else if (key == "--byte-budget") a.config.byte_budget = number(value);
        else if (key == "--ort-threads") {
            auto n = number(value);
            if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("ort-threads exceeds int range");
            a.config.ort_threads = static_cast<int>(n);
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (a.package.empty() || a.image.empty() || a.csv.empty() || a.summary.empty())
        throw std::invalid_argument("package/image/csv/summary are required");
    if (a.backend != "ort" && a.backend != "trt") throw std::invalid_argument("backend must be ort/trt");
    if (a.concurrency > 256 || a.iterations < a.concurrency)
        throw std::invalid_argument("concurrency must be <=256 and iterations >= concurrency");
    if (std::filesystem::absolute(a.csv).lexically_normal() == std::filesystem::absolute(a.summary).lexically_normal())
        throw std::invalid_argument("CSV and summary must be distinct paths");
    if (std::filesystem::exists(a.csv) || std::filesystem::exists(a.summary))
        throw std::invalid_argument("refusing existing evidence output");
    a.config.package_dir = a.package;
    return a;
}
std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}
std::uint64_t checksum(const DepthMap& depth, const RgbImage& image) {
    if (depth.width() != image.width() || depth.height() != image.height() ||
        depth.values().size() != image.width() * image.height())
        throw std::runtime_error("result dimensions mismatch");
    std::uint64_t hash = 14695981039346656037ULL;
    for (float value : depth.values()) {
        if (!std::isfinite(value)) throw std::runtime_error("non-finite result");
        std::uint32_t bits;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        // Canonical low-byte-first float bits, independent of host endianness.
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (bits >> shift) & 255U;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}
double percentile(const std::vector<double>& sorted, double q) {
    const double position = (sorted.size() - 1) * q;
    const auto low = static_cast<std::size_t>(position);
    const auto high = static_cast<std::size_t>(std::ceil(position));
    return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
struct Record {
    std::size_t caller = 0;
    std::string status = "failure", error;
    double wall_ms = 0;
    std::optional<std::uint64_t> hash;
};
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "depth_concurrency_benchmark --backend ort|trt --package DIR --image PPM --csv NEW.csv --summary NEW.json\n"
                     "[--concurrency 1] [--iterations 1000] [--warmup 20] [--worker-count 4]\n"
                     "[--ort-threads 4] [--queue-capacity 4] [--byte-budget 536870912]\n"
                     "Iterations are total steady attempts; synchronous callers share one SDK. No retries.\n";
        return 0;
    }
    try {
        const auto a = parse(argc, argv);
        const auto image = read_ppm(a.image);
        DepthSdk sdk(a.config);
        const auto initial = sdk.snapshot();
        if (initial.backend != a.backend) throw std::invalid_argument("package backend differs from requested backend");
        const auto baseline = sdk.run(image);
        const auto expected = checksum(baseline, image);
        for (std::size_t i = 0; i < a.warmup; ++i) {
            const auto result = sdk.run(image);
            if (result.values().data() == baseline.values().data() || checksum(result, image) != expected)
                throw std::runtime_error("warmup ownership/checksum mismatch");
        }
        const auto before = sdk.snapshot();
        std::vector<Record> records(a.iterations);
        std::vector<std::optional<DepthMap>> retained(a.concurrency);
        std::vector<Clock::time_point> finished(a.concurrency);
        std::atomic<std::size_t> next{0};
        std::mutex gate_mutex;
        std::condition_variable gate;
        std::size_t ready = 0;
        bool start = false, cancel = false;
        std::vector<std::thread> callers;
        callers.reserve(a.concurrency);
        // No throwing initialization inside a caller before it signals readiness.
        auto caller = [&](std::size_t id) {
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                ++ready;
                gate.notify_all();
                gate.wait(lock, [&] { return start || cancel; });
                if (cancel) return;
            }
            for (;;) {
                const auto index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= a.iterations) break;
                auto& record = records[index];
                record.caller = id;
                const auto began = Clock::now();
                std::optional<DepthMap> result;
                try {
                    result.emplace(sdk.run(image));
                    record.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
                    record.status = "succeeded";
                } catch (const SdkError& error) {
                    record.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
                    record.status = error.code() == SdkErrorCode::Busy ? "busy" :
                                    error.code() == SdkErrorCode::Closed ? "closed" : "failure";
                    record.error = error.what();
                } catch (const std::exception& error) {
                    record.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
                    record.error = error.what();
                } catch (...) {
                    record.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
                    record.error = "unknown SDK exception";
                }
                // Validation is outside request latency, inside steady wall/QPS.
                if (result) {
                    try {
                        const auto pointer = result->values().data();
                        if (pointer == baseline.values().data() ||
                            (retained[id] && pointer == retained[id]->values().data()))
                            throw std::runtime_error("result does not own independent storage");
                        record.hash = checksum(*result, image);
                        if (*record.hash != expected) throw std::runtime_error("result checksum differs from baseline");
                        if (retained[id] && checksum(*retained[id], image) != expected)
                            throw std::runtime_error("previous result changed after subsequent run");
                        retained[id] = std::move(result);
                    } catch (const std::exception& error) {
                        record.status = "failure";
                        record.error = error.what();
                    }
                }
            }
            finished[id] = Clock::now();
        };
        try {
            for (std::size_t i = 0; i < a.concurrency; ++i) callers.emplace_back(caller, i);
        } catch (...) {
            { std::lock_guard<std::mutex> lock(gate_mutex); cancel = true; }
            gate.notify_all();
            for (auto& thread : callers) thread.join();
            throw;
        }
        Clock::time_point began;
        {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate.wait(lock, [&] { return ready == a.concurrency; });
            began = Clock::now();
            start = true;
        }
        gate.notify_all();
        for (auto& thread : callers) thread.join();
        const auto ended = *std::max_element(finished.begin(), finished.end());
        const double elapsed = std::chrono::duration<double, std::milli>(ended - began).count();
        sdk.shutdown();
        const auto after = sdk.snapshot();
        bool ownership_ok = checksum(baseline, image) == expected;
        std::set<const float*> pointers{baseline.values().data()};
        for (const auto& result : retained) if (result) {
            ownership_ok = ownership_ok && checksum(*result, image) == expected && pointers.insert(result->values().data()).second;
        }
        const bool drained = after.inflight == 0 && after.reserved_bytes == 0 && after.queued == 0 && after.executing == 0 && after.shutdown_complete;
        std::size_t succeeded = 0, busy = 0, closed = 0, failure = 0;
        std::vector<double> successful;
        std::ostringstream csv;
        csv << std::setprecision(17) << "index,caller,status,wall_ms,checksum,error\n";
        for (std::size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            if (r.status == "succeeded") { ++succeeded; successful.push_back(r.wall_ms); }
            else if (r.status == "busy") ++busy;
            else if (r.status == "closed") ++closed;
            else ++failure;
            csv << i << ',' << r.caller << ',' << r.status << ',' << r.wall_ms << ',';
            if (r.hash) csv << *r.hash;
            // CSV quoting differs from JSON; messages may contain commas/newlines/quotes.
            csv << ",\"";
            for (char c : r.error) { if (c == '"') csv << '"'; csv << c; }
            csv << "\"\n";
        }
        std::sort(successful.begin(), successful.end());
        const bool counters_ok = after.accepted - before.accepted == succeeded + failure &&
            after.delivered - before.delivered == succeeded + failure &&
            after.rejected_busy - before.rejected_busy == busy && after.rejected_closed - before.rejected_closed == closed;
        std::ostringstream json;
        json << std::setprecision(17)
             << "{\"schema\":\"miniinfer-depth-concurrency-v1\",\"backend\":" << quote(after.backend)
             << ",\"precision\":" << quote(after.precision) << ",\"package\":" << quote(a.package.string())
             << ",\"image\":" << quote(a.image.string()) << ",\"width\":" << image.width() << ",\"height\":" << image.height()
             << ",\"concurrency\":" << a.concurrency << ",\"iterations\":" << a.iterations << ",\"warmup\":" << a.warmup
             << ",\"iteration_scope\":\"total steady attempts\",\"baseline_requests\":1,\"worker_count\":" << a.config.worker_count
             << ",\"ort_threads\":" << a.config.ort_threads << ",\"queue_capacity\":" << a.config.queue_capacity
             << ",\"byte_budget\":" << a.config.byte_budget << ",\"reservation_bytes_per_request\":" << DepthSdk::reservation_bytes(image.width(), image.height())
             << ",\"budget_scope\":\"admission reservations; excludes caller storage, weights and backend arena/context\""
             << ",\"counts\":{\"succeeded\":" << succeeded << ",\"busy\":" << busy << ",\"closed\":" << closed << ",\"failure\":" << failure << "}"
             << ",\"elapsed_steady_ms\":" << elapsed << ",\"successful_qps\":" << (succeeded * 1000.0 / elapsed)
             << ",\"p50_ms\":" << (successful.empty() ? "null" : [&] { std::ostringstream s; s << std::setprecision(17) << percentile(successful, .5); return s.str(); }())
             << ",\"p99_ms\":" << (successful.empty() ? "null" : [&] { std::ostringstream s; s << std::setprecision(17) << percentile(successful, .99); return s.str(); }())
             << ",\"latency_scope\":\"synchronous SDK run including admission, RGB copy, queue, preprocess, inference and postprocess\""
             << ",\"steady_scope\":\"unified release through last caller completion; includes validation and scheduling, excludes setup/warmup/file IO/shutdown\""
             << ",\"percentiles\":\"successful requests only; linear interpolation across all callers\""
             << ",\"checksum_algorithm\":\"FNV1a64 canonical FP32 bytes; exact comparison to one baseline\",\"baseline_checksum\":" << quote(std::to_string(expected))
             << ",\"ownership_ok\":" << (ownership_ok ? "true" : "false") << ",\"counters_ok\":" << (counters_ok ? "true" : "false")
             << ",\"drained\":" << (drained ? "true" : "false")
             << ",\"sdk_snapshot\":{\"accepted\":" << after.accepted << ",\"succeeded\":" << after.succeeded
             << ",\"failed\":" << after.failed << ",\"delivered\":" << after.delivered << ",\"inflight\":" << after.inflight
             << ",\"queued\":" << after.queued << ",\"executing\":" << after.executing << ",\"reserved_bytes\":" << after.reserved_bytes << "}}\n";
        write_text_exclusive(a.csv, csv.str());
        write_text_exclusive(a.summary, json.str());
        std::cout << json.str();
        return ownership_ok && counters_ok && drained && !closed && !failure ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "depth_concurrency_benchmark: " << error.what() << '\n';
        return 1;
    }
}
