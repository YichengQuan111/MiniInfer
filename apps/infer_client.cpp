#include "miniinfer/data_io.hpp"
#include "miniinfer/net/protocol.hpp"
#include "miniinfer/net/socket.hpp"
#include <charconv>
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <system_error>

namespace {
unsigned number(const std::string& text, unsigned maximum, bool allow_zero = false) {
    unsigned value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || (!allow_zero && value == 0) || value > maximum)
        throw std::invalid_argument("invalid numeric option: " + text);
    return value;
}
struct Options {
    std::string host = "127.0.0.1", input;
    unsigned port = 9000, batch = 1, requests = 1, warmup = 0, timeout_ms = 5000;
};
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[i];
        if (name == "--host") options.host = value;
        else if (name == "--port") options.port = number(value, 65535);
        else if (name == "--input") options.input = value;
        else if (name == "--batch") options.batch = number(value, 256);
        else if (name == "--requests") options.requests = number(value, 1000000);
        else if (name == "--warmup") options.warmup = number(value, 1000000, true);
        else if (name == "--timeout-ms") options.timeout_ms = number(value, 600000);
        else throw std::invalid_argument("unknown option: " + name);
    }
    if (options.input.empty()) throw std::invalid_argument("--input is required");
    return options;
}
struct RequestFailure {
    enum Kind { Timeout, Connection, Protocol } kind;
};
void require_io(miniinfer::net::IoResult io) {
    using miniinfer::net::IoStatus;
    if (io.status == IoStatus::Ok) return;
    if (io.status == IoStatus::Timeout) throw RequestFailure{RequestFailure::Timeout};
    if (io.status == IoStatus::Truncated) throw RequestFailure{RequestFailure::Protocol};
    throw RequestFailure{RequestFailure::Connection};
}
double milliseconds(miniinfer::net::Deadline begin, miniinfer::net::Deadline end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
struct Metrics {
    unsigned attempts = 0, successes = 0, business_errors = 0, connection_errors = 0, protocol_errors = 0, timeouts = 0;
    unsigned status_counts[5]{};
    std::vector<double> latency;
    void error(RequestFailure::Kind kind) {
        if (kind == RequestFailure::Timeout) ++timeouts;
        else if (kind == RequestFailure::Protocol) ++protocol_errors;
        else ++connection_errors;
    }
    double percentile(double p) const {
        return latency.empty() ? 0.0 : latency[static_cast<std::size_t>(std::ceil(p * latency.size())) - 1];
    }
};
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "infer_client --input FILE [--batch 1..256] [--host IPv4] [--port N]\n"
                     "             [--requests N] [--warmup N] [--timeout-ms N]\n"
                     "Input is already normalized little-endian FP32; one reusable connection.\n";
        return 0;
    }
    Options options;
    try { options = parse(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    try {
        using namespace miniinfer::net;
        check_wire_platform();
        const auto input = miniinfer::load_f32_file(options.input, options.batch * 64);
        const miniinfer::MlpInput values(options.batch, {input.data(), input.data() + input.size()});
        auto request = mlp_request(1, values);
        const auto timeout = std::chrono::milliseconds(options.timeout_ms);
        const auto before_connect = Clock::now();
        UniqueFd socket;
        unsigned connect_errors = 0, connect_timeouts = 0;
        try { socket = connect_tcp_until(before_connect + timeout, options.host, static_cast<std::uint16_t>(options.port)); }
        catch (const std::system_error& error) {
            if (error.code().value() == ETIMEDOUT) ++connect_timeouts;
            else ++connect_errors;
            std::cerr << error.what() << '\n';
        }
        const auto connect_ms = milliseconds(before_connect, Clock::now());
        Response result;
        Metrics metrics;
        metrics.latency.reserve(options.requests);
        unsigned warmup_attempts = 0, warmup_successes = 0;
        double wall_ms = 0;
        auto wall_start = Clock::now();
        for (unsigned i = 1; socket && i <= options.warmup + options.requests; ++i) {
            const bool warmup = i <= options.warmup;
            if (!warmup && metrics.attempts == 0) wall_start = Clock::now();
            if (warmup) ++warmup_attempts; else ++metrics.attempts;
            request.request_id = i;
            bool end_connection = false;
            try {
                const auto before = Clock::now();
                require_io(send_frame(socket.get(), request, before + timeout));
                Frame reply;
                require_io(recv_frame(socket.get(), reply, Clock::now() + timeout));
                auto decoded = decode_response(reply, mlp_type, i, options.batch);
                const auto completed = Clock::now();  // Includes full validation/decoding.
                if (decoded.status == Status::Ok) {
                    if (warmup) ++warmup_successes;
                    else { ++metrics.successes; metrics.latency.push_back(milliseconds(before, completed)); }
                    result = std::move(decoded);
                } else {
                    if (!warmup) {
                        ++metrics.business_errors;
                        ++metrics.status_counts[static_cast<unsigned>(decoded.status)];
                    }
                    std::cerr << "server status=" << static_cast<unsigned>(decoded.status) << ": " << decoded.message << '\n';
                    end_connection = warmup || decoded.status == Status::BadRequest || decoded.status == Status::ShuttingDown;
                }
            } catch (const ProtocolError& error) {
                if (!warmup) ++metrics.protocol_errors;
                std::cerr << error.what() << '\n';
                end_connection = true;
            } catch (const RequestFailure& error) {
                if (!warmup) metrics.error(error.kind);
                std::cerr << "request failed (timeout/connection/protocol category=" << error.kind << ")\n";
                end_connection = true;
            }
            if (!warmup) wall_ms = milliseconds(wall_start, Clock::now());
            if (end_connection) break;  // No reconnect, retry or request replay.
        }
        std::sort(metrics.latency.begin(), metrics.latency.end());
        const unsigned errors = metrics.business_errors + metrics.connection_errors + metrics.protocol_errors + metrics.timeouts;
        const bool success = metrics.successes == options.requests && warmup_successes == options.warmup;
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"status\":" << (success ? 0 : 1) << ",\"requests_completed\":" << metrics.successes
                  << ",\"batch\":" << options.batch
                  << ",\"requests_planned\":" << options.requests << ",\"attempts\":" << metrics.attempts
                  << ",\"successes\":" << metrics.successes << ",\"skipped\":" << options.requests - metrics.attempts
                  << ",\"business_errors\":" << metrics.business_errors << ",\"bad_request\":" << metrics.status_counts[1]
                  << ",\"busy\":" << metrics.status_counts[2] << ",\"shutting_down\":" << metrics.status_counts[3]
                  << ",\"inference_failed\":" << metrics.status_counts[4]
                  << ",\"connection_errors\":" << metrics.connection_errors << ",\"protocol_errors\":" << metrics.protocol_errors
                  << ",\"timeouts\":" << metrics.timeouts << ",\"connect_errors\":" << connect_errors
                  << ",\"connect_timeouts\":" << connect_timeouts << ",\"connect_ms\":" << connect_ms
                  << ",\"warmup_attempts\":" << warmup_attempts << ",\"warmup_successes\":" << warmup_successes
                  << ",\"warmup_failures\":" << warmup_attempts - warmup_successes
                  << ",\"wall_ms\":" << wall_ms << ",\"requests_per_sec\":" << (wall_ms > 0 ? metrics.successes * 1000.0 / wall_ms : 0)
                  << ",\"latency_samples\":" << metrics.latency.size() << ",\"p50_ms\":" << metrics.percentile(0.50)
                  << ",\"p95_ms\":" << metrics.percentile(0.95) << ",\"p99_ms\":" << metrics.percentile(0.99)
                  << ",\"labels\":[";
        for (std::size_t i = 0; i < result.labels.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << result.labels[i];
        }
        std::cout << "],\"logits\":[";
        for (std::size_t i = 0; i < result.logits.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << result.logits[i];
        }
        std::cout << "]}\n";
        if (!success) return errors == metrics.business_errors && metrics.business_errors ? 3 : 1;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
