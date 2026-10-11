#pragma once

#include "miniinfer/net/protocol.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <string>

namespace miniinfer::net {
using RunInference = std::function<MlpResult(MlpInput)>;
struct SessionConfig {
    std::chrono::milliseconds read_timeout{5000};
    std::chrono::milliseconds write_timeout{5000};
};
struct SessionStats {
    std::size_t requests = 0;
    std::size_t responses = 0;
    std::size_t bad_requests = 0;
    std::size_t busy = 0;
    std::size_t inference_failures = 0;
    bool protocol_error = false;
    IoStatus io_status = IoStatus::Ok;
    int error_number = 0;
    std::string diagnostic;
};
// Borrows fd; its owner closes it after this function returns. Each request is
// completed before another is read. Accepted inference runs to completion even
// if stop is set or the client disconnects, then a bounded response is attempted.
SessionStats serve_connection(int fd, const RunInference& run,
                              const SessionConfig& config = {},
                              const std::atomic<bool>* stop = nullptr);
}  // namespace miniinfer::net
