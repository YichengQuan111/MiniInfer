#include "miniinfer/net/session.hpp"
#include "miniinfer/bounded_queue.hpp"

#include <optional>
#include <stdexcept>
#include <utility>

namespace miniinfer::net {
SessionStats serve_connection(int fd, const RunInference& run,
                              const SessionConfig& config, const std::atomic<bool>* stop) {
    if (config.read_timeout.count() <= 0 || config.write_timeout.count() <= 0)
        throw std::invalid_argument("session timeouts must be positive");
    check_wire_platform();
    SessionStats stats;
    bool have_id = false;
    std::uint32_t previous_id = 0;
    const auto stopping = [&] { return stop && stop->load(std::memory_order_relaxed); };
    const auto record_io = [&](const IoResult& io, const char* operation) {
        stats.io_status = io.status;
        stats.error_number = io.error_number;
        if (io.status != IoStatus::Ok && io.status != IoStatus::Eof && io.status != IoStatus::Stopped)
            stats.diagnostic = std::string(operation) + " failed (io_status=" +
                               std::to_string(static_cast<int>(io.status)) +
                               ", errno=" + std::to_string(io.error_number) + ")";
    };
    const auto reply = [&](const Frame& response) {
        // A request already executed must get a bounded response attempt even
        // during shutdown. Do not pass stop to this write operation.
        const auto io = send_frame(fd, response, Clock::now() + config.write_timeout);
        record_io(io, "send frame");
        if (io.status != IoStatus::Ok) return false;
        ++stats.responses;
        return true;
    };
    const auto reject = [&](const Frame& request, const char* reason) {
        ++stats.bad_requests;
        stats.protocol_error = true;
        stats.diagnostic = reason;
        (void)reply(error_response(request, Status::BadRequest, reason));
    };
    for (;;) {
        Frame request;
        IoResult io;
        try {
            io = recv_frame(fd, request, Clock::now() + config.read_timeout, stop);
        } catch (const ProtocolError& error) {
            stats.protocol_error = true;
            stats.diagnostic = error.what();
            return stats;
        }
        record_io(io, "receive frame");
        if (io.status != IoStatus::Ok) return stats;
        ++stats.requests;
        if (request.type != echo_type && request.type != mlp_type) {
            stats.protocol_error = true;
            stats.diagnostic = "expected request direction";
            return stats;
        }
        if (have_id && request.request_id <= previous_id) {
            reject(request, "request_id must strictly increase");
            return stats;
        }
        previous_id = request.request_id;
        have_id = true;
        std::optional<MlpInput> input;
        try {
            if (request.type == echo_type) {
                if (request.body.size() > 1024) throw ProtocolError("echo exceeds 1024 bytes");
            } else {
                input.emplace(decode_mlp_request(request));
            }
        } catch (const ProtocolError& error) {
            reject(request, error.what());
            return stats;
        } catch (const std::invalid_argument&) {
            reject(request, "invalid MLP input");
            return stats;
        }
        if (stopping()) {
            (void)reply(error_response(request, Status::ShuttingDown, "server shutting down"));
            return stats;
        }
        if (request.type == echo_type) {
            if (!reply(echo_response(request))) return stats;
            continue;
        }
        Frame response;
        bool close_after_response = false;
        try {
            // run consumes owned input and returns owned output. Disconnect or
            // stop cannot destroy these while computation is in progress.
            auto result = run(std::move(*input));
            response = mlp_response(request.request_id, result);
        } catch (const QueueFull&) {
            ++stats.busy;
            response = error_response(request, Status::Busy, "inference queue full");
        } catch (const PoolClosed&) {
            response = error_response(request, Status::ShuttingDown, "server shutting down");
            close_after_response = true;
        } catch (const std::exception& error) {
            ++stats.inference_failures;
            stats.diagnostic = std::string("inference failed: ") + error.what();
            response = error_response(request, Status::InferenceFailed, "inference failed");
        } catch (...) {
            ++stats.inference_failures;
            stats.diagnostic = "inference failed with non-standard exception";
            response = error_response(request, Status::InferenceFailed, "inference failed");
        }
        if (!reply(response) || close_after_response) return stats;
    }
}
}  // namespace miniinfer::net
