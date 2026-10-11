#pragma once

#include "miniinfer/net/session.hpp"

#include <cstdint>
#include <memory>

namespace miniinfer::net {

struct ServerConfig {
    std::string host = "127.0.0.1";
    std::uint16_t port = 9000;
    std::size_t connection_threads = 4;
    std::size_t waiting_connections = 8;
    std::chrono::milliseconds queue_timeout{1000};
    SessionConfig session;
    // Zero is unlimited. Counts accepted sockets, including capacity rejections.
    std::size_t max_connections = 0;
};

struct ServerStats {
    std::size_t active = 0;  // Includes a connection reserved for a worker.
    std::size_t pending = 0;
    std::size_t accepted = 0;  // Admitted; excludes capacity rejections.
    std::size_t rejected = 0;
    std::size_t expired = 0;
    std::size_t discarded = 0;  // Waiting/reserved connections removed on stop.
    std::size_t completed = 0;  // Sessions that returned, including failed I/O.
    std::size_t clean_connections = 0;
    std::size_t io_failures = 0;
    std::size_t protocol_errors = 0;
    std::size_t session_failures = 0;
    std::size_t requests = 0;
    std::size_t responses = 0;
    std::size_t bad_requests = 0;
    std::size_t busy = 0;
    std::size_t inference_failures = 0;
    std::size_t peak_active = 0;
    std::size_t peak_pending = 0;
    std::size_t accept_errors = 0;
};

// Starts listening and fixed connection threads in the constructor. run and
// drain may execute concurrently; their captured backend must outlive this
// object. drain must close backend submission and finish previously accepted
// work (Runtime::shutdown does both). A stop check cannot be atomic with the
// subsequent run call: backend closure is the final acceptance boundary, and
// late submissions must throw PoolClosed. Callbacks must not call shutdown on
// this server. Its lifetime must not end while external methods are in use.
class TcpServer final {
public:
    TcpServer(ServerConfig config, RunInference run, std::function<void()> drain);
    ~TcpServer();
    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;
    TcpServer(TcpServer&&) = delete;
    TcpServer& operator=(TcpServer&&) = delete;

    std::uint16_t port() const noexcept;
    void request_stop();
    // Concurrent/repeated calls are safe. Stops intake/reads, clears waiting
    // sockets, calls drain without a server lock, then joins every thread.
    // Completed inference still gets a write_timeout-bounded reply attempt.
    void shutdown();
    bool finished() const;
    ServerStats snapshot() const;
    bool wait_for_stats(const std::function<bool(const ServerStats&)>& predicate,
                        Deadline deadline) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace miniinfer::net
