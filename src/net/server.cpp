#include "miniinfer/net/server.hpp"
#include "miniinfer/net/socket.hpp"

#include <fcntl.h>
#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace miniinfer::net {
struct TcpServer::Impl {
    struct Connection {
        UniqueFd fd;
        Deadline expires;
    };
    struct Slot {
        bool occupied = false;
        std::optional<Connection> connection;
    };

    ServerConfig config;
    RunSession run;
    std::function<void()> drain;
    TcpListener listener;
    const std::uint16_t bound_port;
    std::atomic<bool> stop{false};
    mutable std::mutex mutex;
    mutable std::condition_variable changed;
    ServerStats stats;
    std::vector<Slot> slots;
    std::deque<Connection> waiting;
    bool accept_done = false;
    std::vector<std::thread> workers;
    std::thread accept_thread;
    std::mutex shutdown_mutex;
    std::condition_variable shutdown_changed;
    bool shutdown_running = false;
    bool shutdown_done = false;
    std::exception_ptr shutdown_error;

    static TcpListener bind(const ServerConfig& config, const RunSession& run,
                            const std::function<void()>& drain) {
        if (!run || !drain) throw std::invalid_argument("server callbacks must be provided");
        if (config.connection_threads == 0 || config.queue_timeout.count() <= 0 ||
            config.session.read_timeout.count() <= 0 || config.session.write_timeout.count() <= 0)
            throw std::invalid_argument("server thread count and timeouts must be positive");
        check_wire_platform();
        auto result = listen_tcp(config.port, config.host);
        const int flags = ::fcntl(result.fd.get(), F_GETFL, 0);
        if (flags < 0 || ::fcntl(result.fd.get(), F_SETFL, flags | O_NONBLOCK) < 0)
            throw std::system_error(errno, std::generic_category(), "fcntl(listener)");
        return result;
    }

    Impl(ServerConfig settings, RunSession execute, std::function<void()> finish)
        : config(std::move(settings)), run(std::move(execute)), drain(std::move(finish)),
          listener(bind(config, run, drain)), bound_port(listener.port),
          slots(config.connection_threads) {
        try {
            workers.reserve(config.connection_threads);
            for (std::size_t index = 0; index < slots.size(); ++index)
                workers.emplace_back([this, index] { connection_loop(index); });
            accept_thread = std::thread([this] { accept_loop(); });
        } catch (...) {
            request_stop();
            for (auto& worker : workers) if (worker.joinable()) worker.join();
            throw;
        }
    }

    void request_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop.store(true, std::memory_order_relaxed);
            // These descriptors have never been moved to a connection thread.
            // Active descriptors are closed exclusively by their owning worker.
            stats.discarded += waiting.size();
            waiting.clear();
            stats.pending = 0;
        }
        changed.notify_all();
    }

    void accept_loop() noexcept {
        std::size_t seen = 0;
        try {
            while (!stop.load(std::memory_order_relaxed) &&
                   (config.max_connections == 0 || seen < config.max_connections)) {
                pollfd event{listener.fd.get(), POLLIN, 0};
                const int ready = ::poll(&event, 1, 20);
                if (ready < 0) {
                    if (errno == EINTR) continue;
                    throw std::system_error(errno, std::generic_category(), "poll(listener)");
                }
                if (ready == 0) continue;
                if (event.revents & (POLLERR | POLLHUP | POLLNVAL))
                    throw std::runtime_error("listener poll failed");
                // Accept one socket per iteration so neither incoming floods
                // nor a full queue can prevent the stop check.
                UniqueFd fd(::accept4(listener.fd.get(), nullptr, nullptr,
                                      SOCK_CLOEXEC | SOCK_NONBLOCK));
                if (!fd) {
                    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                        errno == ECONNABORTED) continue;
                    throw std::system_error(errno, std::generic_category(), "accept4");
                }
                ++seen;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (stop.load(std::memory_order_relaxed)) break;
                    const auto idle = std::find_if(slots.begin(), slots.end(),
                                                  [](const Slot& slot) { return !slot.occupied; });
                    Connection connection{std::move(fd), Clock::now() + config.queue_timeout};
                    if (idle != slots.end()) {
                        idle->connection.emplace(std::move(connection));
                        idle->occupied = true;
                        ++stats.active;
                        ++stats.accepted;
                        stats.peak_active = std::max(stats.peak_active, stats.active);
                    } else if (waiting.size() < config.waiting_connections) {
                        waiting.push_back(std::move(connection));
                        ++stats.pending;
                        ++stats.accepted;
                        stats.peak_pending = std::max(stats.peak_pending, stats.pending);
                    } else {
                        ++stats.rejected;
                        // connection is closed here, before the next accept.
                    }
                }
                changed.notify_all();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++stats.accept_errors;
            }
            request_stop();
        }
        // Only the accept thread closes its listener; no cross-thread fd reuse.
        listener.fd.reset();
        {
            std::lock_guard<std::mutex> lock(mutex);
            accept_done = true;
        }
        changed.notify_all();
    }

    void accumulate(const SessionStats& session) {
        ++stats.completed;
        stats.requests += session.requests;
        stats.responses += session.responses;
        stats.bad_requests += session.bad_requests;
        stats.busy += session.busy;
        stats.inference_failures += session.inference_failures;
        if (session.protocol_error) ++stats.protocol_errors;
        const bool failed_io = session.io_status == IoStatus::Timeout ||
                               session.io_status == IoStatus::Truncated ||
                               session.io_status == IoStatus::SystemError;
        if (failed_io) ++stats.io_failures;
        if (!failed_io && !session.protocol_error) ++stats.clean_connections;
    }

    void connection_loop(std::size_t index) noexcept {
        for (;;) {
            std::optional<Connection> connection;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] {
                    return slots[index].connection.has_value() || accept_done ||
                           stop.load(std::memory_order_relaxed);
                });
                if (!slots[index].connection) return;
                connection.emplace(std::move(*slots[index].connection));
                slots[index].connection.reset();
            }
            SessionStats session;
            bool ran = false;
            bool failed = false;
            if (!stop.load(std::memory_order_relaxed)) {
                try {
                    ran = true;
                    session = run(connection->fd.get(), config.session, &stop);
                } catch (...) {
                    failed = true;
                }
            }
            // Release ownership before advertising the worker as available.
            connection.reset();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (failed) {
                    ++stats.completed;
                    ++stats.session_failures;
                } else if (ran) {
                    accumulate(session);
                } else {
                    ++stats.discarded;
                }
                slots[index].occupied = false;
                --stats.active;
                // Waiting timeout is measured from admission, and tested before
                // moving a queued descriptor to this connection thread.
                while (!waiting.empty()) {
                    if (stop.load(std::memory_order_relaxed)) break;
                    if (Clock::now() >= waiting.front().expires) {
                        waiting.pop_front();
                        --stats.pending;
                        ++stats.expired;
                        continue;
                    }
                    slots[index].connection.emplace(std::move(waiting.front()));
                    waiting.pop_front();
                    --stats.pending;
                    slots[index].occupied = true;
                    ++stats.active;
                    break;
                }
            }
            changed.notify_all();
        }
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(shutdown_mutex);
            if (shutdown_running) shutdown_changed.wait(lock, [&] { return shutdown_done; });
            if (shutdown_done) {
                if (shutdown_error) std::rethrow_exception(shutdown_error);
                return;
            }
            shutdown_running = true;
        }
        request_stop();
        if (accept_thread.joinable()) accept_thread.join();
        std::exception_ptr error;
        // No server lock is held around backend shutdown or thread joins.
        try { drain(); } catch (...) { error = std::current_exception(); }
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        {
            std::lock_guard<std::mutex> lock(shutdown_mutex);
            shutdown_error = error;
            shutdown_done = true;
        }
        shutdown_changed.notify_all();
        if (error) std::rethrow_exception(error);
    }
};

TcpServer::TcpServer(ServerConfig config, RunInference run, std::function<void()> drain)
    : TcpServer(std::move(config),
                run ? RunSession([execute = std::move(run)](int fd, const SessionConfig& session,
                                                           const std::atomic<bool>* stop) {
                    return serve_connection(fd, execute, session, stop);
                }) : RunSession{}, std::move(drain)) {}

TcpServer::TcpServer(ServerConfig config, RunSession run, std::function<void()> drain)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(run), std::move(drain))) {}

TcpServer::~TcpServer() {
    try { shutdown(); } catch (...) { /* All threads have already been joined. */ }
}

std::uint16_t TcpServer::port() const noexcept { return impl_->bound_port; }
void TcpServer::request_stop() { impl_->request_stop(); }
void TcpServer::shutdown() { impl_->shutdown(); }

bool TcpServer::finished() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->accept_done && impl_->stats.active == 0 && impl_->stats.pending == 0;
}

ServerStats TcpServer::snapshot() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->stats;
}

bool TcpServer::wait_for_stats(const std::function<bool(const ServerStats&)>& predicate,
                             Deadline deadline) const {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    return impl_->changed.wait_until(lock, deadline, [&] { return predicate(impl_->stats); });
}

}  // namespace miniinfer::net
