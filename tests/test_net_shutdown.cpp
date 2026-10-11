#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/net/server.hpp"
#include "miniinfer/net/socket.hpp"
#include "miniinfer/runtime.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace {
using namespace miniinfer;
using namespace miniinfer::net;
using namespace std::chrono_literals;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
MlpInput input(std::size_t batch = 1) { return {batch, std::vector<float>(batch * 64, 0.25f)}; }
ServerConfig settings(std::size_t threads = 2, std::size_t waiting = 1) {
    ServerConfig config;
    config.port = 0;
    config.connection_threads = threads;
    config.waiting_connections = waiting;
    config.session = {5s, 300ms};
    return config;
}
UniqueFd connect(const TcpServer& server) {
    return connect_tcp_until(Clock::now() + 2s, "127.0.0.1", server.port());
}
void send(int fd, const Frame& frame) {
    expect(send_frame(fd, frame, Clock::now() + 2s).status == IoStatus::Ok, "request send failed");
}
Response receive(int fd, std::uint32_t batch = 1) {
    Frame frame;
    expect(recv_frame(fd, frame, Clock::now() + 3s).status == IoStatus::Ok, "response receive failed");
    return decode_response(frame, mlp_type, 1, batch);
}
void wait_stats(const TcpServer& server, const std::function<bool(const ServerStats&)>& predicate) {
    expect(server.wait_for_stats(predicate, Clock::now() + 3s), "server stats deadline");
}
void expect_closed(int fd) {
    std::uint8_t byte{};
    const auto io = recv_exact(fd, &byte, 1, Clock::now() + 1s);
    expect(io.status == IoStatus::Eof || io.status == IoStatus::SystemError, "stopped connection remained open");
}

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool open = false;
    std::size_t submitted = 0, entered = 0, completed = 0, drains = 0, shutdown_callers = 0;
    void blocked() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered;
        changed.notify_all();
        expect(changed.wait_for(lock, 5s, [&] { return open; }), "inference gate deadline");
    }
    void accepted() { std::lock_guard<std::mutex> lock(mutex); ++submitted; changed.notify_all(); }
    void computed() { std::lock_guard<std::mutex> lock(mutex); ++completed; changed.notify_all(); }
    void draining() { std::lock_guard<std::mutex> lock(mutex); ++drains; changed.notify_all(); }
    void caller() { std::lock_guard<std::mutex> lock(mutex); ++shutdown_callers; changed.notify_all(); }
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; changed.notify_all(); }
    void wait(std::size_t submits, std::size_t starts, std::size_t drain_count = 0,
              std::size_t callers = 0) {
        std::unique_lock<std::mutex> lock(mutex);
        expect(changed.wait_for(lock, 3s, [&] {
            return submitted >= submits && entered >= starts && drains >= drain_count &&
                   shutdown_callers >= callers;
        }), "gate state deadline");
    }
    void check(std::size_t accepted, std::size_t executed, std::size_t drain_count) {
        std::lock_guard<std::mutex> lock(mutex);
        expect(submitted == accepted && completed == executed && drains == drain_count,
               "accepted work/drain must finish exactly once");
    }
};

// Declared after the server: every failed assertion releases the task gate
// before waiting for shutdown futures or allowing the server to destruct.
struct ShutdownTasks {
    Gate& gate;
    std::vector<std::future<void>> tasks;
    ~ShutdownTasks() {
        gate.release();
        for (auto& task : tasks) if (task.valid()) task.wait();
    }
    void add(TcpServer& server) {
        tasks.emplace_back(std::async(std::launch::async, [&server, this] {
            gate.caller();
            server.shutdown();
        }));
    }
    void finish() {
        for (auto& task : tasks) {
            expect(task.wait_for(3s) == std::future_status::ready, "shutdown completion deadline");
            task.get();
        }
    }
};

void stopped_io_and_waiting(const std::filesystem::path& path) {
    Runtime runtime(MlpModel::load(path));
    std::atomic<std::size_t> calls{0}, drains{0};
    TcpServer server(settings(), [&](MlpInput request) {
        ++calls;
        return runtime.run(std::move(request));
    }, [&] { ++drains; runtime.shutdown(); });
    auto idle = connect(server);
    wait_stats(server, [](const auto& stats) { return stats.active == 1; });
    auto half = connect(server);
    wait_stats(server, [](const auto& stats) { return stats.active == 2; });
    const auto bytes = encode_frame(mlp_request(1, input()));
    expect(send_all(half.get(), bytes.data(), 7, Clock::now() + 1s).status == IoStatus::Ok,
           "partial header send");
    auto pending = connect(server);
    wait_stats(server, [](const auto& stats) { return stats.pending == 1; });
    send(pending.get(), mlp_request(1, input()));
    const auto started = Clock::now();
    server.request_stop();
    server.request_stop();
    server.shutdown();
    server.shutdown();
    expect(Clock::now() - started < 1s, "idle/half-frame stop ignored cancellation");
    const auto stats = server.snapshot();
    expect(stats.active == 0 && stats.pending == 0 && stats.discarded >= 1 &&
           stats.requests == 0 && calls == 0 && drains == 1, "stop processed waiting/partial input");
    expect_closed(idle.get());
    expect_closed(half.get());
    expect_closed(pending.get());
    std::cout << "stop: idle, partial header and queued request exit without submission\n";
}

void accepted_work_and_concurrent_shutdown(const std::filesystem::path& path, bool reset_peer) {
    Gate gate;
    auto model = MlpModel::load(path);
    Workspace workspace;
    ThreadPool pool(1, 1);
    TcpServer server(settings(reset_peer ? 1 : 2), [&](MlpInput request) {
        auto future = pool.submit([&, request = std::move(request)](std::size_t) {
            gate.blocked();
            auto result = CpuExecutor{}.run(model, request, workspace);
            gate.computed();
            return result;
        });
        gate.accepted();
        return future.get();
    }, [&] { gate.draining(); pool.shutdown(); });
    ShutdownTasks shutdowns{gate, {}};
    auto client = connect(server);
    send(client.get(), mlp_request(1, input(8)));
    gate.wait(1, 1);
    UniqueFd second_client;
    const std::size_t accepted = reset_peer ? 1 : 2;
    if (!reset_peer) {
        second_client = connect(server);
        send(second_client.get(), mlp_request(1, input(32)));
        gate.wait(2, 1);
        expect(pool.pending() == 1, "shutdown setup must include one accepted queued task");
    }
    if (reset_peer) {
        const linger immediate_reset{1, 0};
        expect(::setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &immediate_reset,
                            sizeof(immediate_reset)) == 0, "set TCP reset linger");
        client.reset();
    }
    constexpr std::size_t callers = 4;
    for (std::size_t index = 0; index < callers; ++index) shutdowns.add(server);
    gate.wait(accepted, 1, 1, callers);
    for (auto& future : shutdowns.tasks)
        expect(future.wait_for(0ms) == std::future_status::timeout, "shutdown returned before gate computation");
    gate.check(accepted, 0, 1);
    gate.release();
    if (!reset_peer) {
        const auto response = receive(client.get(), 8);
        expect(response.status == Status::Ok && response.logits.size() == 80,
               "accepted work did not get its shutdown response");
        const auto second_response = receive(second_client.get(), 32);
        expect(second_response.status == Status::Ok && second_response.logits.size() == 320,
               "accepted queued work did not get its shutdown response");
    }
    shutdowns.finish();
    gate.check(accepted, accepted, 1);
    server.shutdown();
    const auto stats = server.snapshot();
    expect(stats.completed == accepted && stats.active == 0 && stats.pending == 0, "shutdown left connection work");
    if (reset_peer) expect(stats.io_failures == 1 && stats.responses == 0, "RST response failure accounting");
    else expect(stats.responses == accepted, "shutdown successful reply accounting");
    bool refused = false;
    try { auto late = connect(server); }
    catch (const std::system_error& error) { refused = error.code().value() == ECONNREFUSED; }
    expect(refused, "shutdown listener still accepts connections");
    std::cout << (reset_peer ? "RST during accepted inference: executes once, failed reply reclaimed\n"
                            : "concurrent shutdown: one drain, executing and queued tasks both reply exactly once\n");
}

void submission_shutdown_race(const std::filesystem::path& path) {
    Gate gate;
    Runtime runtime(MlpModel::load(path));
    std::promise<void> backend_closed;
    auto closure = backend_closed.get_future();
    std::atomic<std::size_t> attempts{0}, rejected{0};
    TcpServer server(settings(1), [&](MlpInput request) {
        // serve_connection has passed its stop check and entered the callback,
        // but the Runtime submission has deliberately not happened yet.
        gate.blocked();
        ++attempts;
        try {
            return runtime.run(std::move(request));
        } catch (const PoolClosed&) {
            ++rejected;
            throw;
        }
    }, [&] {
        gate.draining();
        runtime.shutdown();
        backend_closed.set_value();
    });
    ShutdownTasks shutdowns{gate, {}};
    auto client = connect(server);
    send(client.get(), mlp_request(1, input()));
    gate.wait(0, 1);
    shutdowns.add(server);
    expect(closure.wait_for(3s) == std::future_status::ready, "Runtime closure notification deadline");
    closure.get();
    expect(shutdowns.tasks.front().wait_for(0ms) == std::future_status::timeout,
           "shutdown failed to join the paused connection callback");
    gate.release();
    expect(receive(client.get()).status == Status::ShuttingDown,
           "submission crossing backend shutdown must map to wire status 3");
    expect_closed(client.get());
    shutdowns.finish();
    expect(attempts == 1 && rejected == 1, "late submission was not rejected exactly once");
    gate.check(0, 0, 1);
    const auto stats = server.snapshot();
    expect(stats.completed == 1 && stats.responses == 1 && stats.inference_failures == 0,
           "submission/shutdown race response accounting");
    std::cout << "submission race: backend closes between session stop check and Runtime::run, status 3\n";
}

void closed_backend_mapping(const std::filesystem::path& path) {
    Runtime runtime(MlpModel::load(path));
    runtime.shutdown();
    TcpServer server(settings(1), [&](MlpInput request) { return runtime.run(std::move(request)); },
                     [&] { runtime.shutdown(); });
    auto client = connect(server);
    send(client.get(), mlp_request(1, input()));
    expect(receive(client.get()).status == Status::ShuttingDown, "PoolClosed must map to wire status 3");
    expect_closed(client.get());
    wait_stats(server, [](const auto& stats) { return stats.completed == 1; });
    server.shutdown();
    expect(server.snapshot().inference_failures == 0, "closed backend misclassified as inference failure");
    std::cout << "closed backend: PoolClosed maps to SHUTTING_DOWN and closes connection\n";
}

void slow_reader_write_deadline(const std::filesystem::path& path) {
    auto listener = listen_tcp(0);
    UniqueFd client(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    expect(static_cast<bool>(client), "slow reader socket");
    const int small_buffer = 1024;
    // Set before connect so TCP cannot negotiate a large initial receive window.
    expect(::setsockopt(client.get(), SOL_SOCKET, SO_RCVBUF, &small_buffer, sizeof(small_buffer)) == 0,
           "client receive buffer");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(listener.port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(::connect(client.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0,
           "slow reader connect");
    auto connection = accept_tcp(listener.fd.get());
    expect(::setsockopt(connection.get(), SOL_SOCKET, SO_SNDBUF, &small_buffer, sizeof(small_buffer)) == 0,
           "server send buffer");
    const std::vector<std::uint8_t> padding(4 * 1024 * 1024, 0x42);
    // The client never reads. The actual transfer deadline establishes a full
    // peer window and backpressure; no scheduler sleep guesses readiness.
    const auto filled = send_all(connection.get(), padding.data(), padding.size(), Clock::now() + 250ms);
    expect(filled.status == IoStatus::Timeout && filled.transferred > 0,
           "could not establish slow-reader TCP backpressure");
    const std::uint8_t byte = 0;
    const auto blocked = ::send(connection.get(), &byte, 1, MSG_DONTWAIT | MSG_NOSIGNAL);
    expect(blocked < 0 && (errno == EAGAIN || errno == EWOULDBLOCK), "send queue was not full at session start");
    Runtime runtime(MlpModel::load(path));
    std::atomic<std::size_t> executed{0};
    const SessionConfig session{2s, 100ms};
    auto serving = std::async(std::launch::async, [&] {
        return serve_connection(connection.get(), [&](MlpInput request) {
            ++executed;
            return runtime.run(std::move(request));
        }, session);
    });
    send(client.get(), mlp_request(1, input(256)));
    expect(serving.wait_for(3s) == std::future_status::ready, "slow-reader session failed to honor deadline");
    const auto stats = serving.get();
    expect(executed == 1 && stats.requests == 1 && stats.responses == 0 &&
           stats.io_status == IoStatus::Timeout, "blocked response must hit whole-frame write deadline");
    runtime.shutdown();
    std::cout << "slow reader: real TCP queue prefilled to EAGAIN, batch256 reply times out\n";
}

std::size_t resources(const char* directory) {
    return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator(directory),
                                                 std::filesystem::directory_iterator{}));
}
void initialize_thread_runtime() {
    // GCC TSan lazily starts a persistent helper on the first user-thread
    // creation. Measure the leak baseline only after that initialization.
    std::atomic<pid_t> worker_tid{0};
    std::thread([&] { worker_tid.store(static_cast<pid_t>(::syscall(SYS_gettid))); }).join();
    // Join can precede the kernel removing the worker's /proc task entry by a
    // few instructions. Identify that exact worker and await its disappearance
    // instead of counting it as part of the permanent runtime baseline.
    const auto worker_path = std::filesystem::path("/proc/self/task") /
                             std::to_string(worker_tid.load());
    const auto deadline = Clock::now() + 1s;
    while (std::filesystem::exists(worker_path) && Clock::now() < deadline) std::this_thread::yield();
    expect(!std::filesystem::exists(worker_path), "thread runtime warmup worker did not exit");
}
void expect_resources(std::size_t fds, std::size_t threads) {
    expect(resources("/proc/self/fd") == fds, "fd count did not return to baseline");
    // pthread join synchronizes C++ thread completion. Linux can briefly retain
    // a dead task directory while finishing kernel cleanup; bounded yielding
    // observes its disappearance without a fixed scheduling sleep.
    const auto deadline = Clock::now() + 1s;
    while (resources("/proc/self/task") != threads && Clock::now() < deadline) std::this_thread::yield();
    expect(resources("/proc/self/task") == threads, "thread count did not return to baseline");
}
void repeated_lifetimes() {
    const auto fds = resources("/proc/self/fd"), threads = resources("/proc/self/task");
    for (unsigned iteration = 0; iteration < 12; ++iteration) {
        {
            std::atomic<std::size_t> drains{0};
            TcpServer server(settings(2), [](MlpInput) -> MlpResult {
                throw std::runtime_error("echo must not invoke inference");
            }, [&] { ++drains; });
            auto client = connect(server);
            send(client.get(), echo_request(1, {1, 2, 3}));
            Frame frame;
            expect(recv_frame(client.get(), frame, Clock::now() + 1s).status == IoStatus::Ok,
                   "repeated lifecycle echo response");
            expect(decode_response(frame, echo_type, 1).echo == std::vector<std::uint8_t>({1, 2, 3}),
                   "repeated lifecycle echo payload");
            server.request_stop();
            server.shutdown();
            server.shutdown();
            expect(drains == 1, "repeated lifecycle duplicate drain");
        }
        expect_resources(fds, threads);
    }
    std::cout << "12 start/connect/stop lifetimes restore fd and thread counts\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2, "reference model directory required");
        initialize_thread_runtime();
        const auto fds = resources("/proc/self/fd"), threads = resources("/proc/self/task");
        stopped_io_and_waiting(argv[1]);
        accepted_work_and_concurrent_shutdown(argv[1], false);
        accepted_work_and_concurrent_shutdown(argv[1], true);
        submission_shutdown_race(argv[1]);
        closed_backend_mapping(argv[1]);
        slow_reader_write_deadline(argv[1]);
        expect_resources(fds, threads);
        repeated_lifetimes();
        std::cout << "network shutdown tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
