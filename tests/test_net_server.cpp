#include "miniinfer/net/server.hpp"
#include "miniinfer/net/socket.hpp"
#include "miniinfer/runtime.hpp"
#include "miniinfer/data_io.hpp"
#include "miniinfer/cpu_executor.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <sys/socket.h>

namespace {
using namespace miniinfer;
using namespace miniinfer::net;
using namespace std::chrono_literals;
void expect(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool open = false;
    std::size_t entered = 0, submitted = 0, executed = 0;
    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered; changed.notify_all();
        if (!changed.wait_for(lock, 8s, [&] { return open; })) throw std::runtime_error("gate deadline");
        ++executed;
    }
    void accepted() { std::lock_guard<std::mutex> lock(mutex); ++submitted; changed.notify_all(); }
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; changed.notify_all(); }
    void wait(std::size_t starts, std::size_t accepted) {
        std::unique_lock<std::mutex> lock(mutex);
        expect(changed.wait_for(lock, 5s, [&] { return entered >= starts && submitted >= accepted; }), "gate state deadline");
    }
};
struct ReleaseGate { Gate& gate; ~ReleaseGate() { gate.release(); } };
struct Fixture {
    Buffer inputs, logits;
    explicit Fixture(const std::filesystem::path& path)
        : inputs(load_f32_file(path / "inputs.f32", 64 * 64)),
          logits(load_f32_file(path / "logits.f32", 64 * 10)) {}
    MlpInput input(std::size_t batch, std::size_t start = 0) const {
        std::vector<float> values(batch * 64);
        for (std::size_t row = 0; row < batch; ++row)
            std::copy_n(inputs.data() + ((row + start) % 64) * 64, 64, values.data() + row * 64);
        return {batch, std::move(values)};
    }
    void check(const Response& response, std::size_t batch, std::size_t start = 0) const {
        expect(response.status == Status::Ok && response.batch == batch, "response status/batch");
        for (std::size_t row = 0; row < batch; ++row) {
            const auto* reference = logits.data() + ((row + start) % 64) * 10;
            const auto label = static_cast<std::uint32_t>(std::max_element(reference, reference + 10) - reference);
            expect(response.labels.at(row) == label, "concurrent label mismatch");
            for (std::size_t col = 0; col < 10; ++col)
                expect(std::abs(response.logits.at(row * 10 + col) - reference[col]) <= 1e-4f + 1e-4f * std::abs(reference[col]),
                       "concurrent logits mismatch");
        }
    }
};
UniqueFd connect(const TcpServer& server) {
    return connect_tcp_until(Clock::now() + 3s, "127.0.0.1", server.port());
}
void send(int fd, const Frame& request) {
    expect(send_frame(fd, request, Clock::now() + 3s).status == IoStatus::Ok, "TCP send deadline/error");
}
Response receive(int fd, std::uint32_t id, std::uint32_t batch) {
    Frame frame;
    expect(recv_frame(fd, frame, Clock::now() + 5s).status == IoStatus::Ok, "TCP receive deadline/error");
    return decode_response(frame, mlp_type, id, batch);
}
void wait_stats(const TcpServer& server, std::function<bool(const ServerStats&)> predicate) {
    expect(server.wait_for_stats(predicate, Clock::now() + 5s), "server stats deadline");
}
ServerConfig config(std::size_t workers = 4, std::size_t pending = 8) {
    ServerConfig c; c.port = 0; c.connection_threads = workers; c.waiting_connections = pending;
    c.session = {5s, 200ms};
    return c;
}
std::size_t count(const char* path) {
    return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator{}));
}

void mixed_clients(const std::filesystem::path& path, const Fixture& fixture) {
    Runtime runtime(MlpModel::load(path), {2, 64, 256});
    TcpServer server(config(), [&](MlpInput input) { return runtime.run(std::move(input)); }, [&] { runtime.shutdown(); });
    std::vector<std::future<void>> tasks;
    for (unsigned client = 0; client < 4; ++client) {
        tasks.emplace_back(std::async(std::launch::async, [&, client] {
            auto fd = connect(server);
            constexpr std::size_t batches[]{1, 8, 32, 64, 256};
            for (unsigned i = 0; i < 10; ++i) {
                const auto batch = batches[(i + client) % 5];
                const auto start = client * 9 + i;
                send(fd.get(), mlp_request(i + 1, fixture.input(batch, start)));
                fixture.check(receive(fd.get(), i + 1, batch), batch, start);
            }
        }));
    }
    for (auto& task : tasks) task.get();
    wait_stats(server, [](const auto& s) { return s.completed == 4; });
    server.shutdown();
    const auto s = server.snapshot();
    expect(s.requests == 40 && s.responses == 40 && s.peak_active <= 4 && s.peak_pending <= 8,
           "mixed client counts/bounds");
    std::cout << "mixed clients: 40 reference requests across 4 reused connections\n";
}

void busy_recovery(const std::filesystem::path& path, const Fixture& fixture) {
    Gate gate;
    auto model = MlpModel::load(path);
    Workspace workspace;
    ThreadPool pool(1, 1);
    TcpServer server(config(3, 1), [&](MlpInput input) {
        auto future = pool.submit([&, input = std::move(input)](std::size_t) {
            gate.run();
            return CpuExecutor{}.run(model, input, workspace);
        });
        gate.accepted();
        return future.get();
    }, [&] { pool.shutdown(); });
    ReleaseGate release{gate};  // Releases before server/pool destruction on all assertion failures.
    auto first = connect(server), second = connect(server), third = connect(server);
    send(first.get(), mlp_request(1, fixture.input(1)));
    gate.wait(1, 1);
    send(second.get(), mlp_request(1, fixture.input(8, 4)));
    gate.wait(1, 2);
    expect(pool.pending() == 1, "gate must hold exactly one waiting inference");
    send(third.get(), mlp_request(1, fixture.input(32)));
    expect(receive(third.get(), 1, 32).status == Status::Busy, "full Runtime queue must map to BUSY");
    gate.release();
    fixture.check(receive(first.get(), 1, 1), 1);
    fixture.check(receive(second.get(), 1, 8), 8, 4);
    send(third.get(), mlp_request(2, fixture.input(32, 9)));
    fixture.check(receive(third.get(), 2, 32), 32, 9);
    server.shutdown();
    expect(gate.submitted == 3 && gate.executed == 3, "accepted inference executes exactly once");
    expect(server.snapshot().busy == 1, "BUSY accounting");
    std::cout << "gate overload: workers=1 queue=1 gives BUSY, same connection recovers\n";
}

void bounded_connections(const std::filesystem::path& path) {
    Runtime runtime(MlpModel::load(path));
    auto c = config(1, 1); c.queue_timeout = 20ms;
    TcpServer server(c, [&](MlpInput i) { return runtime.run(std::move(i)); }, [&] { runtime.shutdown(); });
    auto active = connect(server);
    wait_stats(server, [](const auto& s) { return s.active == 1; });
    auto queued = connect(server);
    wait_stats(server, [](const auto& s) { return s.pending == 1; });
    auto rejected = connect(server);
    wait_stats(server, [](const auto& s) { return s.rejected == 1; });
    Frame unused;
    expect(recv_frame(rejected.get(), unused, Clock::now() + 1s).status == IoStatus::Eof, "connection capacity closes immediately");
    // Exercise an actual client read deadline beyond the known queue residence
    // time. This is the timer being tested, not a guessed scheduling sleep.
    expect(recv_frame(active.get(), unused, Clock::now() + 40ms).status == IoStatus::Timeout, "idle client deadline");
    active.reset();
    wait_stats(server, [](const auto& s) { return s.expired == 1; });
    expect(recv_frame(queued.get(), unused, Clock::now() + 1s).status == IoStatus::Eof, "expired queued connection closes");
    server.shutdown();
    const auto s = server.snapshot();
    expect(s.peak_active == 1 && s.peak_pending == 1 && s.active == 0 && s.pending == 0, "connection resource bounds");
    std::cout << "connection bounds: active=1 waiting=1, extra refused and queued deadline enforced\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2, "reference directory required");
        Fixture fixture(argv[1]);
        const auto fds = count("/proc/self/fd");
        mixed_clients(argv[1], fixture);
        busy_recovery(argv[1], fixture);
        bounded_connections(argv[1]);
        expect(count("/proc/self/fd") == fds, "server fd leak");
        std::cout << "network concurrency tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
