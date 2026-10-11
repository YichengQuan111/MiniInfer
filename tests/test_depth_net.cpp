#include "miniinfer/depth/net.hpp"
#include "miniinfer/net/server.hpp"
#include "miniinfer/net/socket.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <sys/socket.h>

namespace {
namespace dn = miniinfer::depth::net;
namespace tn = miniinfer::net;
using miniinfer::depth::RgbImage;
using miniinfer::depth::DepthMap;
using namespace std::chrono_literals;
void expect(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
template<class F> void bad_protocol(F&& operation) {
    bool caught = false;
    try { operation(); } catch (const tn::ProtocolError&) { caught = true; }
    expect(caught, "expected protocol rejection");
}
void put32(std::uint8_t* p, std::uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte)
        p[byte] = static_cast<std::uint8_t>(value >> (24 - 8 * byte));
}
RgbImage image(std::size_t width = 2, std::size_t height = 2) {
    std::vector<std::uint8_t> rgb(3 * width * height);
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i % 256);
    return {width, height, std::move(rgb)};
}
DepthMap infer(const RgbImage& input) {
    std::vector<float> result(input.width() * input.height());
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<float>(input.pixels()[3 * i]) + 0.25f;
    return {input.width(), input.height(), std::move(result)};
}
struct Pair {
    tn::UniqueFd client, server;
    Pair() {
        int fds[2];
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0, "socketpair failed");
        client.reset(fds[0]); server.reset(fds[1]);
    }
};
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool open = false;
    std::size_t entered = 0, completed = 0;
    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered; changed.notify_all();
        expect(changed.wait_for(lock, 5s, [&] { return open; }), "gate execution timeout");
        ++completed; changed.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        expect(changed.wait_for(lock, 3s, [&] { return entered != 0; }), "gate admission timeout");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        open = true; changed.notify_all();
    }
};
struct Release { Gate& gate; ~Release() { gate.release(); } };
struct Session {
    Pair pair;
    std::future<tn::SessionStats> task;
    Session(dn::Budget& budget, dn::RunDepth run = infer,
            tn::SessionConfig config = {300ms, 300ms}, const std::atomic<bool>* stop = nullptr,
            int send_buffer = 0) {
        if (send_buffer)
            expect(::setsockopt(pair.server.get(), SOL_SOCKET, SO_SNDBUF,
                               &send_buffer, sizeof(send_buffer)) == 0, "SO_SNDBUF failed");
        task = std::async(std::launch::async,
            [fd = std::move(pair.server), &budget, run = std::move(run), config, stop] {
                return dn::serve_connection(fd.get(), run, budget, config, stop);
            });
    }
    ~Session() { pair.client.reset(); }
    tn::SessionStats finish() {
        pair.client.reset();
        expect(task.wait_for(3s) == std::future_status::ready, "session completion timeout");
        return task.get();
    }
};
void send(int fd, std::uint32_t id, const RgbImage& input) {
    expect(dn::send_request(fd, id, input, tn::Clock::now() + 3s).status == tn::IoStatus::Ok,
           "send request failed");
}
dn::Response receive(int fd, std::uint32_t id, std::size_t w = 2, std::size_t h = 2) {
    dn::Response response;
    expect(dn::recv_response(fd, response, id, w, h, tn::Clock::now() + 3s).status == tn::IoStatus::Ok,
           "receive response failed");
    return response;
}
void raw_send(int fd, const void* bytes, std::size_t count) {
    expect(tn::send_all(fd, bytes, count, tn::Clock::now() + 3s).status == tn::IoStatus::Ok,
           "raw send failed");
}
std::vector<std::uint8_t> prefix(std::uint32_t id, std::uint32_t width = 2,
                                 std::uint32_t height = 2, std::uint32_t body = 32) {
    const auto header = dn::encode_header({dn::request_type, id, body});
    std::vector<std::uint8_t> bytes(header.begin(), header.end());
    bytes.resize(36);
    put32(bytes.data() + 16, width); put32(bytes.data() + 20, height);
    put32(bytes.data() + 24, 3); put32(bytes.data() + 28, 1);
    return bytes;
}
void empty_budget(const dn::Budget& budget) {
    const auto stats = budget.snapshot();
    expect(stats.inflight == 0 && stats.bytes == 0, "request budget leaked");
}

void roundtrip() {
    dn::Budget budget;
    std::atomic<std::size_t> calls{0};
    Session session(budget, [&](const RgbImage& in) { ++calls; return infer(in); });
    const auto input = image();
    const auto golden = infer(input);
    send(session.pair.client.get(), 1, input);
    auto first = receive(session.pair.client.get(), 1);
    expect(first.status == dn::Status::Ok && first.depth->values() == golden.values(), "depth result mismatch");
    send(session.pair.client.get(), 2, input);
    auto second = receive(session.pair.client.get(), 2);
    expect(first.depth->values() == second.depth->values() &&
           first.depth->values().data() != second.depth->values().data(), "result ownership mismatch");
    // A completed frame leaves the next header intact; duplicate ids reject.
    send(session.pair.client.get(), 2, input);
    expect(receive(session.pair.client.get(), 2).status == dn::Status::BadRequest, "duplicate id accepted");
    const auto stats = session.finish();
    expect(stats.requests == 3 && stats.responses == 3 && stats.bad_requests == 1 && calls == 2,
           "roundtrip statistics mismatch");
    empty_budget(budget);
}

void header_and_metadata_errors() {
    const auto valid = dn::encode_header({dn::request_type, 1, 32});
    expect(valid[4] == 0 && valid[5] == 2 && valid[7] == 3 && valid[15] == 32,
           "wire header encoding mismatch");
    for (int mutation = 0; mutation < 6; ++mutation) {
        dn::Budget budget;
        Session session(budget);
        auto header = valid;
        if (mutation == 0) header[0] = 'X';
        if (mutation == 1) header[5] = 1;
        if (mutation == 2) header[7] = 2;
        if (mutation == 3) put32(header.data() + 12, 0xffffffffu);
        if (mutation == 4) put32(header.data() + 12, 19);
        if (mutation == 5) header[6] = 0x80;
        raw_send(session.pair.client.get(), header.data(), header.size());
        const auto stats = session.finish();
        expect(stats.protocol_error && stats.bad_requests == 1, "malformed header accepted");
        expect(budget.snapshot().peak_bytes == 0, "malformed header allocated budget");
    }
    for (int mutation = 0; mutation < 8; ++mutation) {
        dn::Budget budget;
        Session session(budget);
        auto bytes = prefix(7);
        if (mutation == 0) put32(bytes.data() + 16, 0);
        if (mutation == 1) put32(bytes.data() + 16, 4097);
        if (mutation == 2) { put32(bytes.data() + 16, 4096); put32(bytes.data() + 20, 4096); }
        if (mutation == 3) { put32(bytes.data() + 16, 0xffffffffu); put32(bytes.data() + 20, 0xffffffffu); }
        if (mutation == 4) put32(bytes.data() + 24, 4);
        if (mutation == 5) put32(bytes.data() + 28, 0);
        if (mutation == 6) put32(bytes.data() + 32, 1);
        if (mutation == 7) put32(bytes.data() + 12, 33);
        raw_send(session.pair.client.get(), bytes.data(), bytes.size());
        expect(receive(session.pair.client.get(), 7).status == dn::Status::BadRequest, "malformed metadata accepted");
        const auto stats = session.finish();
        expect(stats.protocol_error && stats.bad_requests == 1, "metadata rejection not recorded");
        expect(budget.snapshot().peak_bytes == 0, "malformed metadata reserved budget");
    }
}

void maximum_image_and_budget() {
    dn::Budget budget;
    {
        std::vector<dn::Budget::Lease> leases;
        for (int i = 0; i < 4; ++i) {
            auto lease = budget.try_acquire(miniinfer::depth::max_pixels);
            expect(lease.has_value(), "default budget cannot admit four maximum images");
            leases.push_back(std::move(*lease));
        }
        expect(!budget.try_acquire(1) && budget.snapshot().bytes <= 128 * 1024 * 1024,
               "default maximum image budget exceeded");
    }
    empty_budget(budget);
    Session session(budget, infer, {3s, 3s});
    auto input = image(4096, 512);
    send(session.pair.client.get(), 1, input);
    const auto response = receive(session.pair.client.get(), 1, 4096, 512);
    expect(response.status == dn::Status::Ok && response.depth->values().size() == miniinfer::depth::max_pixels,
           "maximum legal image rejected");
    for (std::size_t i = 0; i < response.depth->values().size(); ++i)
        expect(response.depth->values()[i] == static_cast<float>(input.pixels()[3 * i]) + .25f,
               "maximum image output corrupted");
    session.finish(); empty_budget(budget);
}

void short_frames_and_read_deadlines() {
    for (const std::size_t bytes : {std::size_t{0}, std::size_t{8}, std::size_t{16},
                                    std::size_t{26}, std::size_t{36}, std::size_t{40}}) {
        dn::Budget budget;
        Session session(budget);
        auto wire = prefix(1); wire.resize(48);
        raw_send(session.pair.client.get(), wire.data(), bytes);
        const auto stats = session.finish();
        expect(stats.io_status == (bytes == 0 ? tn::IoStatus::Eof : tn::IoStatus::Truncated),
               "partial frame EOF classification mismatch");
        empty_budget(budget);
    }
    dn::Budget budget;
    Session session(budget, infer, {70ms, 70ms});
    const auto bytes = prefix(1);
    raw_send(session.pair.client.get(), bytes.data(), bytes.size());
    expect(session.task.wait_for(2s) == std::future_status::ready, "slow request did not expire");
    expect(session.finish().io_status == tn::IoStatus::Timeout, "slow payload timeout mismatch");
    expect(budget.snapshot().peak_inflight == 1, "partial payload never charged");
    empty_budget(budget);
}

void budget_admission_and_recovery() {
    {
        dn::Budget budget(2, 1024 * 1024 + 14 * 4);
        auto lease = budget.try_acquire(4);
        expect(lease && !budget.try_acquire(1), "byte budget was not enforced");
        auto moved = std::move(*lease); lease.reset();
        expect(budget.snapshot().inflight == 1, "moving lease released charge");
        auto replacement = dn::Budget(1).try_acquire(1);
        moved = std::move(*replacement);
        empty_budget(budget);
    }
    dn::Budget budget(1);
    Gate gate;
    Session first(budget, [&](const RgbImage& input) { gate.run(); return infer(input); });
    Release release{gate};
    send(first.pair.client.get(), 1, image()); gate.wait();
    expect(budget.snapshot().inflight == 1, "accepted execution uncharged");
    Session second(budget);
    const auto bytes = prefix(55); // No RGB bytes: Busy must not wait for them.
    raw_send(second.pair.client.get(), bytes.data(), bytes.size());
    expect(receive(second.pair.client.get(), 55).status == dn::Status::Busy, "budget rejection not Busy");
    expect(second.task.wait_for(1s) == std::future_status::ready, "Busy session reused unread payload");
    expect(second.finish().busy == 1, "Busy not counted");
    gate.release();
    expect(receive(first.pair.client.get(), 1).status == dn::Status::Ok, "admitted inference failed");
    first.finish(); empty_budget(budget);
    Session recovered(budget);
    send(recovered.pair.client.get(), 1, image());
    expect(receive(recovered.pair.client.get(), 1).status == dn::Status::Ok, "budget did not recover");
    recovered.finish(); empty_budget(budget);
    const auto stats = budget.snapshot();
    expect(stats.peak_inflight == 1 && stats.rejected == 1, "budget count statistics mismatch");
}

void sdk_rejections_and_failures() {
    for (const auto expected : {dn::Status::Busy, dn::Status::ShuttingDown, dn::Status::InferenceFailed}) {
        dn::Budget budget;
        std::size_t calls = 0;
        Session session(budget, [&](const RgbImage& input) {
            if (++calls == 1) throw dn::DepthNetError(expected, "classified admission error");
            return infer(input);
        });
        send(session.pair.client.get(), 1, image());
        const auto response = receive(session.pair.client.get(), 1);
        expect(response.status == expected && !response.depth, "SDK failure mapping mismatch");
        if (expected != dn::Status::ShuttingDown) {
            send(session.pair.client.get(), 2, image());
            expect(receive(session.pair.client.get(), 2).status == dn::Status::Ok, "SDK rejection lost frame boundary");
        }
        const auto stats = session.finish();
        expect(stats.busy == (expected == dn::Status::Busy ? 1u : 0u) &&
               stats.inference_failures == (expected == dn::Status::InferenceFailed ? 1u : 0u),
               "SDK failure statistics mismatch");
        empty_budget(budget);
    }
    for (int kind = 0; kind < 3; ++kind) {
        dn::Budget budget;
        Session session(budget, [kind](const RgbImage&) -> DepthMap {
            if (kind == 0) throw std::runtime_error("backend failed");
            if (kind == 1) throw 42;
            return DepthMap(1, 1, {1.f});
        });
        send(session.pair.client.get(), 1, image());
        expect(receive(session.pair.client.get(), 1).status == dn::Status::InferenceFailed,
               "backend exception or bad shape accepted");
        expect(session.finish().inference_failures == 1, "backend exception not counted");
        empty_budget(budget);
    }
}

void disconnect_stop_and_write_deadlines() {
    {
        dn::Budget budget;
        Gate gate;
        Session session(budget, [&](const RgbImage& input) { gate.run(); return infer(input); });
        Release release{gate};
        send(session.pair.client.get(), 1, image()); gate.wait();
        session.pair.client.reset();
        gate.release();
        const auto stats = session.finish();
        expect(gate.completed == 1 && stats.requests == 1 && stats.responses == 0,
               "disconnect cancelled accepted execution");
        empty_budget(budget);
    }
    {
        dn::Budget budget;
        Gate gate;
        std::atomic<bool> stop{false};
        Session session(budget, [&](const RgbImage& input) { gate.run(); return infer(input); },
                        {3s, 200ms}, &stop);
        Release release{gate};
        send(session.pair.client.get(), 1, image()); gate.wait();
        stop.store(true); gate.release();
        expect(receive(session.pair.client.get(), 1).status == dn::Status::Ok,
               "stop suppressed accepted result write");
        expect(session.finish().io_status == tn::IoStatus::Stopped && gate.completed == 1,
               "stop did not finish execution and halt reads");
        empty_budget(budget);
    }
    {
        dn::Budget budget;
        Session session(budget, infer, {3s, 70ms}, nullptr, 1024);
        send(session.pair.client.get(), 1, image(512, 512));
        expect(session.task.wait_for(2s) == std::future_status::ready, "slow response did not expire");
        const auto stats = session.finish();
        expect(stats.io_status == tn::IoStatus::Timeout && stats.responses == 0,
               "slow reader write timeout mismatch");
        empty_budget(budget);
    }
    {
        dn::Budget budget;
        std::atomic<bool> stop{true};
        Session session(budget, infer, {3s, 3s}, &stop);
        expect(session.finish().io_status == tn::IoStatus::Stopped, "idle stop failed");
        empty_budget(budget);
    }
}

void client_wire_validation() {
    // Capture deterministic short writes. Request metadata stays big endian;
    // RGB bytes are contiguous and no framing layer changes the raw image.
    std::vector<std::uint8_t> request;
    tn::IoHooks sender;
    sender.send = [&](int, const void* p, std::size_t count, int) -> ssize_t {
        count = std::min<std::size_t>(count, 3);
        const auto* bytes = static_cast<const std::uint8_t*>(p);
        request.insert(request.end(), bytes, bytes + count);
        return static_cast<ssize_t>(count);
    };
    expect(dn::send_request(-1, 0x01020304, image(), tn::Clock::now() + 3s, nullptr, &sender).transferred == 48,
           "short write frame count mismatch");
    expect(request[8] == 1 && request[9] == 2 && request[10] == 3 && request[11] == 4 &&
           request[19] == 2 && request[27] == 3 && request[31] == 1 && request[39] == 3,
           "request wire endian/layout mismatch");
    std::vector<std::uint8_t> valid;
    const auto header = dn::encode_header({dn::response_type, 9, 36});
    valid.assign(header.begin(), header.end()); valid.resize(52);
    put32(valid.data() + 20, 2); put32(valid.data() + 24, 2); put32(valid.data() + 28, 1);
    // Wire little-endian 1, -2, 0.25, 100.
    const std::uint32_t bits[] = {0x3f800000u, 0xc0000000u, 0x3e800000u, 0x42c80000u};
    for (std::size_t i = 0; i < 4; ++i)
        for (unsigned byte = 0; byte < 4; ++byte)
            valid[36 + 4 * i + byte] = static_cast<std::uint8_t>(bits[i] >> (8 * byte));
    const auto decode = [](const std::vector<std::uint8_t>& wire, dn::Response& response) {
        std::size_t offset = 0;
        tn::IoHooks receiver;
        receiver.recv = [&](int, void* p, std::size_t size, int) -> ssize_t {
            const auto count = std::min({size, wire.size() - offset, std::size_t{3}});
            std::memcpy(p, wire.data() + offset, count); offset += count;
            return static_cast<ssize_t>(count);
        };
        return dn::recv_response(-1, response, 9, 2, 2, tn::Clock::now() + 3s, nullptr, &receiver);
    };
    dn::Response response;
    expect(decode(valid, response).transferred == 52 && response.depth->values() == std::vector<float>({1,-2,.25f,100}),
           "FP32 endian/result mismatch");
    for (int mutation = 0; mutation < 9; ++mutation) {
        auto bytes = valid;
        if (mutation == 0) bytes[5] = 1;
        if (mutation == 1) bytes[6] = 0;
        if (mutation == 2) bytes[11] = 10;
        if (mutation == 3) put32(bytes.data() + 16, 99);
        if (mutation == 4) put32(bytes.data() + 20, 0xffffffffu);
        if (mutation == 5) put32(bytes.data() + 28, 0);
        if (mutation == 6) put32(bytes.data() + 32, 1);
        if (mutation == 7) put32(bytes.data() + 12, 40);
        if (mutation == 8) { bytes[36] = 0; bytes[37] = 0; bytes[38] = 0x80; bytes[39] = 0x7f; }
        bad_protocol([&] { decode(bytes, response); });
        expect(!response.depth, "invalid response retained stale output");
    }
    auto partial = valid; partial.resize(40);
    expect(decode(partial, response).status == tn::IoStatus::Truncated && !response.depth,
           "partial response classification mismatch");
    for (int mutation = 0; mutation < 4; ++mutation) {
        auto bytes = valid;
        bytes.resize(36); put32(bytes.data() + 12, 20); put32(bytes.data() + 16, 2);
        std::fill(bytes.begin() + 20, bytes.end(), 0);
        if (mutation == 0) {
            expect(decode(bytes, response).status == tn::IoStatus::Ok && response.status == dn::Status::Busy &&
                   !response.depth, "fixed error response failed");
        } else {
            if (mutation == 1) put32(bytes.data() + 20, 2);
            if (mutation == 2) put32(bytes.data() + 28, 1);
            if (mutation == 3) { put32(bytes.data() + 12, 24); bytes.resize(40); }
            bad_protocol([&] { decode(bytes, response); });
        }
    }
}

void generic_server_shutdown() {
    dn::Budget budget(1);
    Gate gate;
    std::atomic<bool> drained{false};
    tn::ServerConfig config; config.port = 0; config.connection_threads = 1;
    config.session = {3s, 300ms};
    const dn::RunDepth run = [&](const RgbImage& input) { gate.run(); return infer(input); };
    tn::TcpServer server(config,
        tn::RunSession([&](int fd, const tn::SessionConfig& session, const std::atomic<bool>* stop) {
            return dn::serve_connection(fd, run, budget, session, stop);
        }), [&] { drained.store(true); gate.release(); });
    Release release{gate};
    auto client = tn::connect_tcp_until(tn::Clock::now() + 3s, "127.0.0.1", server.port());
    send(client.get(), 1, image()); gate.wait();
    expect(budget.snapshot().inflight == 1, "generic server failed budget admission");
    auto shutdown = std::async(std::launch::async, [&] { server.shutdown(); });
    expect(receive(client.get(), 1).status == dn::Status::Ok, "generic shutdown lost accepted result");
    expect(shutdown.wait_for(3s) == std::future_status::ready, "generic server shutdown deadline");
    shutdown.get(); server.shutdown();
    expect(drained && gate.completed == 1 && server.finished(), "generic server did not drain");
    const auto stats = server.snapshot();
    expect(stats.completed == 1 && stats.requests == 1 && stats.responses == 1 &&
           stats.active == 0 && stats.pending == 0 && stats.session_failures == 0,
           "generic server accounting mismatch");
    empty_budget(budget);
    bool empty_callback_rejected = false;
    try { tn::TcpServer invalid(config, tn::RunSession{}, [] {}); }
    catch (const std::invalid_argument&) { empty_callback_rejected = true; }
    expect(empty_callback_rejected, "generic empty callback accepted");
}
}  // namespace
int main() {
    try {
        roundtrip(); header_and_metadata_errors(); maximum_image_and_budget(); short_frames_and_read_deadlines();
        budget_admission_and_recovery(); sdk_rejections_and_failures();
        disconnect_stop_and_write_deadlines(); client_wire_validation(); generic_server_shutdown();
        std::cout << "depth network contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "depth network contract failed: " << error.what() << '\n';
        return 1;
    }
}
