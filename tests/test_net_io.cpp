#include "miniinfer/net/io.hpp"
#include "miniinfer/net/socket.hpp"

#include <fcntl.h>
#include <sys/socket.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
using namespace miniinfer::net;
using namespace std::chrono_literals;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void status(IoResult result, IoStatus wanted, std::size_t count, const char* message) {
    expect(result.status == wanted && result.transferred == count, message);
}
struct Step { ssize_t count; int error = 0; int advance_ms = 0; };
struct Script {
    Deadline time{};
    std::vector<Step> steps;
    std::size_t next = 0;
    std::size_t offset = 0;
    int polls = 0;
    std::string bytes = "abcdef";
    std::string written;
    IoHooks hooks;
    Script() {
        hooks.now = [&] { return time; };
        hooks.recv = [&](int, void* destination, std::size_t capacity, int flags) {
            expect(flags & MSG_DONTWAIT, "recv must be nonblocking per call");
            const auto step = take();
            if (step.count > 0) {
                expect(static_cast<std::size_t>(step.count) <= capacity, "script recv exceeds capacity");
                expect(offset + step.count <= bytes.size(), "script recv exceeds payload");
                std::memcpy(destination, bytes.data() + offset, step.count);
                offset += static_cast<std::size_t>(step.count);
            }
            return step.count;
        };
        hooks.send = [&](int, const void* source, std::size_t capacity, int flags) {
            expect((flags & (MSG_DONTWAIT | MSG_NOSIGNAL)) == (MSG_DONTWAIT | MSG_NOSIGNAL),
                   "send must avoid blocking and SIGPIPE");
            const auto step = take();
            if (step.count > 0) {
                expect(static_cast<std::size_t>(step.count) <= capacity, "script send exceeds capacity");
                written.append(static_cast<const char*>(source), step.count);
            }
            return step.count;
        };
        hooks.poll = [&](pollfd* pending, nfds_t count, int wait_ms) {
            expect(count == 1 && wait_ms > 0 && wait_ms <= 20, "bounded poll wait");
            ++polls;
            pending->revents = pending->events;
            return 1;
        };
    }
    Step take() {
        expect(next < steps.size(), "unexpected extra syscall");
        const auto step = steps[next++];
        time += std::chrono::milliseconds(step.advance_ms);
        errno = step.error;
        return step;
    }
    Deadline limit() const { return Deadline{} + 100ms; }
};

void test_short_transfers() {
    char buffer[6]{};
    Script reading;
    reading.steps = {{-1, EINTR}, {1}, {-1, EAGAIN}, {1}, {1}, {1}, {1}, {1}};
    status(recv_exact(42, buffer, sizeof(buffer), reading.limit(), nullptr, &reading.hooks),
           IoStatus::Ok, 6, "short recv/EINTR/EAGAIN");
    expect(std::string(buffer, 6) == "abcdef" && reading.polls == 1, "short recv bytes");
    Script writing;
    writing.steps = {{1}, {-1, EINTR}, {2}, {-1, EWOULDBLOCK}, {3}};
    status(send_all(42, "abcdef", 6, writing.limit(), nullptr, &writing.hooks),
           IoStatus::Ok, 6, "short send/EINTR/EAGAIN");
    expect(writing.written == "abcdef" && writing.polls == 1, "short send preserves offsets");
}

void test_eof_and_errors() {
    char buffer[6]{};
    Script clean;
    clean.steps = {{0}};
    status(recv_exact(42, buffer, 6, clean.limit(), nullptr, &clean.hooks), IoStatus::Eof, 0, "clean EOF");
    Script partial;
    partial.steps = {{2}, {0}};
    status(recv_exact(42, buffer, 6, partial.limit(), nullptr, &partial.hooks),
           IoStatus::Truncated, 2, "partial EOF");
    Script reset;
    reset.steps = {{1}, {-1, ECONNRESET}};
    const auto error = recv_exact(42, buffer, 6, reset.limit(), nullptr, &reset.hooks);
    status(error, IoStatus::SystemError, 1, "recv system error retains progress");
    expect(error.error_number == ECONNRESET, "recv errno preserved");
    Script broken;
    broken.steps = {{-1, EPIPE}};
    const auto failed = send_all(42, "abcdef", 6, broken.limit(), nullptr, &broken.hooks);
    status(failed, IoStatus::SystemError, 0, "send system error");
    expect(failed.error_number == EPIPE, "send errno preserved");
    Script zero_send;
    zero_send.steps = {{0}};
    const auto zero = send_all(42, "a", 1, zero_send.limit(), nullptr, &zero_send.hooks);
    expect(zero.status == IoStatus::SystemError && zero.error_number == EIO,
           "zero send cannot loop without progress");
    std::atomic<bool> stopped{true};
    Script empty;
    status(recv_exact(-1, nullptr, 0, Deadline{}, &stopped, &empty.hooks), IoStatus::Ok, 0, "empty recv");
    status(send_all(-1, nullptr, 0, Deadline{}, &stopped, &empty.hooks), IoStatus::Ok, 0, "empty send");
    expect(empty.next == 0 && empty.polls == 0, "empty transfer performs no syscall");
}

void test_deadlines_and_stop() {
    char buffer[6]{};
    Script expired;
    status(recv_exact(42, buffer, 6, Deadline{}, nullptr, &expired.hooks),
           IoStatus::Timeout, 0, "expired deadline prevents syscall");
    Script drip;
    drip.steps = {{1, 0, 3}, {1, 0, 3}};
    status(recv_exact(42, buffer, 6, Deadline{} + 5ms, nullptr, &drip.hooks),
           IoStatus::Timeout, 2, "drip progress must not reset whole deadline");
    Script interrupts;
    interrupts.steps = {{-1, EINTR, 3}, {-1, EINTR, 3}};
    status(send_all(42, "abcdef", 6, Deadline{} + 5ms, nullptr, &interrupts.hooks),
           IoStatus::Timeout, 0, "EINTR must check deadline");
    Script waiting;
    waiting.steps = {{-1, EAGAIN}, {-1, EAGAIN}};
    waiting.hooks.poll = [&](pollfd*, nfds_t, int timeout) {
        expect(timeout > 0 && timeout <= 20, "poll cap");
        ++waiting.polls;
        waiting.time += std::chrono::milliseconds(timeout);
        return 0;
    };
    status(recv_exact(42, buffer, 6, Deadline{} + 25ms, nullptr, &waiting.hooks),
           IoStatus::Timeout, 0, "poll expiry");
    expect(waiting.polls == 2 && waiting.time == Deadline{} + 25ms, "poll recalculates remaining time");
    Script poll_interrupt;
    poll_interrupt.steps = {{-1, EAGAIN}, {-1, EAGAIN}, {6}};
    poll_interrupt.hooks.poll = [&](pollfd* pending, nfds_t, int) {
        if (++poll_interrupt.polls == 1) { errno = EINTR; return -1; }
        pending->revents = POLLIN;
        return 1;
    };
    status(recv_exact(42, buffer, 6, poll_interrupt.limit(), nullptr, &poll_interrupt.hooks),
           IoStatus::Ok, 6, "poll EINTR recovery");
    Script poll_error;
    poll_error.steps = {{-1, EAGAIN}};
    poll_error.hooks.poll = [](pollfd*, nfds_t, int) { errno = EIO; return -1; };
    const auto failure = recv_exact(42, buffer, 6, poll_error.limit(), nullptr, &poll_error.hooks);
    expect(failure.status == IoStatus::SystemError && failure.error_number == EIO, "poll error propagated");
    Script invalid;
    invalid.steps = {{-1, EAGAIN}};
    invalid.hooks.poll = [](pollfd* pending, nfds_t, int) { pending->revents = POLLNVAL; return 1; };
    expect(send_all(42, "a", 1, invalid.limit(), nullptr, &invalid.hooks).error_number == EBADF,
           "poll invalid fd reported");
    std::atomic<bool> stop{true};
    Script pre_stopped;
    status(send_all(42, "a", 1, pre_stopped.limit(), &stop, &pre_stopped.hooks),
           IoStatus::Stopped, 0, "pre-stopped operation");
    stop.store(false);
    Script cancelled;
    cancelled.steps = {{1}, {-1, EAGAIN}};
    cancelled.hooks.poll = [&](pollfd*, nfds_t, int timeout) {
        expect(timeout <= 20, "stop responsiveness poll cap");
        stop.store(true);
        return 0;
    };
    status(send_all(42, "abcdef", 6, cancelled.limit(), &stop, &cancelled.hooks),
           IoStatus::Stopped, 1, "stop during waiting retains progress");
}

struct SocketPair {
    UniqueFd first;
    UniqueFd second;
    SocketPair() {
        int sockets[2];
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) == 0, "socketpair");
        first.reset(sockets[0]);
        second.reset(sockets[1]);
    }
};
void test_native_io() {
    SocketPair pair;
    char bytes[4]{};
    status(send_all(pair.first.get(), "ping", 4, Clock::now() + 1s), IoStatus::Ok, 4, "native send");
    status(recv_exact(pair.second.get(), bytes, 4, Clock::now() + 1s), IoStatus::Ok, 4, "native recv");
    expect(std::string(bytes, 4) == "ping", "native bytes");
    status(recv_exact(pair.second.get(), bytes, 1, Clock::now() + 10ms),
           IoStatus::Timeout, 0, "blocking fd remains deadline bounded");
    status(send_all(pair.first.get(), "xy", 2, Clock::now() + 1s), IoStatus::Ok, 2, "native partial send");
    expect(::shutdown(pair.first.get(), SHUT_WR) == 0, "half close");
    status(recv_exact(pair.second.get(), bytes, 4, Clock::now() + 1s),
           IoStatus::Truncated, 2, "native partial EOF");
    status(recv_exact(pair.second.get(), bytes, 1, Clock::now() + 1s),
           IoStatus::Eof, 0, "native clean EOF");
    // Half-close still permits a response in the other direction.
    status(send_all(pair.second.get(), "pong", 4, Clock::now() + 1s), IoStatus::Ok, 4, "half close response");
    status(recv_exact(pair.first.get(), bytes, 4, Clock::now() + 1s), IoStatus::Ok, 4, "half close read response");
    pair.first.reset();
    const auto broken = send_all(pair.second.get(), "x", 1, Clock::now() + 1s);
    expect(broken.status == IoStatus::SystemError && broken.error_number == EPIPE,
           "disconnected send reports EPIPE without SIGPIPE terminating process");
}
void test_bounded_connect() {
    auto listener = listen_tcp(0);
    auto client = connect_tcp_until(Clock::now() + 1s, "127.0.0.1", listener.port);
    expect(::fcntl(client.get(), F_GETFL) & O_NONBLOCK, "deadline connect returns nonblocking socket");
    auto server = accept_tcp(listener.fd.get());
    char byte{};
    status(send_all(client.get(), "z", 1, Clock::now() + 1s), IoStatus::Ok, 1, "connected send");
    status(recv_exact(server.get(), &byte, 1, Clock::now() + 1s), IoStatus::Ok, 1, "connected recv");
    expect(byte == 'z', "bounded connect TCP bytes");
    bool timeout = false;
    try { auto fd = connect_tcp_until(Clock::now(), "127.0.0.1", listener.port); }
    catch (const std::system_error& error) { timeout = error.code().value() == ETIMEDOUT; }
    expect(timeout, "expired connect deadline");
    std::atomic<bool> stop{true};
    bool stopped = false;
    try { auto fd = connect_tcp_until(Clock::now() + 1s, "127.0.0.1", listener.port, &stop); }
    catch (const std::system_error& error) { stopped = error.code().value() == ECANCELED; }
    expect(stopped, "stopped connect");
    // Shutdown preserves port reservation while disabling acceptance.
    expect(::shutdown(listener.fd.get(), SHUT_RDWR) == 0, "disable listener");
    bool refused = false;
    try { auto fd = connect_tcp_until(Clock::now() + 1s, "127.0.0.1", listener.port); }
    catch (const std::system_error& error) { refused = error.code().value() == ECONNREFUSED; }
    expect(refused, "SO_ERROR reports refused connection");
}
}  // namespace

int main() {
    try {
        test_short_transfers();
        test_eof_and_errors();
        test_deadlines_and_stop();
        test_native_io();
        test_bounded_connect();
        std::cout << "test_net_io passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_net_io failed: " << error.what() << '\n';
        return 1;
    }
}
