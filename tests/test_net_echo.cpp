#include "miniinfer/net/socket.hpp"
#include "miniinfer/net/protocol.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>

#include <array>
#include <cerrno>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace {
using miniinfer::net::UniqueFd;
using namespace miniinfer::net;
using namespace std::chrono_literals;
static_assert(!std::is_copy_constructible_v<UniqueFd>);
static_assert(!std::is_copy_assignable_v<UniqueFd>);
static_assert(std::is_nothrow_move_constructible_v<UniqueFd>);
static_assert(std::is_nothrow_move_assignable_v<UniqueFd>);

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::size_t descriptor_count() {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        (void)entry;
        ++count;
    }
    return count;
}
bool closed(int fd) {
    errno = 0;
    return ::fcntl(fd, F_GETFD) == -1 && errno == EBADF;
}
struct JoiningThread {
    std::thread thread;
    ~JoiningThread() { if (thread.joinable()) thread.join(); }
};

void test_ownership() {
    const auto before = descriptor_count();
    int final_fd = -1;
    {
        UniqueFd first(::socket(AF_INET, SOCK_STREAM, 0));
        UniqueFd second(::socket(AF_INET, SOCK_STREAM, 0));
        expect(first && second, "create ownership sockets");
        final_fd = first.get();
        const int replaced = second.get();
        UniqueFd moved(std::move(first));
        expect(!first && moved.get() == final_fd, "move construction transfers ownership");
        second = std::move(moved);
        expect(!moved && second.get() == final_fd, "move assignment transfers ownership");
        expect(closed(replaced), "move assignment closes replaced descriptor");
        auto& alias = second;
        second = std::move(alias);
        expect(second.get() == final_fd, "self move retains ownership");
        second.reset(second.get());
        expect(::fcntl(final_fd, F_GETFD) >= 0, "same descriptor reset retains ownership");
    }
    expect(closed(final_fd), "destructor closes descriptor");
    {
        UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
        const int owned = fd.get();
        fd.reset();
        expect(!fd && closed(owned), "reset closes descriptor");
    }
    expect(descriptor_count() == before, "ownership test leaked descriptors");
}

void test_connect_failure() {
    const auto before = descriptor_count();
    {
        // Keep an ephemeral port reserved without listening, so connection
        // refusal does not race another service acquiring a just-freed port.
        UniqueFd reservation(::socket(AF_INET, SOCK_STREAM, 0));
        expect(static_cast<bool>(reservation), "create reserved socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(::bind(reservation.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
               "reserve ephemeral port");
        socklen_t length = sizeof(addr);
        expect(::getsockname(reservation.get(), reinterpret_cast<sockaddr*>(&addr), &length) == 0,
               "read reserved port");
        bool refused = false;
        try {
            auto fd = miniinfer::net::connect_tcp(ntohs(addr.sin_port));
        } catch (const std::system_error& error) {
            refused = error.code().value() == ECONNREFUSED;
        }
        expect(refused, "connect to non-listening reserved port must fail");
        bool invalid = false;
        try { auto fd = miniinfer::net::connect_tcp(9000, "not-an-ip"); }
        catch (const std::invalid_argument&) { invalid = true; }
        expect(invalid, "reject invalid IPv4 literal");
    }
    expect(descriptor_count() == before, "failed connect leaked descriptor");
}

void test_echo() {
    const auto before = descriptor_count();
    {
        constexpr unsigned connections = 8;
        auto listener = miniinfer::net::listen_tcp(0);
        expect(listener.port != 0, "system must allocate listener port");
        std::promise<void> ready;
        auto readiness = ready.get_future();
        std::promise<void> done;
        auto result = done.get_future();
        JoiningThread server{std::thread([&] {
            try {
                ready.set_value();  // bind/listen completed before clients connect.
                for (unsigned i = 0; i < connections; ++i) {
                    pollfd pending{listener.fd.get(), POLLIN, 0};
                    int state;
                    do { state = ::poll(&pending, 1, 2000); } while (state < 0 && errno == EINTR);
                    expect(state > 0 && (pending.revents & POLLIN), "accept readiness timed out");
                    auto fd = miniinfer::net::accept_tcp(listener.fd.get());
                    Frame request;
                    expect(recv_frame(fd.get(), request, Clock::now() + 2s).status == IoStatus::Ok, "receive echo frame");
                    expect(send_frame(fd.get(), echo_response(request), Clock::now() + 2s).status == IoStatus::Ok, "send echo frame");
                    expect(recv_frame(fd.get(), request, Clock::now() + 2s).status == IoStatus::Eof, "clean EOF after full request");
                }
                done.set_value();
            } catch (...) {
                done.set_exception(std::current_exception());
            }
        })};
        // Server receives/sends have finite test-only timeouts. If a client
        // assertion throws, its fd closes before this guard joins the worker.
        readiness.get();
        for (unsigned i = 0; i < connections; ++i) {
            auto client = connect_tcp_until(Clock::now() + 2s, "127.0.0.1", listener.port);
            // Empty request verifies EOF; embedded NUL verifies raw bytes;
            // 1024 bytes crosses our small receiving buffer repeatedly.
            std::string message = i == 0 ? std::string{} : std::string(1024, static_cast<char>('a' + i));
            if (!message.empty()) message[15] = '\0';
            const auto request = echo_request(i, {message.begin(), message.end()});
            expect(send_frame(client.get(), request, Clock::now() + 2s).status == IoStatus::Ok, "send request");
            expect(::shutdown(client.get(), SHUT_WR) == 0, "client half close");
            Frame frame;
            expect(recv_frame(client.get(), frame, Clock::now() + 2s).status == IoStatus::Ok, "read response after half close");
            expect(decode_response(frame, echo_type, i).echo == request.body, "TCP echo must preserve bytes");
        }
        result.get();
    }
    expect(descriptor_count() == before, "repeated TCP connections leaked descriptors");
}
}  // namespace

int main() {
    try {
        test_ownership();
        test_connect_failure();
        test_echo();
        std::cout << "test_net_echo passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_net_echo failed: " << error.what() << '\n';
        return 1;
    }
}
