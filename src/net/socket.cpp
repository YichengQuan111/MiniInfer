#include "miniinfer/net/socket.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace miniinfer::net {
namespace {
[[noreturn]] void fail(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}
sockaddr_in address(const std::string& host, std::uint16_t port) {
    sockaddr_in result{};
    result.sin_family = AF_INET;
    result.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &result.sin_addr) != 1)
        throw std::invalid_argument("host must be an IPv4 literal");
    return result;
}
UniqueFd make_socket(bool nonblocking = false) {
    UniqueFd result(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC |
                           (nonblocking ? SOCK_NONBLOCK : 0), 0));
    if (!result) fail("socket");
    return result;
}
}  // namespace

TcpListener listen_tcp(std::uint16_t port, const std::string& host, int backlog) {
    if (backlog <= 0) throw std::invalid_argument("backlog must be positive");
    auto addr = address(host, port);
    auto fd = make_socket();
    const int reuse = 1;
    if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
        fail("setsockopt(SO_REUSEADDR)");
    if (::bind(fd.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0)
        fail("bind");
    if (::listen(fd.get(), backlog) < 0) fail("listen");
    socklen_t length = sizeof(addr);
    if (::getsockname(fd.get(), reinterpret_cast<sockaddr*>(&addr), &length) < 0)
        fail("getsockname");
    return {std::move(fd), ntohs(addr.sin_port)};
}

UniqueFd connect_tcp(std::uint16_t port, const std::string& host) {
    const auto addr = address(host, port);
    auto fd = make_socket();
    // A blocking connect interrupted by a signal is already in progress.
    // Report it and close this socket instead of retrying connect on it.
    if (::connect(fd.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0)
        fail("connect");
    return fd;
}

UniqueFd connect_tcp_until(Deadline deadline, const std::string& host, std::uint16_t port,
                           const std::atomic<bool>* stop) {
    const auto addr = address(host, port);
    auto fd = make_socket(true);
    const auto check = [&] {
        if (stop && stop->load(std::memory_order_relaxed))
            throw std::system_error(ECANCELED, std::generic_category(), "connect stopped");
        if (Clock::now() >= deadline)
            throw std::system_error(ETIMEDOUT, std::generic_category(), "connect timeout");
    };
    check();
    if (::connect(fd.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0) {
        check();
        return fd;
    }
    if (errno != EINPROGRESS && errno != EINTR) fail("connect");
    for (;;) {
        check();
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        // Time may advance between check and computing the remainder.
        if (remaining.count() <= 0) continue;
        pollfd pending{fd.get(), POLLOUT, 0};
        const int wait_ms = static_cast<int>(std::min<std::int64_t>(20, remaining.count()));
        const int ready = ::poll(&pending, 1, wait_ms);
        if (ready < 0) {
            if (errno == EINTR) continue;
            fail("poll(connect)");
        }
        check();
        if (ready == 0) continue;
        int error = 0;
        socklen_t length = sizeof(error);
        if (::getsockopt(fd.get(), SOL_SOCKET, SO_ERROR, &error, &length) < 0)
            fail("getsockopt(SO_ERROR)");
        if (error != 0) throw std::system_error(error, std::generic_category(), "connect");
        return fd;
    }
}

UniqueFd accept_tcp(int listener_fd) {
    int fd;
    do {
        fd = ::accept4(listener_fd, nullptr, nullptr, SOCK_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) fail("accept");
    return UniqueFd(fd);
}

}  // namespace miniinfer::net
