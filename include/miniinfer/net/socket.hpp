#pragma once

#include "miniinfer/net/unique_fd.hpp"
#include "miniinfer/net/io.hpp"

#include <cstdint>
#include <string>

namespace miniinfer::net {

struct TcpListener {
    UniqueFd fd;
    std::uint16_t port;  // Actual bound port, including when port 0 was requested.
};

// IPv4 literals only. These small D16.1 helpers use blocking sockets and throw
// std::system_error on syscall failures; invalid addresses throw invalid_argument.
TcpListener listen_tcp(std::uint16_t port, const std::string& host = "127.0.0.1",
                       int backlog = 16);
UniqueFd connect_tcp(std::uint16_t port, const std::string& host = "127.0.0.1");
// Nonblocking connect with a whole-operation deadline; returns a nonblocking fd.
// Timeout/cancellation throw system_error with ETIMEDOUT/ECANCELED respectively.
UniqueFd connect_tcp_until(Deadline deadline, const std::string& host, std::uint16_t port,
                           const std::atomic<bool>* stop = nullptr);
UniqueFd accept_tcp(int listener_fd);

}  // namespace miniinfer::net
