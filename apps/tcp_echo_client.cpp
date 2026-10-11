#include "miniinfer/net/socket.hpp"
#include "miniinfer/net/protocol.hpp"

#include <sys/socket.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

int main(int argc, char** argv) {
    try {
        std::string host = "127.0.0.1";
        std::uint16_t port = 9000;
        std::string message = "hello MiniInfer";
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "tcp_echo_client [--host IPv4] [--port 1..65535] [--message TEXT]\n";
                return 0;
            }
            if (++i >= argc) throw std::invalid_argument("missing option value");
            if (option == "--host") host = argv[i];
            else if (option == "--message") message = argv[i];
            else if (option == "--port") {
                const std::string value = argv[i];
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("invalid port");
                const auto number = std::stoul(value);
                if (number == 0 || number > 65535) throw std::invalid_argument("port out of range");
                port = static_cast<std::uint16_t>(number);
            } else throw std::invalid_argument("unknown option: " + option);
        }
        // Keep this demonstration bounded so writing the request before reading
        // cannot deadlock against an echo server writing back a large request.
        if (message.size() > 1024) throw std::invalid_argument("message exceeds 1024 bytes");
        using namespace miniinfer::net;
        auto fd = connect_tcp_until(Clock::now() + std::chrono::seconds(5), host, port);
        const auto request = echo_request(1, {message.begin(), message.end()});
        if (send_frame(fd.get(), request, Clock::now() + std::chrono::seconds(5)).status != IoStatus::Ok)
            throw std::runtime_error("echo send failed");
        if (::shutdown(fd.get(), SHUT_WR) < 0)
            throw std::system_error(errno, std::generic_category(), "shutdown");
        Frame frame;
        if (recv_frame(fd.get(), frame, Clock::now() + std::chrono::seconds(5)).status != IoStatus::Ok)
            throw std::runtime_error("echo receive failed");
        const auto response = decode_response(frame, echo_type, 1);
        if (response.status != Status::Ok) throw std::runtime_error("echo business error");
        const std::string echoed(response.echo.begin(), response.echo.end());
        if (echoed != message) throw std::runtime_error("echo response differs from request");
        std::cout << echoed << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tcp_echo_client: " << error.what() << '\n';
        return 1;
    }
}
