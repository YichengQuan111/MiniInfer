#include "miniinfer/net/socket.hpp"
#include "miniinfer/net/protocol.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
unsigned parse_number(const std::string& value, unsigned maximum) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal integer");
    const auto number = std::stoul(value);
    if (number > maximum) throw std::invalid_argument("number out of range");
    return static_cast<unsigned>(number);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string host = "127.0.0.1";
        std::uint16_t port = 9000;
        unsigned connections = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "tcp_echo_server [--host IPv4] [--port 0..65535] "
                             "[--connections N]\nN defaults to 1; 0 serves indefinitely.\n";
                return 0;
            }
            if (++i >= argc) throw std::invalid_argument("missing option value");
            if (option == "--host") host = argv[i];
            else if (option == "--port") port = static_cast<std::uint16_t>(parse_number(argv[i], 65535));
            else if (option == "--connections") connections = parse_number(argv[i], 1000000);
            else throw std::invalid_argument("unknown option: " + option);
        }
        auto listener = miniinfer::net::listen_tcp(port, host);
        std::cout << "Listening on " << host << ':' << listener.port << std::endl;
        for (unsigned served = 0; connections == 0 || served < connections; ++served) {
            auto connection = miniinfer::net::accept_tcp(listener.fd.get());
            using namespace miniinfer::net;
            bool have_id = false;
            std::uint32_t last_id = 0;
            for (;;) {
                Frame request;
                try {
                    if (recv_frame(connection.get(), request, Clock::now() + std::chrono::seconds(5)).status != IoStatus::Ok)
                        break;
                } catch (const ProtocolError&) { break; }
                Frame response;
                bool close = false;
                try {
                    if (have_id && request.request_id <= last_id)
                        throw ProtocolError("request_id must increase");
                    response = echo_response(request);
                } catch (const ProtocolError&) {
                    if (request.type & response_bit) break;
                    response = error_response(request, Status::BadRequest, "invalid echo request");
                    close = true;
                }
                have_id = true;
                last_id = request.request_id;
                if (send_frame(connection.get(), response, Clock::now() + std::chrono::seconds(5)).status != IoStatus::Ok || close)
                    break;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tcp_echo_server: " << error.what() << '\n';
        return 1;
    }
}
