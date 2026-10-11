#include "miniinfer/mlp_model.hpp"
#include "miniinfer/net/server.hpp"
#include "miniinfer/runtime.hpp"

#include <arpa/inet.h>
#include <pthread.h>

#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace {
struct Options {
    std::string model;
    miniinfer::RuntimeConfig runtime;
    miniinfer::net::ServerConfig server;
};
std::size_t number(const std::string& text, std::size_t maximum, bool allow_zero = false) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal integer");
    unsigned long long value;
    try { value = std::stoull(text); }
    catch (const std::exception&) { throw std::invalid_argument("integer out of range"); }
    if ((!allow_zero && value == 0) || value > maximum)
        throw std::invalid_argument("integer out of range");
    return static_cast<std::size_t>(value);
}
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (++i >= argc) throw std::invalid_argument("missing option value: " + argument);
        const std::string value = argv[i];
        if (argument == "--model") options.model = value;
        else if (argument == "--host") options.server.host = value;
        else if (argument == "--port") options.server.port = static_cast<std::uint16_t>(number(value, 65535, true));
        else if (argument == "--connections") options.server.max_connections = number(value, 1000000, true);
        else if (argument == "--connection-threads") options.server.connection_threads = number(value, 1024);
        else if (argument == "--waiting-connections") options.server.waiting_connections = number(value, 1000000);
        else if (argument == "--queue-timeout-ms")
            options.server.queue_timeout = std::chrono::milliseconds(number(value, 600000));
        else if (argument == "--workers") options.runtime.workers = number(value, 1024);
        else if (argument == "--queue-capacity") options.runtime.queue_capacity = number(value, 1000000);
        else if (argument == "--read-timeout-ms")
            options.server.session.read_timeout = std::chrono::milliseconds(number(value, 600000));
        else if (argument == "--write-timeout-ms")
            options.server.session.write_timeout = std::chrono::milliseconds(number(value, 600000));
        else throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.model.empty()) throw std::invalid_argument("--model is required");
    in_addr address{};
    if (::inet_pton(AF_INET, options.server.host.c_str(), &address) != 1)
        throw std::invalid_argument("--host must be an IPv4 literal");
    return options;
}
void help() {
    std::cout << "infer_server --model DIR [--host IPv4] [--port 0..65535] "
                 "[--connections N] [--workers N] [--queue-capacity N]\n"
                 "             [--connection-threads N] [--waiting-connections N] [--queue-timeout-ms N]\n"
                 "             [--read-timeout-ms N] [--write-timeout-ms N]\n"
                 "Defaults: host=127.0.0.1 port=9000 connections=0 workers=1 queue-capacity=64\n"
                 "Connection threads=4 waiting connections=8 queue timeout=1000ms.\n"
                 "Timeouts default to 5000ms; range 1..600000. Connections=0 serves indefinitely.\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { help(); return 0; }
    Options options;
    try {
        options = parse(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "infer_server: " << error.what() << '\n';
        return 2;
    }
    try {
        miniinfer::net::check_wire_platform();
        // Block before Runtime creates any workers. All later threads inherit
        // this mask; main consumes termination signals synchronously below.
        sigset_t stop_signals;
        ::sigemptyset(&stop_signals);
        ::sigaddset(&stop_signals, SIGINT);
        ::sigaddset(&stop_signals, SIGTERM);
        const int mask_error = ::pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
        if (mask_error != 0)
            throw std::system_error(mask_error, std::generic_category(), "pthread_sigmask");
        // Model loading is done once; the Runtime owns it through all sessions.
        miniinfer::Runtime runtime(miniinfer::MlpModel::load(options.model), options.runtime);
        // Runtime is declared first and therefore destroyed after the server.
        miniinfer::net::TcpServer server(options.server,
            [&](miniinfer::MlpInput input) { return runtime.run(std::move(input)); },
            [&] { runtime.shutdown(); });
        std::cout << "Listening on " << options.server.host << ':' << server.port() << std::endl;
        while (!server.finished()) {
            const timespec timeout{0, 20000000};
            const int signal = ::sigtimedwait(&stop_signals, nullptr, &timeout);
            if (signal == SIGINT || signal == SIGTERM) break;
            if (signal < 0 && errno != EAGAIN && errno != EINTR)
                throw std::system_error(errno, std::generic_category(), "sigtimedwait");
        }
        server.shutdown();
        const auto stats = server.snapshot();
        if (stats.rejected || stats.expired || stats.io_failures || stats.protocol_errors ||
            stats.session_failures || stats.busy || stats.inference_failures || stats.accept_errors) {
            std::cerr << "infer_server: accepted=" << stats.accepted << " rejected=" << stats.rejected
                      << " expired=" << stats.expired << " completed=" << stats.completed
                      << " requests=" << stats.requests << " responses=" << stats.responses
                      << " bad_requests=" << stats.bad_requests << " busy=" << stats.busy
                      << " io_failures=" << stats.io_failures << " protocol_errors=" << stats.protocol_errors
                      << " inference_failures=" << stats.inference_failures
                      << " session_failures=" << stats.session_failures
                      << " accept_errors=" << stats.accept_errors << '\n';
        }
        return stats.accept_errors || stats.session_failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "infer_server: " << error.what() << '\n';
        return 1;
    }
}
