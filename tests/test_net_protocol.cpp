#include "miniinfer/net/protocol.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace miniinfer::net;
using namespace std::chrono_literals;
void expect(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class F> void bad(F f) {
    try { f(); } catch (const ProtocolError&) { return; }
    throw std::runtime_error("expected ProtocolError");
}
void put32(std::vector<std::uint8_t>& b, std::size_t at, std::uint32_t n) {
    for (int i = 3; i >= 0; --i) { b[at + i] = static_cast<std::uint8_t>(n); n >>= 8; }
}
void codecs() {
    check_wire_platform();
    const auto frame = echo_request(0x01020304, {'a', 0, 'b'});
    const std::vector<std::uint8_t> fixed{
        'M','I','N','I',0,1,0,1,1,2,3,4,0,0,0,3,'a',0,'b'};
    expect(encode_frame(frame) == fixed, "fixed header byte order");
    auto h = encode_header({mlp_type, 99, 65540});
    expect(decode_header(h).body_size == 65540, "max header size");
    for (const auto index : {0, 4, 5, 6, 7}) {
        auto corrupt = h; corrupt[index] = 42;
        bad([&] { (void)decode_header(corrupt); });
    }
    for (int i = 12; i < 16; ++i) h[i] = 255;
    bad([&] { (void)decode_header(h); });
    bad([] { (void)encode_header({1, 1, 65541}); });
    bad([] { (void)echo_request(1, std::vector<std::uint8_t>(1025)); });
    const auto e = decode_response(echo_response(frame), 1, frame.request_id);
    expect(e.echo == frame.body, "echo round trip");
    bad([&] { (void)decode_response(echo_response(frame), 1, 100); });
    bad([&] { (void)decode_response(echo_response(frame), 2, frame.request_id); });
    for (std::uint32_t code = 1; code <= 4; ++code) {
        const auto r = decode_response(error_response(frame, static_cast<Status>(code), "error"), 1, frame.request_id);
        expect(r.status == static_cast<Status>(code) && r.message == "error", "error response");
    }
    auto corrupt = echo_response(frame);
    put32(corrupt.body, 0, 5);
    bad([&] { (void)decode_response(corrupt, 1, frame.request_id); });
    corrupt.body.resize(3);
    bad([&] { (void)decode_response(corrupt, 1, frame.request_id); });
    bad([&] { (void)error_response(frame, Status::Busy, std::string(1025, 'x')); });
    for (std::uint32_t batch : {1, 8, 32, 64, 256}) {
        std::vector<float> values(batch * 64, 1.0f);
        values[1] = -2.5f; values[2] = -0.0f;
        auto request = mlp_request(7, miniinfer::MlpInput(batch, values));
        expect(request.body.size() == 4 + batch * 256, "request length");
        expect(request.body[4] == 0x3f && request.body[5] == 0x80 && request.body[8] == 0xc0 &&
               request.body[9] == 0x20 && request.body[12] == 0x80, "FP32 wire bits");
        auto input = decode_mlp_request(request);
        expect(input.batch() == batch && std::memcmp(input.view().data(), values.data(), values.size() * 4) == 0,
               "request round trip");
        auto changed = request; changed.body.pop_back();
        bad([&] { (void)decode_mlp_request(changed); });
        changed = request; changed.body.push_back(0);
        bad([&] { (void)decode_mlp_request(changed); });
        for (auto invalid : {0U, 257U, 0xffffffffU}) {
            changed = request; put32(changed.body, 0, invalid);
            bad([&] { (void)decode_mlp_request(changed); });
        }
        for (auto invalid : {0x7f800000U, 0xff800000U, 0x7fc00000U}) {
            changed = request; put32(changed.body, 4, invalid);
            bad([&] { (void)decode_mlp_request(changed); });
        }
        Frame response{static_cast<std::uint16_t>(2 | response_bit), 7, std::vector<std::uint8_t>(8 + batch * 44)};
        put32(response.body, 4, batch);
        auto r = decode_response(response, 2, 7, batch);
        expect(r.logits.size() == batch * 10 && r.labels.size() == batch, "response exact length");
        bad([&] { (void)decode_response(response, 2, 7, batch == 1 ? 2 : 1); });
        put32(response.body, 8 + batch * 40, 10);
        bad([&] { (void)decode_response(response, 2, 7, batch); });
        put32(response.body, 8 + batch * 40, 0);
        put32(response.body, 8, 0x7fc00000);
        bad([&] { (void)decode_response(response, 2, 7, batch); });
    }
}

void stream_boundaries() {
    const auto one = echo_request(1, {'x','y','z'});
    const auto two = echo_request(2, {});
    auto bytes = encode_frame(one);
    const auto next = encode_frame(two);
    bytes.insert(bytes.end(), next.begin(), next.end());
    for (std::size_t chunk : {1, 2, 7, 100}) {
        std::size_t offset = 0;
        IoHooks hooks;
        hooks.recv = [&](int, void* out, std::size_t size, int) -> ssize_t {
            const auto n = std::min({size, chunk, bytes.size() - offset});
            std::memcpy(out, bytes.data() + offset, n); offset += n;
            return static_cast<ssize_t>(n);
        };
        Frame actual;
        expect(recv_frame(-1, actual, Clock::now() + 1s, nullptr, &hooks).status == IoStatus::Ok &&
               actual.request_id == 1 && actual.body == one.body, "fragmented first frame");
        expect(recv_frame(-1, actual, Clock::now() + 1s, nullptr, &hooks).status == IoStatus::Ok &&
               actual.request_id == 2 && actual.body.empty(), "concatenated frame boundary");
        expect(recv_frame(-1, actual, Clock::now() + 1s, nullptr, &hooks).status == IoStatus::Eof, "clean frame EOF");
    }
    const auto encoded = encode_frame(one);
    for (std::size_t available = 0; available < encoded.size(); ++available) {
        std::size_t offset = 0;
        IoHooks hooks;
        hooks.recv = [&](int, void* out, std::size_t size, int) -> ssize_t {
            const auto n = std::min(size, available - offset);
            std::memcpy(out, encoded.data() + offset, n); offset += n;
            return static_cast<ssize_t>(n);
        };
        Frame actual;
        const auto io = recv_frame(-1, actual, Clock::now() + 1s, nullptr, &hooks);
        expect(io.status == (available == 0 ? IoStatus::Eof : IoStatus::Truncated), "every partial EOF");
        expect(io.transferred == available, "EOF byte accounting");
    }
    auto oversized = encoded;
    put32(oversized, 12, 0xffffffff);
    std::size_t offset = 0;
    IoHooks hooks;
    hooks.recv = [&](int, void* out, std::size_t n, int) -> ssize_t {
        expect(offset + n <= 16, "oversized header must not consume body");
        std::memcpy(out, oversized.data() + offset, n); offset += n;
        return static_cast<ssize_t>(n);
    };
    Frame actual;
    bad([&] { (void)recv_frame(-1, actual, Clock::now() + 1s, nullptr, &hooks); });
    expect(offset == 16, "oversized header rejected before allocation/body");
    offset = 0;
    auto now = Deadline{};
    hooks.now = [&] { return now; };
    hooks.recv = [&](int, void* out, std::size_t, int) -> ssize_t {
        *static_cast<std::uint8_t*>(out) = encoded[offset++];
        now += 1ms;
        return 1;
    };
    const auto io = recv_frame(-1, actual, Deadline{} + 18ms, nullptr, &hooks);
    expect(io.status == IoStatus::Timeout && io.transferred == 18, "header and body share one deadline");
    std::vector<std::uint8_t> sent;
    hooks = {};
    hooks.send = [&](int, const void* data, std::size_t n, int) -> ssize_t {
        const auto* p = static_cast<const std::uint8_t*>(data);
        n = std::min<std::size_t>(n, 2); sent.insert(sent.end(), p, p + n);
        return static_cast<ssize_t>(n);
    };
    expect(send_frame(-1, one, Clock::now() + 1s, nullptr, &hooks).status == IoStatus::Ok &&
           sent == encoded, "framed short writes preserve all bytes");
}
}  // namespace
int main() {
    try {
        codecs(); stream_boundaries();
        std::cout << "protocol: wire fixtures, bounded codecs, fragmented/concatenated frames, EOF and deadlines passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
