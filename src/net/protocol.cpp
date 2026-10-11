#include "miniinfer/net/protocol.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace miniinfer::net {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw ProtocolError(message);
}
bool valid_type(std::uint16_t type) {
    return type == echo_type || type == mlp_type ||
           type == (echo_type | response_bit) || type == (mlp_type | response_bit);
}
std::uint32_t u32(const std::uint8_t* p) {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) |
           (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
}
void put32(std::uint8_t* p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v >> 24);
    p[1] = static_cast<std::uint8_t>(v >> 16);
    p[2] = static_cast<std::uint8_t>(v >> 8);
    p[3] = static_cast<std::uint8_t>(v);
}
void append32(std::vector<std::uint8_t>& bytes, std::uint32_t v) {
    const auto offset = bytes.size();
    bytes.resize(offset + 4);
    put32(bytes.data() + offset, v);
}
void append_float(std::vector<std::uint8_t>& bytes, float value) {
    require(std::isfinite(value), "non-finite FP32 value");
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    append32(bytes, bits);
}
float read_float(const std::uint8_t* bytes) {
    auto bits = u32(bytes);
    float value;
    std::memcpy(&value, &bits, 4);
    require(std::isfinite(value), "non-finite FP32 value");
    return value;
}
void check_batch(std::size_t batch) {
    require(batch >= 1 && batch <= 256, "batch must be in [1,256]");
}
void check_header(const Header& h) {
    require(valid_type(h.type), "unknown message type");
    require(h.body_size <= max_body_bytes, "frame body exceeds limit");
}
}  // namespace

void check_wire_platform() {
    require(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
            std::numeric_limits<float>::digits == 24 &&
            std::numeric_limits<float>::max_exponent == 128,
            "wire protocol requires IEEE 754 binary32");
}

std::array<std::uint8_t, header_bytes> encode_header(const Header& h) {
    check_header(h);
    std::array<std::uint8_t, header_bytes> result{'M','I','N','I',0,1,0,0,0,0,0,0,0,0,0,0};
    result[6] = static_cast<std::uint8_t>(h.type >> 8);
    result[7] = static_cast<std::uint8_t>(h.type);
    put32(result.data() + 8, h.request_id);
    put32(result.data() + 12, h.body_size);
    return result;
}

Header decode_header(const std::array<std::uint8_t, header_bytes>& b) {
    require(b[0] == 'M' && b[1] == 'I' && b[2] == 'N' && b[3] == 'I', "bad frame magic");
    require(b[4] == 0 && b[5] == 1, "unsupported protocol version");
    Header h{static_cast<std::uint16_t>((std::uint16_t{b[6]} << 8) | b[7]),
             u32(b.data() + 8), u32(b.data() + 12)};
    check_header(h);
    return h;
}

std::vector<std::uint8_t> encode_frame(const Frame& frame) {
    require(frame.body.size() <= max_body_bytes, "frame body exceeds limit");
    const auto header = encode_header({frame.type, frame.request_id,
                                      static_cast<std::uint32_t>(frame.body.size())});
    std::vector<std::uint8_t> result;
    result.reserve(header_bytes + frame.body.size());
    result.insert(result.end(), header.begin(), header.end());
    result.insert(result.end(), frame.body.begin(), frame.body.end());
    return result;
}

IoResult recv_frame(int fd, Frame& frame, Deadline deadline,
                    const std::atomic<bool>* stop, IoHooks* hooks) {
    std::array<std::uint8_t, header_bytes> bytes{};
    auto io = recv_exact(fd, bytes.data(), bytes.size(), deadline, stop, hooks);
    if (io.status != IoStatus::Ok) return io;
    const auto h = decode_header(bytes);  // Validate the global bound before allocation.
    Frame received{h.type, h.request_id, std::vector<std::uint8_t>(h.body_size)};
    io = recv_exact(fd, received.body.data(), received.body.size(), deadline, stop, hooks);
    io.transferred += header_bytes;
    if (io.status == IoStatus::Eof) io.status = IoStatus::Truncated;
    if (io.status == IoStatus::Ok) frame = std::move(received);
    return io;
}

IoResult send_frame(int fd, const Frame& frame, Deadline deadline,
                    const std::atomic<bool>* stop, IoHooks* hooks) {
    auto bytes = encode_frame(frame);
    return send_all(fd, bytes.data(), bytes.size(), deadline, stop, hooks);
}

Frame echo_request(std::uint32_t id, const std::vector<std::uint8_t>& bytes) {
    require(bytes.size() <= 1024, "echo exceeds 1024 bytes");
    return {echo_type, id, bytes};
}

Frame mlp_request(std::uint32_t id, const MlpInput& input) {
    check_wire_platform();
    check_batch(input.batch());
    input.validate(256);
    Frame frame{mlp_type, id, {}};
    frame.body.reserve(4 + input.batch() * 256);
    append32(frame.body, static_cast<std::uint32_t>(input.batch()));
    const auto view = input.view();
    for (std::size_t i = 0; i < view.size(); ++i) append_float(frame.body, view.data()[i]);
    return frame;
}

MlpInput decode_mlp_request(const Frame& frame) {
    check_wire_platform();
    require(frame.type == mlp_type, "expected MLP request");
    require(frame.body.size() >= 4 && frame.body.size() <= max_body_bytes, "bad MLP body size");
    const auto batch = u32(frame.body.data());
    // Constraining batch first makes every later product bounded, even on 32-bit size_t.
    check_batch(batch);
    require(frame.body.size() == 4 + std::size_t{batch} * 256, "MLP body length mismatch");
    std::vector<float> values(std::size_t{batch} * 64);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = read_float(frame.body.data() + 4 + i * 4);
    return MlpInput(batch, std::move(values));
}

Frame echo_response(const Frame& request) {
    require(request.type == echo_type && request.body.size() <= 1024, "invalid echo request");
    Frame result{static_cast<std::uint16_t>(echo_type | response_bit), request.request_id, {}};
    append32(result.body, 0);
    result.body.insert(result.body.end(), request.body.begin(), request.body.end());
    return result;
}

Frame mlp_response(std::uint32_t id, const MlpResult& result) {
    check_wire_platform();
    check_batch(result.batch());
    Frame frame{static_cast<std::uint16_t>(mlp_type | response_bit), id, {}};
    frame.body.reserve(8 + result.batch() * 44);
    append32(frame.body, 0);
    append32(frame.body, static_cast<std::uint32_t>(result.batch()));
    const auto view = result.logits();
    for (std::size_t i = 0; i < view.size(); ++i) append_float(frame.body, view.data()[i]);
    for (auto label : result.labels()) {
        require(label >= 0 && label <= 9, "invalid MLP label");
        append32(frame.body, static_cast<std::uint32_t>(label));
    }
    return frame;
}

Frame error_response(const Frame& request, Status status, const std::string& message) {
    require(request.type == echo_type || request.type == mlp_type, "invalid request type");
    require(status != Status::Ok && static_cast<std::uint32_t>(status) <= 4, "invalid error status");
    require(message.size() <= 1024, "error description exceeds 1024 bytes");
    Frame frame{static_cast<std::uint16_t>(request.type | response_bit), request.request_id, {}};
    append32(frame.body, static_cast<std::uint32_t>(status));
    frame.body.insert(frame.body.end(), message.begin(), message.end());
    return frame;
}

Response decode_response(const Frame& frame, std::uint16_t request_type,
                         std::uint32_t request_id, std::uint32_t expected_batch) {
    check_wire_platform();
    require(request_type == echo_type || request_type == mlp_type, "invalid request type");
    require(frame.type == (request_type | response_bit), "response type mismatch");
    require(frame.request_id == request_id, "response request_id mismatch");
    require(frame.body.size() >= 4 && frame.body.size() <= max_body_bytes, "invalid response length");
    const auto code = u32(frame.body.data());
    require(code <= 4, "unknown response status");
    Response result;
    result.status = static_cast<Status>(code);
    if (result.status != Status::Ok) {
        require(frame.body.size() <= 1028, "error description too long");
        result.message.assign(frame.body.begin() + 4, frame.body.end());
    } else if (request_type == echo_type) {
        require(frame.body.size() <= 1028, "echo response too long");
        result.echo.assign(frame.body.begin() + 4, frame.body.end());
    } else {
        require(frame.body.size() >= 8, "missing response batch");
        result.batch = u32(frame.body.data() + 4);
        check_batch(result.batch);
        require(expected_batch == result.batch, "response batch mismatch");
        require(frame.body.size() == 8 + std::size_t{result.batch} * 44, "MLP response length mismatch");
        result.logits.resize(std::size_t{result.batch} * 10);
        result.labels.resize(result.batch);
        for (std::size_t i = 0; i < result.logits.size(); ++i)
            result.logits[i] = read_float(frame.body.data() + 8 + i * 4);
        for (std::size_t i = 0; i < result.labels.size(); ++i) {
            result.labels[i] = u32(frame.body.data() + 8 + result.logits.size() * 4 + i * 4);
            require(result.labels[i] <= 9, "invalid response label");
        }
    }
    return result;
}
}  // namespace miniinfer::net
