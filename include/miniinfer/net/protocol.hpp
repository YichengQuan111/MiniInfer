#pragma once

#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/net/io.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace miniinfer::net {

constexpr std::size_t header_bytes = 16;
constexpr std::size_t max_body_bytes = 65540;
constexpr std::uint16_t echo_type = 1, mlp_type = 2, response_bit = 0x8000;
enum class Status : std::uint32_t {
    Ok = 0, BadRequest = 1, Busy = 2, ShuttingDown = 3, InferenceFailed = 4
};
class ProtocolError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct Header {
    std::uint16_t type;
    std::uint32_t request_id;
    std::uint32_t body_size;
};
struct Frame {
    std::uint16_t type = 0;
    std::uint32_t request_id = 0;
    std::vector<std::uint8_t> body;
};
// Every decoded result owns its storage, including after the socket closes.
struct Response {
    Status status = Status::Ok;
    std::string message;
    std::uint32_t batch = 0;
    std::vector<float> logits;
    std::vector<std::uint32_t> labels;
    std::vector<std::uint8_t> echo;
};

void check_wire_platform();
std::array<std::uint8_t, header_bytes> encode_header(const Header& header);
Header decode_header(const std::array<std::uint8_t, header_bytes>& bytes);
std::vector<std::uint8_t> encode_frame(const Frame& frame);
// io.transferred includes the header. EOF after any header byte is Truncated.
IoResult recv_frame(int fd, Frame& frame, Deadline deadline,
                    const std::atomic<bool>* stop = nullptr, IoHooks* hooks = nullptr);
IoResult send_frame(int fd, const Frame& frame, Deadline deadline,
                    const std::atomic<bool>* stop = nullptr, IoHooks* hooks = nullptr);
Frame echo_request(std::uint32_t id, const std::vector<std::uint8_t>& bytes);
Frame mlp_request(std::uint32_t id, const MlpInput& input);
MlpInput decode_mlp_request(const Frame& frame);
Frame echo_response(const Frame& request);
Frame mlp_response(std::uint32_t id, const MlpResult& result);
Frame error_response(const Frame& request, Status status, const std::string& message);
Response decode_response(const Frame& frame, std::uint16_t request_type,
                         std::uint32_t request_id, std::uint32_t expected_batch = 0);

}  // namespace miniinfer::net
