#pragma once

#include "miniinfer/depth/image.hpp"
#include "miniinfer/net/session.hpp"

#include <array>
#include <memory>
#include <optional>

namespace miniinfer::depth::net {

using Status = miniinfer::net::Status;
using RunDepth = std::function<DepthMap(const RgbImage&)>;
inline constexpr std::size_t header_bytes = 16, metadata_bytes = 20;
inline constexpr std::uint16_t request_type = 3, response_type = 0x8003;
inline constexpr std::uint32_t rgb8_format = 1, relative_f32_format = 1;
inline constexpr std::size_t max_request_body_bytes = metadata_bytes + 3 * max_pixels;
inline constexpr std::size_t max_response_body_bytes = metadata_bytes + 4 * max_pixels;

// Adapters use Busy / ShuttingDown for SDK admission failures; other exceptions
// become InferenceFailed. The wire never includes exception text.
class DepthNetError : public std::runtime_error {
public:
    DepthNetError(Status status, const std::string& message);
    Status status() const noexcept { return status_; }
private:
    Status status_;
};

struct Header {
    std::uint16_t type;
    std::uint32_t request_id, body_size;
};
std::array<std::uint8_t, header_bytes> encode_header(const Header& header);
Header decode_header(const std::array<std::uint8_t, header_bytes>& bytes);

struct BudgetStats {
    std::size_t inflight = 0, bytes = 0;
    std::size_t peak_inflight = 0, peak_bytes = 0, rejected = 0;
};

// Share a single Budget across every session served by a server. Admission is
// nonblocking and jointly limits count and memory. A lease remains charged
// through receive, execution and the final write attempt. The reservation
// covers transport-owned image/result/encoding storage, not backend tensors.
class Budget final {
    struct State;
public:
    class Lease final {
    public:
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        friend class Budget;
        Lease(std::shared_ptr<State> state, std::size_t bytes);
        void release() noexcept;
        std::shared_ptr<State> state_;
        std::size_t bytes_;
    };
    explicit Budget(std::size_t max_inflight = 4,
                    std::size_t max_bytes = 128 * 1024 * 1024);
    Budget(const Budget&) = delete;
    Budget& operator=(const Budget&) = delete;
    std::optional<Lease> try_acquire(std::size_t pixels);
    BudgetStats snapshot() const;
private:
    std::shared_ptr<State> state_;
};

struct Response {
    Status status = Status::InferenceFailed;
    std::optional<DepthMap> depth;
};

// MINI v2 headers and metadata are big endian; the successful response's HW
// values are little-endian IEEE FP32 relative depth (larger means nearer).
// A response owns its storage. Partial receive clears depth and reports I/O.
miniinfer::net::IoResult send_request(
    int fd, std::uint32_t request_id, const RgbImage& image,
    miniinfer::net::Deadline deadline, const std::atomic<bool>* stop = nullptr,
    miniinfer::net::IoHooks* hooks = nullptr);
miniinfer::net::IoResult recv_response(
    int fd, Response& response, std::uint32_t request_id,
    std::size_t expected_width, std::size_t expected_height,
    miniinfer::net::Deadline deadline, const std::atomic<bool>* stop = nullptr,
    miniinfer::net::IoHooks* hooks = nullptr);

// Borrows fd and follows the transport SessionConfig cancellation/deadlines.
// Invalid frames close the session. If admission fails after metadata, Busy is
// sent without consuming the RGB payload, then the connection is closed: the
// client must reconnect and must not reuse that connection. Accepted execution
// completes despite disconnect/stop and gets a write-timeout-bounded attempt.
miniinfer::net::SessionStats serve_connection(
    int fd, const RunDepth& run, Budget& budget,
    const miniinfer::net::SessionConfig& config = {},
    const std::atomic<bool>* stop = nullptr);

}  // namespace miniinfer::depth::net
