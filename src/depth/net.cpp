#include "miniinfer/depth/net.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>

namespace miniinfer::depth::net {
namespace transport = miniinfer::net;
namespace {
void require(bool valid, const char* reason) {
    if (!valid) throw transport::ProtocolError(reason);
}
std::uint32_t get32(const std::uint8_t* p) {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) |
           (std::uint32_t{p[2]} << 8) | p[3];
}
void put32(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value >> 24);
    p[1] = static_cast<std::uint8_t>(value >> 16);
    p[2] = static_cast<std::uint8_t>(value >> 8);
    p[3] = static_cast<std::uint8_t>(value);
}
void check_header(const Header& header) {
    require(header.type == request_type || header.type == response_type, "unknown depth frame type");
    const auto limit = header.type == request_type ? max_request_body_bytes : max_response_body_bytes;
    require(header.body_size >= metadata_bytes && header.body_size <= limit,
            "depth body size outside limits");
}
std::size_t dimensions(std::size_t width, std::size_t height) {
    try { return validate_dimensions(width, height); }
    catch (const std::invalid_argument&) { throw transport::ProtocolError("invalid depth dimensions"); }
}

// Accumulates the entire frame's transferred count and converts body EOF into
// Truncated because its header has already arrived. One deadline covers all
// chunks, preventing trickle input/output from extending the operation.
class Transfer {
public:
    Transfer(int fd, transport::Deadline deadline, const std::atomic<bool>* stop,
             transport::IoHooks* hooks = nullptr)
        : fd_(fd), deadline_(deadline), stop_(stop), hooks_(hooks) {}
    bool receive(void* data, std::size_t size) {
        return add(transport::recv_exact(fd_, data, size, deadline_, stop_, hooks_));
    }
    bool send(const void* data, std::size_t size) {
        return add(transport::send_all(fd_, data, size, deadline_, stop_, hooks_));
    }
    transport::IoResult result() const { return result_; }
private:
    bool add(transport::IoResult next) {
        if (next.status == transport::IoStatus::Eof && result_.transferred != 0)
            next.status = transport::IoStatus::Truncated;
        result_.status = next.status;
        result_.error_number = next.error_number;
        result_.transferred += next.transferred;
        return next.status == transport::IoStatus::Ok;
    }
    int fd_;
    transport::Deadline deadline_;
    const std::atomic<bool>* stop_;
    transport::IoHooks* hooks_;
    transport::IoResult result_{transport::IoStatus::Ok, 0};
};

transport::IoResult send_response(int fd, std::uint32_t id, Status status,
                                  const DepthMap* depth, transport::Deadline deadline) {
    std::array<std::uint8_t, metadata_bytes> meta{};
    put32(meta.data(), static_cast<std::uint32_t>(status));
    std::size_t pixels = 0;
    if (status == Status::Ok) {
        require(depth != nullptr, "missing successful depth response");
        pixels = dimensions(depth->width(), depth->height());
        require(depth->values().size() == pixels, "depth result count mismatch");
        validate_finite(depth->values());
        put32(meta.data() + 4, static_cast<std::uint32_t>(depth->width()));
        put32(meta.data() + 8, static_cast<std::uint32_t>(depth->height()));
        put32(meta.data() + 12, relative_f32_format);
    }
    const auto header = encode_header({response_type, id,
        static_cast<std::uint32_t>(metadata_bytes + 4 * pixels)});
    Transfer transfer(fd, deadline, nullptr);
    if (!transfer.send(header.data(), header.size()) || !transfer.send(meta.data(), meta.size()))
        return transfer.result();
    // The output vector is borrowed while this bounded stack encoding buffer is
    // sent. No full-sized encoded frame or response copy is ever allocated.
    std::array<std::uint8_t, 16384> chunk{};
    for (std::size_t offset = 0; offset < pixels;) {
        const auto count = std::min(pixels - offset, chunk.size() / 4);
        for (std::size_t index = 0; index < count; ++index) {
            std::uint32_t bits;
            const float value = depth->values()[offset + index];
            std::memcpy(&bits, &value, sizeof(bits));
            for (unsigned byte = 0; byte < 4; ++byte)
                chunk[4 * index + byte] = static_cast<std::uint8_t>(bits >> (8 * byte));
        }
        if (!transfer.send(chunk.data(), count * 4)) return transfer.result();
        offset += count;
    }
    return transfer.result();
}
}  // namespace

DepthNetError::DepthNetError(Status status, const std::string& message)
    : std::runtime_error(message), status_(status) {
    if (status != Status::Busy && status != Status::ShuttingDown && status != Status::InferenceFailed)
        throw std::invalid_argument("DepthNetError requires an SDK failure status");
}

std::array<std::uint8_t, header_bytes> encode_header(const Header& header) {
    check_header(header);
    std::array<std::uint8_t, header_bytes> bytes{'M', 'I', 'N', 'I', 0, 2};
    bytes[6] = static_cast<std::uint8_t>(header.type >> 8);
    bytes[7] = static_cast<std::uint8_t>(header.type);
    put32(bytes.data() + 8, header.request_id);
    put32(bytes.data() + 12, header.body_size);
    return bytes;
}
Header decode_header(const std::array<std::uint8_t, header_bytes>& bytes) {
    require(bytes[0] == 'M' && bytes[1] == 'I' && bytes[2] == 'N' && bytes[3] == 'I',
            "bad depth frame magic");
    require(bytes[4] == 0 && bytes[5] == 2, "unsupported depth protocol version");
    Header header{static_cast<std::uint16_t>((std::uint16_t{bytes[6]} << 8) | bytes[7]),
                  get32(bytes.data() + 8), get32(bytes.data() + 12)};
    check_header(header);
    return header;
}

struct Budget::State {
    State(std::size_t count, std::size_t bytes) : max_count(count), max_bytes(bytes) {}
    const std::size_t max_count, max_bytes;
    mutable std::mutex mutex;
    BudgetStats stats;
};
Budget::Budget(std::size_t max_inflight, std::size_t max_bytes)
    : state_(std::make_shared<State>(max_inflight, max_bytes)) {
    if (max_inflight == 0 || max_bytes == 0)
        throw std::invalid_argument("depth request budget must be positive");
}
Budget::Lease::Lease(std::shared_ptr<State> state, std::size_t bytes)
    : state_(std::move(state)), bytes_(bytes) {}
Budget::Lease::~Lease() { release(); }
Budget::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)), bytes_(other.bytes_) {}
Budget::Lease& Budget::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        release();
        state_ = std::move(other.state_);
        bytes_ = other.bytes_;
    }
    return *this;
}
void Budget::Lease::release() noexcept {
    if (!state_) return;
    // Keep State alive until its mutex guard is destroyed, even if Budget has
    // already gone away and this lease holds the final shared reference.
    auto state = std::move(state_);
    std::lock_guard<std::mutex> lock(state->mutex);
    --state->stats.inflight;
    state->stats.bytes -= bytes_;
}
std::optional<Budget::Lease> Budget::try_acquire(std::size_t pixels) {
    if (pixels == 0 || pixels > max_pixels)
        throw std::invalid_argument("depth budget pixels outside limits");
    const auto reservation = 14 * pixels + 1024 * 1024;
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto& stats = state_->stats;
    if (stats.inflight >= state_->max_count || reservation > state_->max_bytes - stats.bytes) {
        ++stats.rejected;
        return std::nullopt;
    }
    ++stats.inflight;
    stats.bytes += reservation;
    stats.peak_inflight = std::max(stats.peak_inflight, stats.inflight);
    stats.peak_bytes = std::max(stats.peak_bytes, stats.bytes);
    return Lease(state_, reservation);
}
BudgetStats Budget::snapshot() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->stats;
}

transport::IoResult send_request(int fd, std::uint32_t request_id, const RgbImage& image,
                                 transport::Deadline deadline, const std::atomic<bool>* stop,
                                 transport::IoHooks* hooks) {
    transport::check_wire_platform();
    const auto pixels = dimensions(image.width(), image.height());
    require(image.pixels().size() == 3 * pixels, "RGB count mismatch");
    const auto header = encode_header({request_type, request_id,
        static_cast<std::uint32_t>(metadata_bytes + 3 * pixels)});
    std::array<std::uint8_t, metadata_bytes> meta{};
    put32(meta.data(), static_cast<std::uint32_t>(image.width()));
    put32(meta.data() + 4, static_cast<std::uint32_t>(image.height()));
    put32(meta.data() + 8, 3);
    put32(meta.data() + 12, rgb8_format);
    Transfer transfer(fd, deadline, stop, hooks);
    if (transfer.send(header.data(), header.size()) && transfer.send(meta.data(), meta.size()))
        (void)transfer.send(image.pixels().data(), image.pixels().size());
    return transfer.result();
}

transport::IoResult recv_response(int fd, Response& response, std::uint32_t request_id,
                                  std::size_t expected_width, std::size_t expected_height,
                                  transport::Deadline deadline, const std::atomic<bool>* stop,
                                  transport::IoHooks* hooks) {
    transport::check_wire_platform();
    const auto expected_pixels = dimensions(expected_width, expected_height);
    response.depth.reset();
    response.status = Status::InferenceFailed;
    Transfer transfer(fd, deadline, stop, hooks);
    std::array<std::uint8_t, header_bytes> bytes{};
    if (!transfer.receive(bytes.data(), bytes.size())) return transfer.result();
    const auto header = decode_header(bytes);
    require(header.type == response_type && header.request_id == request_id,
            "unexpected depth response type or request_id");
    std::array<std::uint8_t, metadata_bytes> meta{};
    if (!transfer.receive(meta.data(), meta.size())) return transfer.result();
    const auto code = get32(meta.data());
    require(code <= static_cast<std::uint32_t>(Status::InferenceFailed), "unknown depth response status");
    const auto status = static_cast<Status>(code);
    require(get32(meta.data() + 16) == 0, "depth response reserved field must be zero");
    if (status != Status::Ok) {
        require(header.body_size == metadata_bytes && get32(meta.data() + 4) == 0 &&
                get32(meta.data() + 8) == 0 && get32(meta.data() + 12) == 0,
                "invalid depth error response shape");
        response.status = status;
        return transfer.result();
    }
    require(get32(meta.data() + 4) == expected_width && get32(meta.data() + 8) == expected_height &&
            get32(meta.data() + 12) == relative_f32_format,
            "depth response shape or format mismatch");
    require(header.body_size == metadata_bytes + 4 * expected_pixels, "depth response length mismatch");
    std::vector<float> values(expected_pixels);
    std::array<std::uint8_t, 16384> chunk{};
    for (std::size_t offset = 0; offset < expected_pixels;) {
        const auto count = std::min(expected_pixels - offset, chunk.size() / 4);
        if (!transfer.receive(chunk.data(), 4 * count)) return transfer.result();
        for (std::size_t index = 0; index < count; ++index) {
            std::uint32_t bits = 0;
            for (unsigned byte = 0; byte < 4; ++byte)
                bits |= std::uint32_t{chunk[4 * index + byte]} << (8 * byte);
            std::memcpy(&values[offset + index], &bits, sizeof(bits));
            require(std::isfinite(values[offset + index]), "non-finite depth response value");
        }
        offset += count;
    }
    response.depth.emplace(expected_width, expected_height, std::move(values));
    response.status = Status::Ok;
    return transfer.result();
}

transport::SessionStats serve_connection(int fd, const RunDepth& run, Budget& budget,
                                        const transport::SessionConfig& config,
                                        const std::atomic<bool>* stop) {
    if (!run || config.read_timeout.count() <= 0 || config.write_timeout.count() <= 0)
        throw std::invalid_argument("depth session requires callback and positive timeouts");
    transport::check_wire_platform();
    transport::SessionStats stats;
    std::optional<std::uint32_t> previous_id;
    const auto record = [&](const transport::IoResult& io) {
        stats.io_status = io.status;
        stats.error_number = io.error_number;
        if (io.status != transport::IoStatus::Ok && io.status != transport::IoStatus::Eof &&
            io.status != transport::IoStatus::Stopped)
            stats.diagnostic = "depth socket I/O failed";
    };
    const auto reply = [&](std::uint32_t id, Status status, const DepthMap* result = nullptr) {
        const auto io = send_response(fd, id, status, result, transport::Clock::now() + config.write_timeout);
        record(io);
        if (io.status == transport::IoStatus::Ok) { ++stats.responses; return true; }
        return false;
    };
    for (;;) {
        Transfer transfer(fd, transport::Clock::now() + config.read_timeout, stop);
        std::array<std::uint8_t, header_bytes> bytes{};
        if (!transfer.receive(bytes.data(), bytes.size())) { record(transfer.result()); return stats; }
        Header header{};
        try {
            header = decode_header(bytes);
            require(header.type == request_type, "expected depth request direction");
        } catch (const transport::ProtocolError& error) {
            stats.protocol_error = true;
            ++stats.bad_requests;
            stats.diagnostic = error.what();
            return stats;
        }
        ++stats.requests;
        const auto reject = [&](const char* reason) {
            stats.protocol_error = true;
            ++stats.bad_requests;
            stats.diagnostic = reason;
            (void)reply(header.request_id, Status::BadRequest);
        };
        if (previous_id && header.request_id <= *previous_id) {
            reject("depth request_id must strictly increase");
            return stats;
        }
        previous_id = header.request_id;
        std::array<std::uint8_t, metadata_bytes> meta{};
        if (!transfer.receive(meta.data(), meta.size())) { record(transfer.result()); return stats; }
        const auto width = get32(meta.data()), height = get32(meta.data() + 4);
        std::size_t pixels;
        try {
            pixels = dimensions(width, height);
            require(get32(meta.data() + 8) == 3 && get32(meta.data() + 12) == rgb8_format &&
                    get32(meta.data() + 16) == 0, "invalid RGB metadata");
            require(header.body_size == metadata_bytes + 3 * pixels, "RGB request length mismatch");
        } catch (const transport::ProtocolError& error) {
            reject(error.what());
            return stats;
        }
        auto lease = budget.try_acquire(pixels);
        if (!lease) {
            ++stats.busy;
            (void)reply(header.request_id, Status::Busy);
            return stats;
        }
        // The vector is moved into the owning RgbImage: no receive-to-image copy.
        std::vector<std::uint8_t> rgb(3 * pixels);
        if (!transfer.receive(rgb.data(), rgb.size())) { record(transfer.result()); return stats; }
        RgbImage image(width, height, std::move(rgb));
        if (stop && stop->load(std::memory_order_relaxed)) {
            (void)reply(header.request_id, Status::ShuttingDown);
            return stats;
        }
        std::optional<DepthMap> result;
        Status status = Status::Ok;
        try {
            result.emplace(run(image));
            require(result->width() == width && result->height() == height &&
                    result->values().size() == pixels, "backend depth shape mismatch");
            validate_finite(result->values());
        } catch (const DepthNetError& error) {
            status = error.status();
            if (status == Status::Busy) ++stats.busy;
            if (status == Status::InferenceFailed) ++stats.inference_failures;
            stats.diagnostic = error.what();
        } catch (const std::exception& error) {
            status = Status::InferenceFailed;
            ++stats.inference_failures;
            stats.diagnostic = error.what();
        } catch (...) {
            status = Status::InferenceFailed;
            ++stats.inference_failures;
            stats.diagnostic = "depth inference failed with non-standard exception";
        }
        if (!reply(header.request_id, status, status == Status::Ok ? &*result : nullptr) ||
            status == Status::ShuttingDown) return stats;
        // lease releases at iteration end, after the response transfer.
    }
}
}  // namespace miniinfer::depth::net
