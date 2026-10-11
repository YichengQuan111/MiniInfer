#pragma once

#include <unistd.h>

namespace miniinfer::net {

// A descriptor has one owner. On Linux close(EINTR) must not be retried: the
// descriptor number has already been released and might have been reused.
class UniqueFd {
public:
    UniqueFd() noexcept = default;
    explicit UniqueFd(int fd) noexcept : fd_(fd) {}
    ~UniqueFd() { reset(); }
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    UniqueFd& operator=(UniqueFd&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
    int get() const noexcept { return fd_; }
    explicit operator bool() const noexcept { return fd_ >= 0; }
    void reset(int fd = -1) noexcept {
        if (fd_ == fd) return;
        const int old = fd_;
        fd_ = fd;
        if (old >= 0) (void)::close(old);
    }

private:
    int fd_ = -1;
};

}  // namespace miniinfer::net
