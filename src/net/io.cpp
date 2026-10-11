#include "miniinfer/net/io.hpp"

#include <sys/socket.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>

namespace miniinfer::net {
namespace {
Deadline now(IoHooks* hooks) {
    return hooks && hooks->now ? hooks->now() : Clock::now();
}
IoResult transfer(int fd, void* receive_data, const void* send_data, std::size_t size,
                  Deadline deadline, const std::atomic<bool>* stop, IoHooks* hooks,
                  bool receiving) {
    std::size_t offset = 0;
    // An empty transfer is complete even if cancellation/deadline is set.
    while (offset < size) {
        if (stop && stop->load(std::memory_order_relaxed))
            return {IoStatus::Stopped, offset};
        if (now(hooks) >= deadline) return {IoStatus::Timeout, offset};
        ssize_t count;
        if (receiving) {
            auto* bytes = static_cast<char*>(receive_data) + offset;
            count = hooks && hooks->recv
                        ? hooks->recv(fd, bytes, size - offset, MSG_DONTWAIT)
                        : ::recv(fd, bytes, size - offset, MSG_DONTWAIT);
        } else {
            const auto* bytes = static_cast<const char*>(send_data) + offset;
            constexpr int flags = MSG_DONTWAIT | MSG_NOSIGNAL;
            count = hooks && hooks->send
                        ? hooks->send(fd, bytes, size - offset, flags)
                        : ::send(fd, bytes, size - offset, flags);
        }
        if (count > 0) {
            if (static_cast<std::size_t>(count) > size - offset)
                return {IoStatus::SystemError, offset, EIO};
            offset += static_cast<std::size_t>(count);
            continue;
        }
        if (count == 0) {
            if (receiving) return {offset == 0 ? IoStatus::Eof : IoStatus::Truncated, offset};
            return {IoStatus::SystemError, offset, EIO};
        }
        const int error = errno;
        if (error == EINTR) continue;
        if (error != EAGAIN && error != EWOULDBLOCK)
            return {IoStatus::SystemError, offset, error};

        if (stop && stop->load(std::memory_order_relaxed))
            return {IoStatus::Stopped, offset};
        const auto current = now(hooks);
        if (current >= deadline) return {IoStatus::Timeout, offset};
        // Round up so a sub-millisecond remainder does not busy-spin. The cap
        // bounds cancellation latency even when the deadline is far away.
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - current);
        const int wait_ms = static_cast<int>(std::min<std::int64_t>(20, remaining.count()));
        pollfd pending{fd, static_cast<short>(receiving ? POLLIN : POLLOUT), 0};
        const int ready = hooks && hooks->poll ? hooks->poll(&pending, 1, wait_ms)
                                             : ::poll(&pending, 1, wait_ms);
        if (ready < 0 && errno != EINTR)
            return {IoStatus::SystemError, offset, errno};
        if (ready > 0 && (pending.revents & POLLNVAL))
            return {IoStatus::SystemError, offset, EBADF};
        // HUP/ERR may accompany readable bytes. Let recv/send report the
        // definitive EOF/error on the next iteration, after checking time/stop.
    }
    return {IoStatus::Ok, offset};
}
}  // namespace

IoResult recv_exact(int fd, void* data, std::size_t size, Deadline deadline,
                    const std::atomic<bool>* stop, IoHooks* hooks) {
    return transfer(fd, data, nullptr, size, deadline, stop, hooks, true);
}
IoResult send_all(int fd, const void* data, std::size_t size, Deadline deadline,
                  const std::atomic<bool>* stop, IoHooks* hooks) {
    return transfer(fd, nullptr, data, size, deadline, stop, hooks, false);
}
}  // namespace miniinfer::net
