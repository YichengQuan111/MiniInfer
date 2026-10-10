#pragma once

#include <cstddef>
#include <memory>

namespace miniinfer {

// Fixed-size, uniquely owned CPU FP32 storage, independent of tensor shape.
// data() borrows memory from the current owner; callers must not free it.
class Buffer final {
public:
    Buffer() noexcept = default;
    // Zero elements produce an empty buffer. Non-empty storage starts at zero.
    // Size errors throw before allocation; allocation failures propagate.
    explicit Buffer(std::size_t element_count);
    ~Buffer() = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    // Moves transfer ownership and leave the source empty; self-move is safe.
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    float* data() noexcept;
    const float* data() const noexcept;
    std::size_t size() const noexcept;  // Capacity measured in float elements.
    std::size_t size_bytes() const noexcept;
    bool empty() const noexcept;

private:
    std::unique_ptr<float[]> data_;
    std::size_t size_ = 0;
};

}  // namespace miniinfer
