#include "miniinfer/buffer.hpp"

#include "miniinfer/checked_size.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace miniinfer {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "MiniInfer requires IEEE 754 32-bit float storage");

Buffer::Buffer(std::size_t element_count) {
    const auto bytes = checked_multiply(element_count, sizeof(float));
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("Buffer byte size exceeds PTRDIFF_MAX");
    }
    if (element_count != 0) {
        data_ = std::make_unique<float[]>(element_count);
    }
    size_ = element_count;
}

Buffer::Buffer(Buffer&& other) noexcept
    : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

float* Buffer::data() noexcept {
    return data_.get();
}

const float* Buffer::data() const noexcept {
    return data_.get();
}

std::size_t Buffer::size() const noexcept {
    return size_;
}

std::size_t Buffer::size_bytes() const noexcept {
    // The constructor already validated this product; moves preserve the bound.
    return size_ * sizeof(float);
}

bool Buffer::empty() const noexcept {
    return size_ == 0;
}

}  // namespace miniinfer
