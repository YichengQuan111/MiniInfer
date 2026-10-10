#pragma once

#include "miniinfer/checked_size.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace miniinfer {

// A non-owning view over a contiguous, row-major two-dimensional FP32 array.
// The caller keeps the storage alive and supplies its accessible capacity.
template <class T>
class BasicTensorView final {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, const float>,
                  "BasicTensorView supports float and const float only");

public:
    using value_type = T;

    BasicTensorView() = delete;

    BasicTensorView(T* data, std::size_t available_elements, std::size_t rows,
                    std::size_t cols)
        : data_(data),
          available_elements_(available_elements),
          rows_(rows),
          cols_(cols),
          size_(0),
          size_bytes_(0) {
        if (rows_ == 0 || cols_ == 0) {
            throw std::invalid_argument("tensor view dimensions must be nonzero");
        }
        if (data_ == nullptr) {
            throw std::invalid_argument("tensor view data must not be null");
        }
        if (reinterpret_cast<std::uintptr_t>(data_) % alignof(float) != 0) {
            throw std::invalid_argument("tensor view data must be aligned for float");
        }

        size_ = checked_multiply(rows_, cols_);
        size_bytes_ = checked_multiply(size_, sizeof(float));
        if (size_bytes_ > static_cast<std::size_t>(
                              std::numeric_limits<std::ptrdiff_t>::max())) {
            throw std::length_error("tensor view exceeds PTRDIFF_MAX bytes");
        }
        if (available_elements_ < size_) {
            throw std::invalid_argument("tensor view capacity is smaller than its shape");
        }
    }

    T* data() const noexcept { return data_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }

    T* row(std::size_t r) const {
        if (r >= rows_) {
            throw std::out_of_range("tensor view row index is out of range");
        }
        return data_ + r * cols_;
    }

    T& at(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("tensor view index is out of range");
        }
        return data_[r * cols_ + c];
    }

    BasicTensorView<const float> as_const() const {
        return BasicTensorView<const float>(data_, available_elements_, rows_, cols_);
    }

private:
    T* data_;
    std::size_t available_elements_;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t size_;
    std::size_t size_bytes_;
};

using TensorView = BasicTensorView<float>;
using ConstTensorView = BasicTensorView<const float>;

}  // namespace miniinfer
