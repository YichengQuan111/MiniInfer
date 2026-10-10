#pragma once

#include "miniinfer/buffer.hpp"
#include "miniinfer/tensor_view.hpp"
#include <cstddef>

namespace miniinfer {

// Mutable scratch storage belongs to one executor/worker at a time.
class Workspace final {
public:
    explicit Workspace(std::size_t max_batch = 256);
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept;
    Workspace& operator=(Workspace&&) = delete;
    std::size_t max_batch() const noexcept { return max_batch_; }
    TensorView hidden(std::size_t batch) &;
    TensorView hidden(std::size_t batch) && = delete;

private:
    std::size_t max_batch_;
    Buffer hidden_;
};

}  // namespace miniinfer
