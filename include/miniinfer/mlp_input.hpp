#pragma once

#include "miniinfer/tensor_view.hpp"
#include <cstddef>
#include <vector>

namespace miniinfer {

// Owns normalized [batch,64] FP32 input. Copying makes an independent request.
class MlpInput final {
public:
    MlpInput(std::size_t batch, std::vector<float> values);
    std::size_t batch() const noexcept { return batch_; }
    void validate(std::size_t max_batch) const;
    ConstTensorView view() const &;
    ConstTensorView view() const && = delete;

private:
    std::size_t batch_;
    std::vector<float> values_;
};

}  // namespace miniinfer
