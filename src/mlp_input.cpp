#include "miniinfer/mlp_input.hpp"
#include "miniinfer/checked_size.hpp"
#include "miniinfer/mlp_model.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace miniinfer {

MlpInput::MlpInput(std::size_t batch, std::vector<float> values)
    : batch_(batch), values_(std::move(values)) {
    validate(std::numeric_limits<std::size_t>::max());
}

void MlpInput::validate(std::size_t max_batch) const {
    if (batch_ == 0 || batch_ > max_batch) {
        throw std::invalid_argument("MLP input batch must be in [1,max_batch]");
    }
    if (values_.size() != checked_multiply(batch_, MlpModel::input_features)) {
        throw std::invalid_argument("MLP input must contain exactly batch*64 elements");
    }
    for (float value : values_) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("MLP input must contain finite FP32 values");
        }
    }
}

ConstTensorView MlpInput::view() const & {
    return {values_.data(), values_.size(), batch_, MlpModel::input_features};
}

}  // namespace miniinfer
