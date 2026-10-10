#include "miniinfer/mlp_result.hpp"
#include "miniinfer/checked_size.hpp"
#include "miniinfer/mlp_model.hpp"

namespace miniinfer {

MlpResult::MlpResult(std::size_t batch)
    : batch_(batch), logits_(checked_multiply(batch, MlpModel::output_features)), labels_(batch) {}

ConstTensorView MlpResult::logits() const & {
    return {logits_.data(), logits_.size(), batch_, MlpModel::output_features};
}

}  // namespace miniinfer
