#include "miniinfer/workspace.hpp"
#include "miniinfer/checked_size.hpp"
#include "miniinfer/mlp_model.hpp"
#include <stdexcept>
#include <utility>

namespace miniinfer {
namespace {
std::size_t workspace_elements(std::size_t max_batch) {
    if (max_batch == 0) {
        throw std::invalid_argument("Workspace max_batch must be positive");
    }
    return checked_multiply(max_batch, MlpModel::hidden_features);
}
}  // namespace

Workspace::Workspace(std::size_t max_batch)
    : max_batch_(max_batch), hidden_(workspace_elements(max_batch)) {}

Workspace::Workspace(Workspace&& other) noexcept
    : max_batch_(std::exchange(other.max_batch_, 0)), hidden_(std::move(other.hidden_)) {}

TensorView Workspace::hidden(std::size_t batch) & {
    if (batch == 0 || batch > max_batch_) {
        throw std::invalid_argument("Workspace batch must be in [1,max_batch]");
    }
    return {hidden_.data(), hidden_.size(), batch, MlpModel::hidden_features};
}

}  // namespace miniinfer
