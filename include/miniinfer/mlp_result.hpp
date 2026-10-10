#pragma once

#include "miniinfer/buffer.hpp"
#include "miniinfer/tensor_view.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace miniinfer {

class CpuExecutor;

// Independent output ownership; never borrows a worker's Workspace.
class MlpResult final {
public:
    MlpResult(const MlpResult&) = delete;
    MlpResult& operator=(const MlpResult&) = delete;
    MlpResult(MlpResult&&) noexcept = default;
    MlpResult& operator=(MlpResult&&) noexcept = default;
    std::size_t batch() const noexcept { return batch_; }
    ConstTensorView logits() const &;
    ConstTensorView logits() const && = delete;
    const std::vector<std::int32_t>& labels() const & { return labels_; }
    const std::vector<std::int32_t>& labels() const && = delete;

private:
    friend class CpuExecutor;
    explicit MlpResult(std::size_t batch);
    std::size_t batch_;
    Buffer logits_;
    std::vector<std::int32_t> labels_;
};

}  // namespace miniinfer
