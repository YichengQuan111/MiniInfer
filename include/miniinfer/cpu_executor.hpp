#pragma once

#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/workspace.hpp"

namespace miniinfer {

// Stateless execution; model/input are read-only and Workspace is exclusive.
class CpuExecutor final {
public:
    MlpResult run(const MlpModel& model, const MlpInput& input, Workspace& workspace) const;
};

}  // namespace miniinfer
