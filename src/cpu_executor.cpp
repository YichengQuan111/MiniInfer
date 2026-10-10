#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/cpu_ops.hpp"

namespace miniinfer {

MlpResult CpuExecutor::run(const MlpModel& model, const MlpInput& input,
                           Workspace& workspace) const {
    input.validate(workspace.max_batch());
    // Obtain all model views before allocating a result or writing scratch data.
    const auto w1 = model.w1();
    const auto b1 = model.b1();
    const auto w2 = model.w2();
    const auto b2 = model.b2();
    auto hidden = workspace.hidden(input.batch());
    MlpResult result(input.batch());
    TensorView logits(result.logits_.data(), result.logits_.size(), input.batch(),
                      MlpModel::output_features);
    cpu_ops::linear(input.view(), w1, b1, hidden);
    cpu_ops::relu(hidden);
    cpu_ops::linear(hidden.as_const(), w2, b2, logits);
    cpu_ops::argmax(logits.as_const(), result.labels_.data(), result.labels_.size());
    return result;
}

}  // namespace miniinfer
