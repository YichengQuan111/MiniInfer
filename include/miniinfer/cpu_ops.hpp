#pragma once

#include "miniinfer/tensor_view.hpp"

#include <cstddef>
#include <cstdint>

namespace miniinfer::cpu_ops {

void linear(ConstTensorView input, ConstTensorView weights, ConstTensorView bias,
            TensorView output);
void relu(TensorView values);
void argmax(ConstTensorView logits, std::int32_t* labels,
            std::size_t available_labels);

}  // namespace miniinfer::cpu_ops
