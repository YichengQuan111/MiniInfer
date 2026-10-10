#include "miniinfer/cpu_ops.hpp"

#include "miniinfer/checked_size.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace miniinfer::cpu_ops {
namespace {

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

AddressRange checked_range(const void* data, std::size_t byte_count) {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
    const std::uintmax_t available_address_space =
        static_cast<std::uintmax_t>(std::numeric_limits<std::uintptr_t>::max() - begin);
    if (static_cast<std::uintmax_t>(byte_count) > available_address_space) {
        throw std::invalid_argument("tensor memory range overflows uintptr_t");
    }
    return {begin, begin + static_cast<std::uintptr_t>(byte_count)};
}

bool overlaps(AddressRange lhs, AddressRange rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

void require_finite(ConstTensorView values, const char* name) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values.data()[i])) {
            throw std::invalid_argument(name);
        }
    }
}

}  // namespace

void linear(ConstTensorView input, ConstTensorView weights, ConstTensorView bias,
            TensorView output) {
    if (weights.cols() != input.cols() || bias.rows() != 1 ||
        bias.cols() != weights.rows() || output.rows() != input.rows() ||
        output.cols() != weights.rows()) {
        throw std::invalid_argument("linear tensor shapes are incompatible");
    }

    const AddressRange input_range = checked_range(input.data(), input.size_bytes());
    const AddressRange weights_range = checked_range(weights.data(), weights.size_bytes());
    const AddressRange bias_range = checked_range(bias.data(), bias.size_bytes());
    const AddressRange output_range = checked_range(output.data(), output.size_bytes());
    if (overlaps(output_range, input_range) || overlaps(output_range, weights_range) ||
        overlaps(output_range, bias_range)) {
        throw std::invalid_argument("linear output must not overlap an input");
    }

    require_finite(input, "linear input contains a non-finite value");
    require_finite(weights, "linear weights contain a non-finite value");
    require_finite(bias, "linear bias contains a non-finite value");

    for (std::size_t batch = 0; batch < input.rows(); ++batch) {
        for (std::size_t out = 0; out < weights.rows(); ++out) {
            float sum = 0.0F;
            for (std::size_t in = 0; in < input.cols(); ++in) {
                sum += input.data()[batch * input.cols() + in] *
                       weights.data()[out * weights.cols() + in];
            }
            const float result = sum + bias.data()[out];
            if (!std::isfinite(result)) {
                throw std::overflow_error("linear result is not finite");
            }
            output.data()[batch * output.cols() + out] = result;
        }
    }
}

void relu(TensorView values) {
    const AddressRange values_range = checked_range(values.data(), values.size_bytes());
    (void)values_range;

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values.data()[i])) {
            throw std::invalid_argument("ReLU input contains a non-finite value");
        }
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        const float value = values.data()[i];
        values.data()[i] = value > 0.0F ? value : 0.0F;
    }
}

void argmax(ConstTensorView logits, std::int32_t* labels,
            std::size_t available_labels) {
    if (logits.cols() > static_cast<std::size_t>(
                            std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("argmax class count exceeds int32_t range");
    }
    if (labels == nullptr) {
        throw std::invalid_argument("argmax labels must not be null");
    }
    if (reinterpret_cast<std::uintptr_t>(labels) % alignof(std::int32_t) != 0) {
        throw std::invalid_argument("argmax labels are not aligned for int32_t");
    }
    if (available_labels < logits.rows()) {
        throw std::invalid_argument("argmax label capacity is smaller than batch size");
    }

    const std::size_t labels_bytes = checked_multiply(logits.rows(), sizeof(std::int32_t));
    const AddressRange logits_range = checked_range(logits.data(), logits.size_bytes());
    const AddressRange labels_range = checked_range(labels, labels_bytes);
    if (overlaps(logits_range, labels_range)) {
        throw std::invalid_argument("argmax labels must not overlap logits");
    }

    require_finite(logits, "argmax logits contain a non-finite value");

    for (std::size_t row = 0; row < logits.rows(); ++row) {
        const std::size_t row_offset = row * logits.cols();
        std::size_t best_index = 0;
        float best_value = logits.data()[row_offset];
        for (std::size_t col = 1; col < logits.cols(); ++col) {
            const float value = logits.data()[row_offset + col];
            if (value > best_value) {
                best_value = value;
                best_index = col;
            }
        }
        labels[row] = static_cast<std::int32_t>(best_index);
    }
}

}  // namespace miniinfer::cpu_ops
