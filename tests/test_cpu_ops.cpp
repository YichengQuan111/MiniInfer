#include "miniinfer/cpu_ops.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using miniinfer::ConstTensorView;
using miniinfer::TensorView;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable, std::string_view message) {
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

void expect_near(float actual, float expected, std::string_view message) {
    if (std::fabs(actual - expected) > 1.0e-6F) {
        throw std::runtime_error(std::string(message));
    }
}

void test_linear_rectangular_multi_batch_and_bias() {
    const float input_data[] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 2.0F};
    const float weights_data[] = {2.0F, -1.0F, 0.5F, -3.0F, 4.0F, 1.0F};
    const float bias_data[] = {0.25F, -2.0F};
    float output_data[] = {-99.0F, -99.0F, -99.0F, -99.0F};
    const ConstTensorView input(input_data, 6, 2, 3);
    const ConstTensorView weights(weights_data, 6, 2, 3);
    const ConstTensorView bias(bias_data, 2, 1, 2);
    const TensorView output(output_data, 4, 2, 2);

    miniinfer::cpu_ops::linear(input, weights, bias, output);

    expect_near(output_data[0], 1.75F, "linear first batch/output must include bias");
    expect_near(output_data[1], 6.0F, "linear first batch must use the second weight row");
    expect_near(output_data[2], -1.25F, "linear second batch must use its own input row");
    expect_near(output_data[3], 5.0F, "linear second batch/output must include bias");
}

void test_relu_negative_zero_and_positive_values() {
    float values_data[] = {-3.0F, 0.0F, 2.5F, -0.25F, -0.0F};
    const TensorView values(values_data, 5, 1, 5);

    miniinfer::cpu_ops::relu(values);

    expect(values_data[0] == 0.0F && !std::signbit(values_data[0]),
           "ReLU must turn a negative value into positive zero");
    expect(values_data[1] == 0.0F, "ReLU must preserve zero as zero");
    expect(values_data[2] == 2.5F, "ReLU must preserve positive values");
    expect(values_data[3] == 0.0F, "ReLU must turn a fractional negative into zero");
    expect(values_data[4] == 0.0F && !std::signbit(values_data[4]),
           "ReLU max(x, 0) must normalize negative zero");
}

void test_argmax_ties_choose_lowest_index() {
    const float logits_data[] = {1.0F, 3.0F, 3.0F, -2.0F,
                                 5.0F, -1.0F, 4.0F, 5.0F,
                                 -4.0F, -3.0F, -2.0F, -1.0F};
    std::int32_t labels[] = {-1, -1, -1, 77};
    const ConstTensorView logits(logits_data, 12, 3, 4);

    miniinfer::cpu_ops::argmax(logits, labels, 4);

    expect(labels[0] == 1, "argmax must pick the first class among tied maxima");
    expect(labels[1] == 0, "argmax must pick the first class among tied row maxima");
    expect(labels[2] == 3, "argmax must select the unique maximum");
    expect(labels[3] == 77, "argmax must not write beyond the batch labels");
}

void test_linear_shape_alias_and_non_finite_validation() {
    const float input_data[] = {1.0F, 2.0F};
    const float weights_data[] = {3.0F, 4.0F};
    const float bias_data[] = {1.0F};
    float output_data[] = {-9.0F, -9.0F};
    const ConstTensorView input(input_data, 2, 1, 2);
    const ConstTensorView weights(weights_data, 2, 1, 2);
    const ConstTensorView bias(bias_data, 1, 1, 1);
    const TensorView wrong_shape_output(output_data, 2, 2, 1);

    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::linear(input, weights, bias, wrong_shape_output); },
        "linear must reject incompatible output shape");

    const TensorView aliased_output(const_cast<float*>(input_data), 1, 1, 1);
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::linear(input, weights, bias, aliased_output); },
        "linear must reject output overlapping an input");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float non_finite_bias_data[] = {nan};
    const ConstTensorView non_finite_bias(non_finite_bias_data, 1, 1, 1);
    float untouched[] = {123.0F};
    const TensorView one_output(untouched, 1, 1, 1);
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::linear(input, weights, non_finite_bias, one_output); },
        "linear must reject non-finite bias before computation");
    expect(untouched[0] == 123.0F,
           "linear validation must finish before writing any output");

    const float huge_input_data[] = {std::numeric_limits<float>::max()};
    const float huge_weight_data[] = {2.0F};
    const float zero_bias_data[] = {0.0F};
    const ConstTensorView huge_input(huge_input_data, 1, 1, 1);
    const ConstTensorView huge_weight(huge_weight_data, 1, 1, 1);
    const ConstTensorView zero_bias(zero_bias_data, 1, 1, 1);
    float overflow_output_data[] = {-17.0F};
    const TensorView overflow_output(overflow_output_data, 1, 1, 1);
    expect_throws<std::overflow_error>(
        [&] {
            miniinfer::cpu_ops::linear(huge_input, huge_weight, zero_bias,
                                       overflow_output);
        },
        "linear must reject a non-finite computed result");
    expect(overflow_output_data[0] == -17.0F,
           "linear must check each result before writing that output element");
}

void test_relu_rejects_non_finite_before_writing() {
    const float inf = std::numeric_limits<float>::infinity();
    float values_data[] = {-1.0F, inf};
    const TensorView values(values_data, 2, 1, 2);

    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::relu(values); },
        "ReLU must reject infinity");
    expect(values_data[0] == -1.0F,
           "ReLU must validate all elements before modifying the tensor");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    float nan_values_data[] = {-2.0F, nan};
    const TensorView nan_values(nan_values_data, 2, 1, 2);
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::relu(nan_values); },
        "ReLU must reject NaN");
    expect(nan_values_data[0] == -2.0F,
           "ReLU must reject NaN before modifying earlier elements");
}

void test_argmax_shape_alias_finite_and_capacity_validation() {
    const float logits_data[] = {0.0F, 1.0F, 2.0F, 3.0F};
    const ConstTensorView logits(logits_data, 4, 2, 2);
    std::int32_t too_short[] = {-7};
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(logits, too_short, 1); },
        "argmax must reject insufficient output capacity");
    expect(too_short[0] == -7, "argmax must not write when capacity is too small");
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(logits, nullptr, 2); },
        "argmax must reject a null labels pointer");

    alignas(std::int32_t) unsigned char misaligned_storage[sizeof(std::int32_t) + 1]{};
    auto* misaligned_labels = reinterpret_cast<std::int32_t*>(misaligned_storage + 1);
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(logits, misaligned_labels, 2); },
        "argmax must reject a misaligned labels pointer");

    auto* aliased_labels = reinterpret_cast<std::int32_t*>(
        const_cast<float*>(logits_data));
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(logits, aliased_labels, 2); },
        "argmax must reject labels that overlap logits");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float non_finite_data[] = {4.0F, 1.0F, 2.0F, nan};
    const ConstTensorView non_finite_logits(non_finite_data, 4, 2, 2);
    std::int32_t untouched[] = {33, 44};
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(non_finite_logits, untouched, 2); },
        "argmax must reject non-finite logits");
    expect(untouched[0] == 33 && untouched[1] == 44,
           "argmax must validate every row before writing labels");
}

void test_pointer_range_overflow_is_rejected_before_access() {
    static_assert(sizeof(float) == sizeof(std::int32_t),
                  "MiniInfer CPU ops require 32-bit float and labels");
    const std::uintptr_t last_aligned_start =
        std::numeric_limits<std::uintptr_t>::max() - (alignof(float) - 1U);
    auto* invalid_float_pointer = reinterpret_cast<float*>(last_aligned_start);
    const TensorView invalid_relu_view(invalid_float_pointer, 1, 1, 1);
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::relu(invalid_relu_view); },
        "ReLU must reject a byte range that wraps uintptr_t");

    const ConstTensorView invalid_logits(invalid_float_pointer, 1, 1, 1);
    std::int32_t label = -1;
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(invalid_logits, &label, 1); },
        "argmax must reject a logits range that wraps uintptr_t");

    const float one_logit_data[] = {1.0F};
    const ConstTensorView one_logit(one_logit_data, 1, 1, 1);
    const std::uintptr_t last_aligned_label_start =
        std::numeric_limits<std::uintptr_t>::max() - (alignof(std::int32_t) - 1U);
    auto* invalid_label_pointer = reinterpret_cast<std::int32_t*>(last_aligned_label_start);
    expect_throws<std::invalid_argument>(
        [&] { miniinfer::cpu_ops::argmax(one_logit, invalid_label_pointer, 1); },
        "argmax must reject a label range that wraps uintptr_t");
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    constexpr TestCase tests[] = {
        {"linear rectangular multi-batch and bias",
         test_linear_rectangular_multi_batch_and_bias},
        {"ReLU negative, zero and positive values",
         test_relu_negative_zero_and_positive_values},
        {"argmax ties choose lowest index", test_argmax_ties_choose_lowest_index},
        {"linear validation", test_linear_shape_alias_and_non_finite_validation},
        {"ReLU finite validation", test_relu_rejects_non_finite_before_writing},
        {"argmax validation", test_argmax_shape_alias_finite_and_capacity_validation},
        {"pointer range overflow", test_pointer_range_overflow_is_rejected_before_access},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "test_cpu_ops: " << test.name << ": " << error.what() << '\n';
            return 1;
        } catch (...) {
            std::cerr << "test_cpu_ops: " << test.name << ": unknown failure\n";
            return 1;
        }
    }

    std::cout << "test_cpu_ops: " << passed << " test groups passed\n";
    return 0;
}
