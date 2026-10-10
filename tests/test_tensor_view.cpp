#include "miniinfer/tensor_view.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using miniinfer::ConstTensorView;
using miniinfer::TensorView;

static_assert(!std::is_default_constructible_v<TensorView>,
              "TensorView must not have a default constructor");
static_assert(std::is_same_v<decltype(std::declval<const TensorView&>().data()), float*>,
              "mutable view data() must return float*");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().data()),
                             const float*>,
              "const view data() must return const float*");
static_assert(std::is_same_v<decltype(std::declval<const TensorView&>().row(0)), float*>,
              "mutable view row() must return float*");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().row(0)),
                             const float*>,
              "const view row() must return const float*");
static_assert(std::is_same_v<decltype(std::declval<const TensorView&>().at(0, 0)), float&>,
              "mutable view at() must return float&");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().at(0, 0)),
                             const float&>,
              "const view at() must return const float&");
static_assert(std::is_same_v<decltype(std::declval<const TensorView&>().as_const()),
                             ConstTensorView>,
              "as_const() must return ConstTensorView");
static_assert(!std::is_constructible_v<TensorView, ConstTensorView>,
              "a const view must not convert to a mutable view");
static_assert(!std::is_constructible_v<TensorView, const float*, std::size_t,
                                       std::size_t, std::size_t>,
              "a const pointer must not construct a mutable view");
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

void test_row_major_shape_and_mutable_aliases() {
    float values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    TensorView view(values, 6, 2, 3);

    expect(view.data() == values, "view must borrow the supplied storage");
    expect(view.rows() == 2 && view.cols() == 3, "view must retain its rectangular shape");
    expect(view.size() == 6 && view.size_bytes() == 6 * sizeof(float),
           "view size queries must report the logical element and byte counts");
    expect(view.row(0) == values && view.row(1) == values + 3,
           "row pointers must follow contiguous row-major layout");
    expect(view.at(0, 0) == 1.0F && view.at(0, 2) == 3.0F &&
               view.at(1, 0) == 4.0F && view.at(1, 2) == 6.0F,
           "at() must map row and column indices in row-major order");

    view.at(1, 1) = 50.0F;
    view.row(0)[1] = 20.0F;
    expect(values[4] == 50.0F && values[1] == 20.0F,
           "view writes must update the borrowed array");
}

void test_const_views_and_copy_borrowing() {
    float values[] = {7.0F, 8.0F, 9.0F, 10.0F};
    TensorView mutable_view(values, 4, 2, 2);
    TensorView copied_view = mutable_view;
    copied_view.at(0, 1) = 80.0F;
    expect(values[1] == 80.0F && mutable_view.at(0, 1) == 80.0F,
           "copied views must borrow the same storage");

    ConstTensorView readonly_view = mutable_view.as_const();
    expect(readonly_view.data() == values && readonly_view.at(1, 0) == 9.0F,
           "as_const() must retain the same storage and permit reads");

    const float fixed_values[] = {11.0F, 12.0F, 13.0F};
    ConstTensorView direct_const_view(fixed_values, 3, 1, 3);
    expect(direct_const_view.row(0)[2] == 13.0F,
           "const pointer views must support read access");
}

void test_extra_capacity_and_insufficient_capacity() {
    float values[] = {1.0F, 2.0F, 3.0F, 4.0F, 99.0F};
    TensorView prefix(values, 5, 2, 2);
    expect(prefix.size() == 4 && prefix.size_bytes() == 4 * sizeof(float),
           "logical size must exclude extra available capacity");
    expect(prefix.at(1, 1) == 4.0F, "view must address its complete logical prefix");
    expect_throws<std::invalid_argument>(
        [&values] { TensorView too_small(values, 3, 2, 2); },
        "capacity shorter than the logical shape must throw invalid_argument");
}

void test_invalid_construction_arguments() {
    float value = 1.0F;
    expect_throws<std::invalid_argument>(
        [&value] { TensorView zero_rows(&value, 1, 0, 1); },
        "zero rows must throw invalid_argument");
    expect_throws<std::invalid_argument>(
        [&value] { TensorView zero_cols(&value, 1, 1, 0); },
        "zero columns must throw invalid_argument");
    expect_throws<std::invalid_argument>(
        [] { TensorView null_view(nullptr, 1, 1, 1); },
        "null data must throw invalid_argument");

    if (alignof(float) > 1) {
        alignas(float) unsigned char storage[sizeof(float) + alignof(float)]{};
        float* const misaligned = reinterpret_cast<float*>(storage + 1);
        expect(reinterpret_cast<std::uintptr_t>(misaligned) % alignof(float) != 0,
               "test pointer must actually be misaligned");
        expect_throws<std::invalid_argument>(
            [misaligned] { TensorView bad_alignment(misaligned, 1, 1, 1); },
            "misaligned data must throw invalid_argument");
    }
}

void test_index_bounds() {
    float values[] = {1.0F, 2.0F, 3.0F, 4.0F};
    TensorView view(values, 4, 2, 2);
    expect_throws<std::out_of_range>([&view] { (void)view.row(2); },
                                     "row index equal to rows must throw out_of_range");
    expect_throws<std::out_of_range>([&view] { (void)view.at(2, 0); },
                                     "at() row index equal to rows must throw out_of_range");
    expect_throws<std::out_of_range>([&view] { (void)view.at(0, 2); },
                                     "at() column index equal to columns must throw out_of_range");
}

void test_size_overflow_checks() {
    float value = 0.0F;
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    expect_throws<std::overflow_error>(
        [&value, max] { TensorView element_overflow(&value, max, max, 2); },
        "rows times columns overflow must throw overflow_error");

    const std::size_t byte_overflow_count = max / sizeof(float) + 1;
    expect_throws<std::overflow_error>(
        [&value, max, byte_overflow_count] {
            TensorView byte_overflow(&value, max, byte_overflow_count, 1);
        },
        "logical element count times sizeof(float) overflow must throw overflow_error");
}

void test_ptrdiff_byte_boundary() {
    float value = 0.0F;
    const std::size_t ptrdiff_max = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    const std::size_t largest_fitting_count = ptrdiff_max / sizeof(float);
    const std::size_t first_exceeding_count = largest_fitting_count + 1;

    expect_throws<std::invalid_argument>(
        [&value, largest_fitting_count] {
            TensorView within_limit(&value, 0, largest_fitting_count, 1);
        },
        "largest whole-float byte size within PTRDIFF_MAX must pass size validation");

    expect_throws<std::length_error>(
        [&value, first_exceeding_count] {
            TensorView beyond_limit(&value, 0, first_exceeding_count, 1);
        },
        "first whole-float byte size above PTRDIFF_MAX must throw length_error");
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    constexpr TestCase tests[] = {
        {"row-major shape and mutable aliases", test_row_major_shape_and_mutable_aliases},
        {"const views and copy borrowing", test_const_views_and_copy_borrowing},
        {"extra and insufficient capacity", test_extra_capacity_and_insufficient_capacity},
        {"invalid construction arguments", test_invalid_construction_arguments},
        {"index bounds", test_index_bounds},
        {"size overflow checks", test_size_overflow_checks},
        {"PTRDIFF byte boundary", test_ptrdiff_byte_boundary},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "test_tensor_view: " << test.name << ": " << error.what() << '\n';
            return 1;
        } catch (...) {
            std::cerr << "test_tensor_view: " << test.name << ": unknown failure\n";
            return 1;
        }
    }

    std::cout << "test_tensor_view: " << passed << " test groups passed\n";
    return 0;
}
