#include "miniinfer/buffer.hpp"
#include "miniinfer/checked_size.hpp"

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

using miniinfer::Buffer;

static_assert(!std::is_copy_constructible_v<Buffer>, "Buffer must not be copy constructible");
static_assert(!std::is_copy_assignable_v<Buffer>, "Buffer must not be copy assignable");
static_assert(std::is_nothrow_move_constructible_v<Buffer>, "Buffer move construction must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<Buffer>, "Buffer move assignment must be noexcept");
static_assert(std::is_same_v<decltype(std::declval<Buffer&>().data()), float*>,
              "mutable data() must return float*");
static_assert(std::is_same_v<decltype(std::declval<const Buffer&>().data()), const float*>,
              "const data() must return const float*");

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

void test_default_and_zero_buffers() {
    Buffer default_buffer;
    expect(default_buffer.empty(), "default buffer must be empty");
    expect(default_buffer.size() == 0, "default buffer size must be zero");
    expect(default_buffer.size_bytes() == 0, "default buffer byte size must be zero");
    expect(default_buffer.data() == nullptr, "default buffer data must be null");

    Buffer zero_buffer(0);
    expect(zero_buffer.empty(), "zero-sized buffer must be empty");
    expect(zero_buffer.size() == 0, "zero-sized buffer size must be zero");
    expect(zero_buffer.size_bytes() == 0, "zero-sized buffer byte size must be zero");
    expect(zero_buffer.data() == nullptr, "zero-sized buffer data must be null");
}

void test_storage_initialization_and_access() {
    Buffer buffer(4);
    expect(!buffer.empty(), "nonzero buffer must not be empty");
    expect(buffer.size() == 4, "size must count float elements");
    expect(buffer.size_bytes() == 4 * sizeof(float), "byte size must match float storage");
    expect(buffer.data() != nullptr, "nonzero buffer data must not be null");
    expect(reinterpret_cast<std::uintptr_t>(buffer.data()) % alignof(float) == 0,
           "data must be aligned for float");

    for (std::size_t index = 0; index < buffer.size(); ++index) {
        expect(buffer.data()[index] == 0.0F, "new storage must be zero initialized");
    }

    buffer.data()[0] = 1.25F;
    buffer.data()[1] = -2.5F;
    buffer.data()[2] = 3.75F;
    buffer.data()[3] = 8.0F;
    expect(buffer.data()[0] == 1.25F && buffer.data()[1] == -2.5F &&
               buffer.data()[2] == 3.75F && buffer.data()[3] == 8.0F,
           "data must support indexed reads and writes");

    const Buffer& const_buffer = buffer;
    const float* const_data = const_buffer.data();
    expect(const_data == buffer.data(), "const data() must expose the same storage");
    expect(const_data[1] == -2.5F, "const data() must be readable");
}

void test_move_construction_and_source_reuse() {
    Buffer source(3);
    source.data()[0] = 11.0F;
    source.data()[1] = 12.0F;
    source.data()[2] = 13.0F;
    float* const original_data = source.data();

    Buffer destination(std::move(source));
    expect(destination.data() == original_data, "move construction must transfer storage");
    expect(destination.size() == 3, "move destination must retain element count");
    expect(destination.size_bytes() == 3 * sizeof(float), "move destination must retain byte count");
    expect(destination.data()[0] == 11.0F && destination.data()[2] == 13.0F,
           "move destination must retain values");
    expect(source.empty() && source.size() == 0 && source.size_bytes() == 0,
           "moved-from source must be empty");
    expect(source.data() == nullptr, "moved-from source data must be null");

    source = Buffer(2);
    expect(source.size() == 2 && source.data() != nullptr,
           "moved-from source must be reusable");
    expect(source.data()[0] == 0.0F && source.data()[1] == 0.0F,
           "reused source storage must be initialized");
}

void test_move_assignment_replaces_destination() {
    Buffer source(2);
    source.data()[0] = 21.0F;
    source.data()[1] = 22.0F;
    float* const transferred_data = source.data();

    Buffer destination(5);
    destination.data()[0] = -99.0F;
    destination = std::move(source);

    expect(destination.data() == transferred_data, "move assignment must transfer source storage");
    expect(destination.size() == 2 && destination.size_bytes() == 2 * sizeof(float),
           "move assignment must replace destination size");
    expect(destination.data()[0] == 21.0F && destination.data()[1] == 22.0F,
           "move assignment must preserve source values");
    expect(source.empty() && source.data() == nullptr,
           "move-assigned source must become empty");

    source = Buffer(1);
    source.data()[0] = 31.0F;
    expect(source.size() == 1 && source.data()[0] == 31.0F,
           "move-assigned source must support subsequent use");
}

void test_moves_from_empty_and_self_move() {
    Buffer empty;
    Buffer moved_empty(std::move(empty));
    expect(empty.empty() && empty.data() == nullptr, "moving from empty must preserve an empty source");
    expect(moved_empty.empty() && moved_empty.data() == nullptr,
           "move construction from empty must produce an empty destination");

    Buffer destination(2);
    destination.data()[0] = 41.0F;
    destination = std::move(moved_empty);
    expect(destination.empty() && destination.data() == nullptr,
           "move assignment from empty must release destination storage");
    expect(moved_empty.empty() && moved_empty.data() == nullptr,
           "move assignment from empty must leave source empty");

    Buffer self(2);
    self.data()[0] = 51.0F;
    self.data()[1] = 52.0F;
    float* const self_data = self.data();
    Buffer& self_alias = self;
    self = std::move(self_alias);
    expect(self.data() == self_data && self.size() == 2 &&
               self.data()[0] == 51.0F && self.data()[1] == 52.0F,
           "self move assignment must preserve contents and ownership");
}

void test_checked_multiply() {
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    expect(miniinfer::checked_multiply(7, 6) == 42, "ordinary multiplication must be exact");
    expect(miniinfer::checked_multiply(0, 0) == 0, "two zero operands must produce zero");
    expect(miniinfer::checked_multiply(0, max) == 0, "zero left operand must produce zero");
    expect(miniinfer::checked_multiply(max, 0) == 0, "zero right operand must produce zero");
    expect(miniinfer::checked_multiply(1, max) == max, "maximum times one must be representable");
    expect(miniinfer::checked_multiply(max, 1) == max, "one times maximum must be representable");

    const std::size_t largest_even_factor = max / 2;
    expect(miniinfer::checked_multiply(largest_even_factor, 2) == largest_even_factor * 2,
           "largest even product within size_t must be representable");
    expect_throws<std::overflow_error>(
        [] { (void)miniinfer::checked_multiply(std::numeric_limits<std::size_t>::max() / 2 + 1, 2); },
        "overflow with large left operand must throw");
    expect_throws<std::overflow_error>(
        [] { (void)miniinfer::checked_multiply(2, std::numeric_limits<std::size_t>::max() / 2 + 1); },
        "overflow with large right operand must throw");
}

void test_buffer_size_limits() {
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    const std::size_t count_overflow = max / sizeof(float) + 1;
    expect_throws<std::overflow_error>([count_overflow] { Buffer too_large(count_overflow); },
                                       "float byte-size overflow must throw overflow_error");

    const std::size_t ptrdiff_limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const std::size_t first_count_over_ptrdiff = ptrdiff_limit / sizeof(float) + 1;
    const std::size_t first_bytes_over_ptrdiff =
        miniinfer::checked_multiply(first_count_over_ptrdiff, sizeof(float));
    expect(first_bytes_over_ptrdiff > ptrdiff_limit,
           "boundary test size must exceed PTRDIFF_MAX without size_t overflow");
    expect_throws<std::length_error>([first_count_over_ptrdiff] { Buffer too_large(first_count_over_ptrdiff); },
                                     "object larger than PTRDIFF_MAX must throw length_error");
}

void test_scope_destruction() {
    for (int iteration = 0; iteration < 8; ++iteration) {
        Buffer first(16);
        Buffer second(8);
        first.data()[0] = static_cast<float>(iteration);
        second = std::move(first);
        expect(second.size() == 16 && second.data()[0] == static_cast<float>(iteration),
               "scope-owned moved buffer must remain usable until destruction");
    }
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    constexpr TestCase tests[] = {
        {"default and zero buffers", test_default_and_zero_buffers},
        {"storage initialization and access", test_storage_initialization_and_access},
        {"move construction and source reuse", test_move_construction_and_source_reuse},
        {"move assignment replaces destination", test_move_assignment_replaces_destination},
        {"empty moves and self move", test_moves_from_empty_and_self_move},
        {"checked multiply", test_checked_multiply},
        {"buffer size limits", test_buffer_size_limits},
        {"scope destruction", test_scope_destruction},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "test_buffer: " << test.name << ": " << error.what() << '\n';
            return 1;
        } catch (...) {
            std::cerr << "test_buffer: " << test.name << ": unknown failure\n";
            return 1;
        }
    }

    std::cout << "test_buffer: " << passed << " test groups passed\n";
    return 0;
}
