#pragma once

#include <cstddef>

namespace miniinfer {

// Zero factors are valid. Throws std::overflow_error instead of wrapping.
std::size_t checked_multiply(std::size_t lhs, std::size_t rhs);

}  // namespace miniinfer
