#include "miniinfer/checked_size.hpp"

#include <limits>
#include <stdexcept>

namespace miniinfer {

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error("size multiplication overflow");
    }
    return lhs * rhs;
}

}  // namespace miniinfer
