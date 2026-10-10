#include "miniinfer/version.hpp"

namespace miniinfer {

std::string_view version() noexcept {
    return MINIINFER_VERSION_STRING;
}

}  // namespace miniinfer
