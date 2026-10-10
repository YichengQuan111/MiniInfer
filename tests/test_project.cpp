#include "miniinfer/version.hpp"

#include <cstddef>
#include <iostream>
#include <string_view>

namespace {

int fail(std::string_view message) {
    std::cerr << "test_project: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    const std::string_view value = miniinfer::version();
    if (value.empty()) {
        return fail("version must not be empty");
    }

    std::size_t field_start = 0;
    std::size_t dot_count = 0;
    for (std::size_t index = 0; index <= value.size(); ++index) {
        if (index != value.size() && value[index] != '.') {
            continue;
        }

        if (index == field_start) {
            return fail("version fields must not be empty");
        }

        for (std::size_t digit = field_start; digit < index; ++digit) {
            if (value[digit] < '0' || value[digit] > '9') {
                return fail("version fields must contain only decimal digits");
            }
        }

        if (index < value.size()) {
            ++dot_count;
            if (dot_count > 2) {
                return fail("version must contain exactly three numeric fields");
            }
        }
        field_start = index + 1;
    }

    if (dot_count != 2) {
        return fail("version must contain exactly three numeric fields");
    }

    return 0;
}
