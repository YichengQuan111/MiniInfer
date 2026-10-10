#include "miniinfer/mlp_model.hpp"
#include "miniinfer/checked_size.hpp"
#include "miniinfer/data_io.hpp"
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace miniinfer {

MlpModel::MlpModel(Buffer w1, Buffer b1, Buffer w2, Buffer b2) noexcept
    : w1_(std::move(w1)), b1_(std::move(b1)), w2_(std::move(w2)), b2_(std::move(b2)) {}

MlpModel MlpModel::load(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
        throw std::runtime_error("model directory '" + directory.string() +
                                 "': expected a directory" +
                                 (error ? ": " + error.message() : std::string{}));
    }

    auto w1 = load_f32_file(directory / "w1.f32", checked_multiply(hidden_features, input_features));
    auto b1 = load_f32_file(directory / "b1.f32", hidden_features);
    auto w2 = load_f32_file(directory / "w2.f32", checked_multiply(output_features, hidden_features));
    auto b2 = load_f32_file(directory / "b2.f32", output_features);
    // No model is published until all four loads have succeeded.
    return MlpModel(std::move(w1), std::move(b1), std::move(w2), std::move(b2));
}

ConstTensorView MlpModel::w1() const & {
    return {w1_.data(), w1_.size(), hidden_features, input_features};
}

ConstTensorView MlpModel::b1() const & {
    return {b1_.data(), b1_.size(), 1, hidden_features};
}

ConstTensorView MlpModel::w2() const & {
    return {w2_.data(), w2_.size(), output_features, hidden_features};
}

ConstTensorView MlpModel::b2() const & {
    return {b2_.data(), b2_.size(), 1, output_features};
}

}  // namespace miniinfer
