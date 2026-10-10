#pragma once

#include "miniinfer/buffer.hpp"
#include "miniinfer/tensor_view.hpp"

#include <cstddef>
#include <filesystem>

namespace miniinfer {

// Owns the fixed digits MLP weights. Published views are read-only and borrowed.
class MlpModel final {
public:
    static constexpr std::size_t input_features = 64;
    static constexpr std::size_t hidden_features = 128;
    static constexpr std::size_t output_features = 10;

    static MlpModel load(const std::filesystem::path& directory);

    MlpModel(const MlpModel&) = delete;
    MlpModel& operator=(const MlpModel&) = delete;
    MlpModel(MlpModel&&) noexcept = default;
    MlpModel& operator=(MlpModel&&) = delete;
    ~MlpModel() = default;

    ConstTensorView w1() const &;  // [128,64], weights[out,in]
    ConstTensorView b1() const &;  // [1,128], row-vector representation
    ConstTensorView w2() const &;  // [10,128], weights[out,in]
    ConstTensorView b2() const &;  // [1,10], row-vector representation
    // A temporary model cannot leave behind a dangling weight view.
    ConstTensorView w1() const && = delete;
    ConstTensorView b1() const && = delete;
    ConstTensorView w2() const && = delete;
    ConstTensorView b2() const && = delete;

private:
    MlpModel(Buffer w1, Buffer b1, Buffer w2, Buffer b2) noexcept;

    Buffer w1_;
    Buffer b1_;
    Buffer w2_;
    Buffer b2_;
};

}  // namespace miniinfer
