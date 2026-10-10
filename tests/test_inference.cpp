#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/workspace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using miniinfer::ConstTensorView;
using miniinfer::CpuExecutor;
using miniinfer::MlpInput;
using miniinfer::MlpModel;
using miniinfer::MlpResult;
using miniinfer::Workspace;

constexpr std::size_t kReferenceSamples = 64;
constexpr std::size_t kInputFeatures = 64;
constexpr std::size_t kOutputFeatures = 10;
constexpr float kAtol = 1.0e-4F;
constexpr float kRtol = 1.0e-4F;

static_assert(sizeof(float) == sizeof(std::uint32_t),
              "inference fixtures require 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24 &&
                  std::numeric_limits<float>::max_exponent == 128,
              "inference fixtures require IEEE-754 binary32 float");
static_assert(!std::is_copy_constructible_v<MlpResult>,
              "MlpResult must not be copy constructible");
static_assert(!std::is_copy_assignable_v<MlpResult>, "MlpResult must not be copy assignable");
static_assert(std::is_same_v<decltype(std::declval<const MlpResult&>().logits()),
                             ConstTensorView>,
              "MlpResult logits must be exposed as a read-only view");
static_assert(std::is_same_v<decltype(std::declval<const MlpResult&>().labels()),
                             const std::vector<std::int32_t>&>,
              "MlpResult labels must be exposed as a read-only reference");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().data()),
                             const float*>,
              "result logits must not expose mutable storage");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().at(0, 0)),
                             const float&>,
              "result logits must not expose mutable elements");

template <typename T, typename = void>
struct HasConstRvalueLogits : std::false_type {};
template <typename T>
struct HasConstRvalueLogits<T, std::void_t<decltype(std::declval<const T&&>().logits())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasConstRvalueLabels : std::false_type {};
template <typename T>
struct HasConstRvalueLabels<T, std::void_t<decltype(std::declval<const T&&>().labels())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasRvalueInputView : std::false_type {};
template <typename T>
struct HasRvalueInputView<T, std::void_t<decltype(std::declval<T&&>().view())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasRvalueWorkspaceHidden : std::false_type {};
template <typename T>
struct HasRvalueWorkspaceHidden<T, std::void_t<decltype(std::declval<T&&>().hidden(1))>>
    : std::true_type {};

static_assert(!HasConstRvalueLogits<MlpResult>::value,
              "MlpResult::logits must be unavailable on a temporary result");
static_assert(!HasConstRvalueLabels<MlpResult>::value,
              "MlpResult::labels must be unavailable on a temporary result");
static_assert(!HasRvalueInputView<MlpInput>::value,
              "MlpInput::view must be unavailable on a temporary input");
static_assert(!HasRvalueWorkspaceHidden<Workspace>::value,
              "Workspace::hidden must be unavailable on a temporary workspace");
static_assert(std::is_nothrow_move_constructible_v<Workspace>,
              "Workspace move construction must be noexcept");

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
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(message) + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(std::string(message) + ": expected exception was not thrown");
}

std::vector<std::uint8_t> read_exact_file(const fs::path& path, std::size_t expected_bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("could not open reference file: " + path.string());
    }
    file.seekg(0, std::ios::end);
    const std::streamoff actual_size = file.tellg();
    if (actual_size < 0 || static_cast<std::uintmax_t>(actual_size) != expected_bytes) {
        std::ostringstream message;
        message << "reference file has " << actual_size << " bytes, expected "
                << expected_bytes << ": " << path.string();
        throw std::runtime_error(message.str());
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(expected_bytes);
    if (!bytes.empty()) {
        file.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    if (!file || file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("could not read complete reference file: " + path.string());
    }
    return bytes;
}

std::uint32_t read_u32_le(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::vector<float> read_f32_file(const fs::path& path, std::size_t count) {
    const auto bytes = read_exact_file(path, count * sizeof(float));
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t bits = read_u32_le(bytes.data() + index * sizeof(float));
        std::memcpy(&values[index], &bits, sizeof(bits));
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("reference file contains a non-finite FP32 value: " +
                                     path.string());
        }
    }
    return values;
}

std::vector<std::int32_t> read_i32_file(const fs::path& path, std::size_t count) {
    const auto bytes = read_exact_file(path, count * sizeof(std::int32_t));
    std::vector<std::int32_t> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t bits = read_u32_le(bytes.data() + index * sizeof(std::int32_t));
        std::memcpy(&values[index], &bits, sizeof(bits));
    }
    return values;
}

struct ReferenceData {
    std::vector<float> inputs;
    std::vector<float> logits;
    std::vector<std::int32_t> labels;
};

ReferenceData load_reference_data(const fs::path& directory) {
    ReferenceData reference{
        read_f32_file(directory / "inputs.f32", kReferenceSamples * kInputFeatures),
        read_f32_file(directory / "logits.f32", kReferenceSamples * kOutputFeatures),
        read_i32_file(directory / "labels.i32", kReferenceSamples),
    };
    for (std::size_t sample = 0; sample < kReferenceSamples; ++sample) {
        if (reference.labels[sample] < 0 || reference.labels[sample] >=
                                                  static_cast<std::int32_t>(kOutputFeatures)) {
            throw std::runtime_error("reference label is outside [0,10) at sample " +
                                     std::to_string(sample));
        }
    }
    return reference;
}

std::vector<float> make_batch_inputs(const ReferenceData& reference, std::size_t first_sample,
                                    std::size_t batch) {
    std::vector<float> values(batch * kInputFeatures);
    for (std::size_t row = 0; row < batch; ++row) {
        const std::size_t sample = (first_sample + row) % kReferenceSamples;
        std::copy_n(reference.inputs.data() + sample * kInputFeatures, kInputFeatures,
                    values.data() + row * kInputFeatures);
    }
    return values;
}

std::size_t argmax_row(ConstTensorView logits, std::size_t row) {
    std::size_t best = 0;
    for (std::size_t column = 1; column < logits.cols(); ++column) {
        if (logits.at(row, column) > logits.at(row, best)) {
            best = column;
        }
    }
    return best;
}

struct ErrorStats {
    std::size_t compared_samples = 0;
    std::size_t compared_logits = 0;
    float max_absolute_error = 0.0F;
    double max_relative_error_nonzero_reference = 0.0;
};

void compare_result_to_reference(const MlpResult& result, const ReferenceData& reference,
                                 std::size_t first_sample, ErrorStats& stats,
                                 std::string_view batch_description) {
    const ConstTensorView logits = result.logits();
    expect(logits.rows() == result.batch(), "result logits row count must match result batch");
    expect(logits.cols() == kOutputFeatures, "result logits must have ten columns");
    expect(result.labels().size() == result.batch(), "result label count must match result batch");

    for (std::size_t row = 0; row < result.batch(); ++row) {
        const std::size_t sample = (first_sample + row) % kReferenceSamples;
        for (std::size_t column = 0; column < kOutputFeatures; ++column) {
            const float actual = logits.at(row, column);
            const float expected = reference.logits[sample * kOutputFeatures + column];
            const float absolute_error = std::abs(actual - expected);
            const float tolerance = kAtol + kRtol * std::abs(expected);
            stats.max_absolute_error = std::max(stats.max_absolute_error, absolute_error);
            if (expected != 0.0F) {
                const double relative_error = static_cast<double>(absolute_error) /
                                              std::abs(static_cast<double>(expected));
                stats.max_relative_error_nonzero_reference =
                    std::max(stats.max_relative_error_nonzero_reference, relative_error);
            }
            ++stats.compared_logits;
            if (!std::isfinite(actual) || absolute_error > tolerance) {
                std::ostringstream message;
                message << std::setprecision(9) << "batch=" << batch_description
                        << ", sample=" << sample << ", row=" << row << ", column=" << column
                        << ", actual=" << actual << ", expected=" << expected
                        << ", abs_error=" << absolute_error << ", tolerance=" << tolerance;
                throw std::runtime_error(message.str());
            }
        }

        const std::int32_t expected_label = reference.labels[sample];
        const std::int32_t actual_label = result.labels()[row];
        const std::size_t logits_label = argmax_row(logits, row);
        if (actual_label != expected_label || actual_label < 0 ||
            actual_label >= static_cast<std::int32_t>(kOutputFeatures) ||
            static_cast<std::size_t>(actual_label) != logits_label) {
            std::ostringstream message;
            message << "batch=" << batch_description << ", sample=" << sample << ", row=" << row
                    << ", expected_label=" << expected_label << ", actual_label=" << actual_label
                    << ", logits_argmax=" << logits_label;
            throw std::runtime_error(message.str());
        }
        ++stats.compared_samples;
    }
}

void test_reference_batches(const MlpModel& model, const ReferenceData& reference,
                            ErrorStats& stats) {
    constexpr std::array<std::size_t, 5> batch_sizes{1, 8, 32, 64, 256};
    Workspace workspace(256);
    float* const workspace_storage = workspace.hidden(1).data();
    CpuExecutor executor;

    for (const std::size_t requested_batch : batch_sizes) {
        const std::size_t samples_to_run = requested_batch == 256 ? 256 : kReferenceSamples;
        for (std::size_t first_sample = 0; first_sample < samples_to_run;
             first_sample += requested_batch) {
            const std::size_t current_batch =
                std::min(requested_batch, samples_to_run - first_sample);
            MlpInput input(current_batch, make_batch_inputs(reference, first_sample, current_batch));
            input.validate(256);
            MlpResult result = executor.run(model, input, workspace);
            if (workspace.hidden(current_batch).data() != workspace_storage) {
                throw std::runtime_error("workspace storage address changed while varying batch size");
            }
            std::ostringstream description;
            description << requested_batch << " (call batch " << current_batch << ')';
            compare_result_to_reference(result, reference, first_sample, stats, description.str());
        }
    }
}

void test_input_and_result_ownership(const MlpModel& model, const ReferenceData& reference) {
    std::vector<float> caller_values = make_batch_inputs(reference, 3, 1);
    const std::vector<float> original_values = caller_values;
    MlpInput owned_input(1, caller_values);  // Passing an lvalue must copy into the input.
    std::fill(caller_values.begin(), caller_values.end(), -1000.0F);
    const ConstTensorView owned_view = owned_input.view();
    for (std::size_t column = 0; column < kInputFeatures; ++column) {
        if (owned_view.at(0, column) != original_values[column]) {
            std::ostringstream message;
            message << "MlpInput did not retain copied value at sample=3, column=" << column;
            throw std::runtime_error(message.str());
        }
    }

    Workspace workspace(256);
    CpuExecutor executor;
    MlpResult old_result = executor.run(model, owned_input, workspace);
    const ConstTensorView old_logits = old_result.logits();
    const std::vector<float> old_logits_snapshot(old_logits.data(),
                                                  old_logits.data() + old_logits.size());
    const std::vector<std::int32_t> old_labels_snapshot = old_result.labels();
    const float* const old_result_address = old_logits.data();
    float* const workspace_address = workspace.hidden(1).data();

    MlpInput next_input(8, make_batch_inputs(reference, 40, 8));
    MlpResult next_result = executor.run(model, next_input, workspace);
    ErrorStats ownership_stats;
    compare_result_to_reference(old_result, reference, 3, ownership_stats, "owned input");
    // The earlier result is independently owned and remains unchanged after another run.
    expect(old_result.logits().data() == old_result_address,
           "a subsequent run must not replace prior result storage");
    expect(old_result.logits().data() != next_result.logits().data(),
           "distinct results must own distinct logits storage");
    expect(workspace.hidden(8).data() == workspace_address,
           "workspace allocation must be reused when the batch changes");
    for (std::size_t index = 0; index < old_logits_snapshot.size(); ++index) {
        if (old_result.logits().data()[index] != old_logits_snapshot[index]) {
            throw std::runtime_error("a subsequent run changed an earlier result at sample=3, column=" +
                                     std::to_string(index % kOutputFeatures));
        }
    }
    expect(old_result.labels() == old_labels_snapshot,
           "a subsequent run must not change earlier result labels");
    expect(old_result.logits().data() != workspace_address,
           "result logits must not alias workspace hidden storage");
    compare_result_to_reference(next_result, reference, 40, ownership_stats, "next result");

    Workspace independent_workspace(256);
    float* const independent_address = independent_workspace.hidden(1).data();
    const std::uintptr_t workspace_begin = reinterpret_cast<std::uintptr_t>(workspace_address);
    const std::uintptr_t independent_begin = reinterpret_cast<std::uintptr_t>(independent_address);
    const std::uintptr_t workspace_end = workspace_begin + 256U * 128U * sizeof(float);
    const std::uintptr_t independent_end = independent_begin + 256U * 128U * sizeof(float);
    expect(workspace_end <= independent_begin || independent_end <= workspace_begin,
           "independent workspaces must not overlap");
}

void test_invalid_input_and_workspace_contracts(const MlpModel& model) {
    expect_throws<std::invalid_argument>([] { MlpInput input(0, {}); },
                                         "zero-sized input must be rejected");
    expect_throws<std::invalid_argument>([] { MlpInput input(1, {}); },
                                         "empty input data must be rejected");
    expect_throws<std::invalid_argument>([] { MlpInput input(2, std::vector<float>(127)); },
                                         "incomplete input data must be rejected");
    expect_throws<std::overflow_error>(
        [] { MlpInput input(std::numeric_limits<std::size_t>::max(), {}); },
        "input batch multiplication overflow must be rejected");

    std::vector<float> nan_values(kInputFeatures, 0.0F);
    nan_values[17] = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>([&] { MlpInput input(1, nan_values); },
                                         "NaN input must be rejected");
    std::vector<float> inf_values(kInputFeatures, 0.0F);
    inf_values[21] = std::numeric_limits<float>::infinity();
    expect_throws<std::invalid_argument>([&] { MlpInput input(1, inf_values); },
                                         "infinite input must be rejected");

    MlpInput valid_input(1, std::vector<float>(kInputFeatures, 0.0F));
    expect_throws<std::invalid_argument>([&] { valid_input.validate(0); },
                                         "input must reject a zero max_batch limit");
    MlpInput over_limit_input(3, std::vector<float>(3 * kInputFeatures, 0.0F));
    expect_throws<std::invalid_argument>([&] { over_limit_input.validate(2); },
                                         "input above max_batch must be rejected");
    expect_throws<std::invalid_argument>(
        [&] { Workspace small(2); (void)CpuExecutor{}.run(model, over_limit_input, small); },
        "executor must reject input above Workspace capacity before writing");

    Workspace workspace(2);
    expect(workspace.max_batch() == 2, "workspace must preserve its finite capacity");
    expect(workspace.hidden(2).rows() == 2, "workspace must allow its capacity boundary");
    expect_throws<std::invalid_argument>([&] { (void)workspace.hidden(0); },
                                         "zero-sized workspace view must be rejected");
    expect_throws<std::invalid_argument>([&] { (void)workspace.hidden(3); },
                                         "workspace view above capacity must be rejected");
    expect_throws<std::overflow_error>(
        [] { Workspace enormous(std::numeric_limits<std::size_t>::max()); },
        "workspace size multiplication overflow must be rejected before allocation");
    expect_throws<std::invalid_argument>([] { Workspace empty(0); },
                                         "zero-capacity workspace must be rejected");

    Workspace source(4);
    Workspace destination(std::move(source));
    expect(destination.max_batch() == 4, "moved workspace must retain capacity");
    expect(source.max_batch() == 0, "moved-from workspace must report zero capacity");
    expect_throws<std::invalid_argument>([&] { (void)source.hidden(1); },
                                         "moved-from workspace must reject views");

    MlpInput input_source(1, std::vector<float>(kInputFeatures, 0.0F));
    MlpInput input_destination(std::move(input_source));
    expect(input_destination.batch() == 1, "moved input must retain its batch");
    expect_throws<std::invalid_argument>([&] { input_source.validate(256); },
                                         "moved-from input must reject validation");
    expect_throws<std::invalid_argument>(
        [&] {
            CpuExecutor executor;
            (void)executor.run(model, input_source, destination);
        },
        "executor must reject a moved-from input");
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "test_inference: expected exactly one reference data directory argument\n";
        return 2;
    }

    try {
        const fs::path reference_directory(argv[1]);
        const ReferenceData reference = load_reference_data(reference_directory);
        const MlpModel model = MlpModel::load(reference_directory);
        ErrorStats stats;
        test_reference_batches(model, reference, stats);
        test_input_and_result_ownership(model, reference);
        test_invalid_input_and_workspace_contracts(model);
        std::cout << "test_inference: 3 groups passed; reference_batches=5; compared_samples="
                  << stats.compared_samples << "; compared_logits=" << stats.compared_logits
                  << "; max_abs_error=" << std::setprecision(9) << stats.max_absolute_error
                  << "; max_relative_error_nonzero_reference="
                  << std::setprecision(9) << stats.max_relative_error_nonzero_reference << '\n';
    } catch (const std::exception& error) {
        std::cerr << "test_inference: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "test_inference: unknown failure\n";
        return 1;
    }
    return 0;
}
