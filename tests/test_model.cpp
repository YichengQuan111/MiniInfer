#include "miniinfer/mlp_model.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using miniinfer::ConstTensorView;
using miniinfer::MlpModel;

static_assert(MlpModel::input_features == 64, "model input feature count must be 64");
static_assert(MlpModel::hidden_features == 128, "model hidden feature count must be 128");
static_assert(MlpModel::output_features == 10, "model output feature count must be 10");
static_assert(!std::is_copy_constructible_v<MlpModel>, "MlpModel must not be copy constructible");
static_assert(!std::is_copy_assignable_v<MlpModel>, "MlpModel must not be copy assignable");
static_assert(std::is_nothrow_move_constructible_v<MlpModel>,
              "MlpModel move construction must be noexcept");
static_assert(!std::is_move_assignable_v<MlpModel>, "MlpModel move assignment must be deleted");
static_assert(std::is_same_v<decltype(std::declval<const MlpModel&>().w1()), ConstTensorView>,
              "w1() must return a ConstTensorView by value");
static_assert(std::is_same_v<decltype(std::declval<const MlpModel&>().b1()), ConstTensorView>,
              "b1() must return a ConstTensorView by value");
static_assert(std::is_same_v<decltype(std::declval<const MlpModel&>().w2()), ConstTensorView>,
              "w2() must return a ConstTensorView by value");
static_assert(std::is_same_v<decltype(std::declval<const MlpModel&>().b2()), ConstTensorView>,
              "b2() must return a ConstTensorView by value");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().data()), const float*>,
              "ConstTensorView::data() must return const float*");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().at(0, 0)),
                             const float&>,
              "ConstTensorView::at() must return const float&");
static_assert(std::is_same_v<decltype(std::declval<const ConstTensorView&>().row(0)),
                             const float*>,
              "ConstTensorView::row() must return const float*");

template <typename T, typename = void>
struct HasConstRvalueW1 : std::false_type {};
template <typename T>
struct HasConstRvalueW1<T, std::void_t<decltype(std::declval<const T&&>().w1())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasConstRvalueB1 : std::false_type {};
template <typename T>
struct HasConstRvalueB1<T, std::void_t<decltype(std::declval<const T&&>().b1())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasConstRvalueW2 : std::false_type {};
template <typename T>
struct HasConstRvalueW2<T, std::void_t<decltype(std::declval<const T&&>().w2())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasConstRvalueB2 : std::false_type {};
template <typename T>
struct HasConstRvalueB2<T, std::void_t<decltype(std::declval<const T&&>().b2())>>
    : std::true_type {};

static_assert(!HasConstRvalueW1<MlpModel>::value, "w1() must be deleted on const rvalues");
static_assert(!HasConstRvalueB1<MlpModel>::value, "b1() must be deleted on const rvalues");
static_assert(!HasConstRvalueW2<MlpModel>::value, "w2() must be deleted on const rvalues");
static_assert(!HasConstRvalueB2<MlpModel>::value, "b2() must be deleted on const rvalues");
static_assert(sizeof(float) == sizeof(std::uint32_t), "test fixtures require 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24 &&
                  std::numeric_limits<float>::max_exponent == 128,
              "test fixtures require IEEE-754 binary32 float");

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

struct TensorSpec {
    const char* filename;
    std::size_t rows;
    std::size_t cols;
    std::vector<float> values;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const fs::path base = fs::temp_directory_path();
        static std::atomic<std::uint64_t> sequence{0};
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (std::uint64_t attempt = 0; attempt < 100; ++attempt) {
            const std::uint64_t suffix = sequence.fetch_add(1, std::memory_order_relaxed);
            path_ = base / ("miniinfer_model_test_" + std::to_string(tick) + "_" +
                            std::to_string(suffix) + "_" + std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path_, error)) {
                created_ = true;
                return;
            }
            if (error) {
                throw std::runtime_error("cannot create temporary test directory " +
                                         path_.string() + ": " + error.message());
            }
        }
        throw std::runtime_error("cannot create unique temporary test directory under " +
                                 base.string());
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory() {
        if (!created_) {
            return;
        }
        for (const char* filename : {"w1.f32", "b1.f32", "w2.f32", "b2.f32",
                                     "directory_path_is_a_file"}) {
            std::error_code ignored;
            fs::remove(path_ / filename, ignored);
        }
        std::error_code ignored;
        fs::remove(path_, ignored);
    }

    const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
    bool created_ = false;
};

std::vector<float> make_values(std::size_t count, int layer) {
    std::vector<float> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        int value = 0;
        switch (layer) {
            case 0:
                value = static_cast<int>((index * 3) % 101) - 50;
                break;
            case 1:
                value = 100 + static_cast<int>((index * 5) % 31);
                break;
            case 2:
                value = -200 + static_cast<int>((index * 7) % 41);
                break;
            default:
                value = 300 + static_cast<int>(index * 2);
                break;
        }
        values.push_back(static_cast<float>(value));
    }
    return values;
}

std::array<TensorSpec, 4> make_specs() {
    return {{{"w1.f32", 128, 64, make_values(128 * 64, 0)},
             {"b1.f32", 1, 128, make_values(128, 1)},
             {"w2.f32", 10, 128, make_values(10 * 128, 2)},
             {"b2.f32", 1, 10, make_values(10, 3)}}};
}

void encode_u32_le(std::uint32_t bits, std::uint8_t* destination) {
    destination[0] = static_cast<std::uint8_t>(bits & 0xffU);
    destination[1] = static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
    destination[2] = static_cast<std::uint8_t>((bits >> 16U) & 0xffU);
    destination[3] = static_cast<std::uint8_t>((bits >> 24U) & 0xffU);
}

std::vector<std::uint8_t> encode_floats_le(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        encode_u32_le(bits, bytes.data() + index * sizeof(float));
    }
    return bytes;
}

void write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create fixture file " + path.string());
    }
    for (const std::uint8_t byte : bytes) {
        stream.put(static_cast<char>(byte));
    }
    if (!stream) {
        throw std::runtime_error("cannot write fixture file " + path.string());
    }
}

void write_tensor(const fs::path& directory, const TensorSpec& spec) {
    write_bytes(directory / spec.filename, encode_floats_le(spec.values));
}

void write_complete_fixture(const fs::path& directory,
                            const std::array<TensorSpec, 4>& specs) {
    for (const TensorSpec& spec : specs) {
        write_tensor(directory, spec);
    }
}

void verify_view(ConstTensorView view, const TensorSpec& spec, std::string_view context) {
    const std::string label = std::string(context) + " (" + spec.filename + ")";
    expect(view.rows() == spec.rows, label + ": rows must match the expected shape");
    expect(view.cols() == spec.cols, label + ": columns must match the expected shape");
    expect(view.size() == spec.values.size(), label + ": element count must match the expected shape");
    expect(view.size_bytes() == spec.values.size() * sizeof(float),
           label + ": byte count must match the expected shape");
    expect(view.data() != nullptr, label + ": nonempty tensor data must not be null");

    for (std::size_t index = 0; index < spec.values.size(); ++index) {
        const std::size_t row = index / spec.cols;
        const std::size_t col = index % spec.cols;
        const float expected = spec.values[index];
        expect(std::isfinite(view.data()[index]), label + ": every value must be finite");
        expect(view.data()[index] == expected, label + ": flat element mismatch at index " +
                                                  std::to_string(index));
        expect(view.at(row, col) == expected, label + ": at(row, col) row-major mismatch at index " +
                                                   std::to_string(index));
        expect(view.row(row)[col] == expected, label + ": row(row)[col] mismatch at index " +
                                                    std::to_string(index));
    }
}

void verify_all_views(const MlpModel& model, const std::array<TensorSpec, 4>& specs,
                      std::string_view context) {
    verify_view(model.w1(), specs[0], context);
    verify_view(model.b1(), specs[1], context);
    verify_view(model.w2(), specs[2], context);
    verify_view(model.b2(), specs[3], context);
}

void verify_reference_view(ConstTensorView view, std::size_t rows, std::size_t cols,
                           std::string_view filename) {
    const std::string label = "reference model (" + std::string(filename) + ")";
    expect(view.rows() == rows, label + ": rows must match the model shape");
    expect(view.cols() == cols, label + ": columns must match the model shape");
    expect(view.size() == rows * cols, label + ": element count must match the model shape");
    expect(view.size_bytes() == rows * cols * sizeof(float),
           label + ": byte count must match the model shape");
    expect(view.data() != nullptr, label + ": nonempty tensor data must not be null");
    for (std::size_t index = 0; index < view.size(); ++index) {
        expect(std::isfinite(view.data()[index]),
               label + ": value must be finite at flat index " + std::to_string(index));
    }
}

void expect_load_error_with_context(const fs::path& directory, std::string_view context,
                                    std::string_view failure_case) {
    const std::string label = std::string(context) + ": " + std::string(failure_case);
    try {
        (void)MlpModel::load(directory);
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(context) == std::string::npos) {
            throw std::runtime_error(label + ": runtime_error omitted expected path/file context; got: " +
                                     error.what());
        }
        return;
    }
    throw std::runtime_error(label + ": expected std::runtime_error");
}

template <typename Mutation>
void run_file_failure_case(const fs::path& directory, const std::array<TensorSpec, 4>& specs,
                           const TensorSpec& target, std::string_view failure_case,
                           Mutation&& mutate) {
    write_complete_fixture(directory, specs);
    const fs::path target_path = directory / target.filename;
    mutate(target_path, target);
    expect_load_error_with_context(directory, target.filename, failure_case);

    std::error_code error;
    if (fs::is_directory(target_path, error)) {
        if (error || !fs::remove(target_path, error) || error) {
            throw std::runtime_error(std::string(target.filename) + ": " +
                                     std::string(failure_case) +
                                     ": cannot remove the test-created replacement directory");
        }
    }
    write_complete_fixture(directory, specs);
    const MlpModel recovered = MlpModel::load(directory);
    verify_all_views(recovered, specs, std::string("recovery after ") + std::string(failure_case));
}

void test_reference_model(const fs::path& reference_directory) {
    const MlpModel model = MlpModel::load(reference_directory);
    verify_reference_view(model.w1(), 128, 64, "w1.f32");
    verify_reference_view(model.b1(), 1, 128, "b1.f32");
    verify_reference_view(model.w2(), 10, 128, "w2.f32");
    verify_reference_view(model.b2(), 1, 10, "b2.f32");
}

void test_deterministic_fixture_and_row_major_mapping(const fs::path&) {
    TemporaryDirectory temporary;
    const auto specs = make_specs();
    write_complete_fixture(temporary.path(), specs);
    const MlpModel model = MlpModel::load(temporary.path());
    verify_all_views(model, specs, "deterministic fixture and row-major mapping");
}

void test_move_construction_getters_on_destination(const fs::path&) {
    TemporaryDirectory temporary;
    const auto specs = make_specs();
    write_complete_fixture(temporary.path(), specs);
    MlpModel source = MlpModel::load(temporary.path());
    MlpModel destination(std::move(source));
    verify_all_views(destination, specs, "move-constructed destination");
}

void test_file_failure_matrix(const fs::path&) {
    TemporaryDirectory temporary;
    const auto specs = make_specs();
    for (const TensorSpec& target : specs) {
        run_file_failure_case(temporary.path(), specs, target, "missing file",
                              [](const fs::path& path, const TensorSpec&) {
                                  std::error_code error;
                                  const bool removed = fs::remove(path, error);
                                  if (error || !removed) {
                                      throw std::runtime_error(path.string() +
                                                               ": could not prepare missing-file case");
                                  }
                              });
        run_file_failure_case(temporary.path(), specs, target, "truncated file (expected bytes minus one)",
                              [](const fs::path& path, const TensorSpec& spec) {
                                  auto bytes = encode_floats_le(spec.values);
                                  bytes.resize(bytes.size() - 1);
                                  write_bytes(path, bytes);
                              });
        run_file_failure_case(temporary.path(), specs, target, "empty file",
                              [](const fs::path& path, const TensorSpec&) {
                                  write_bytes(path, {});
                              });
        run_file_failure_case(temporary.path(), specs, target, "extra trailing byte",
                              [](const fs::path& path, const TensorSpec& spec) {
                                  auto bytes = encode_floats_le(spec.values);
                                  bytes.push_back(0xa5U);
                                  write_bytes(path, bytes);
                              });
        run_file_failure_case(temporary.path(), specs, target, "directory in place of file",
                              [](const fs::path& path, const TensorSpec&) {
                                  std::error_code error;
                                  if (!fs::remove(path, error) || error ||
                                      !fs::create_directory(path, error) || error) {
                                      throw std::runtime_error(path.string() +
                                                               ": could not prepare directory replacement case");
                                  }
                              });
        run_file_failure_case(temporary.path(), specs, target, "NaN payload",
                              [](const fs::path& path, const TensorSpec& spec) {
                                  auto bytes = encode_floats_le(spec.values);
                                  encode_u32_le(0x7fc00000U, bytes.data());
                                  write_bytes(path, bytes);
                              });
        run_file_failure_case(temporary.path(), specs, target, "positive infinity payload",
                              [](const fs::path& path, const TensorSpec& spec) {
                                  auto bytes = encode_floats_le(spec.values);
                                  encode_u32_le(0x7f800000U, bytes.data());
                                  write_bytes(path, bytes);
                              });
    }
}

void test_unreadable_weight_and_recovery(const fs::path&) {
    TemporaryDirectory temporary;
    const auto specs = make_specs();
    write_complete_fixture(temporary.path(), specs);
    const fs::path weight_path = temporary.path() / specs[3].filename;
    const fs::perms original_permissions = fs::status(weight_path).permissions();
    fs::permissions(weight_path, fs::perms::none);

    try {
        std::ifstream probe(weight_path, std::ios::binary);
        if (probe) {
            // Privileged processes can bypass mode bits. Report the gap explicitly.
            fs::permissions(weight_path, original_permissions);
            std::cout << "test_model: skipped unreadable-file case: process bypasses permissions\n";
            return;
        }
        expect_load_error_with_context(temporary.path(), specs[3].filename,
                                       "unreadable weight file");
    } catch (...) {
        std::error_code ignored;
        fs::permissions(weight_path, original_permissions, fs::perm_options::replace, ignored);
        throw;
    }

    fs::permissions(weight_path, original_permissions);
    const MlpModel recovered = MlpModel::load(temporary.path());
    verify_all_views(recovered, specs, "recovered after unreadable file");
}

void test_invalid_model_directories(const fs::path&) {
    TemporaryDirectory temporary;
    const fs::path missing_directory = temporary.path() / "does_not_exist";
    expect(!fs::exists(missing_directory), "missing-directory fixture path must initially not exist");
    expect_load_error_with_context(missing_directory, missing_directory.string(),
                                   "model directory does not exist");

    const fs::path regular_file = temporary.path() / "directory_path_is_a_file";
    write_bytes(regular_file, {0x01U});
    expect_load_error_with_context(regular_file, regular_file.string(),
                                   "model directory path is a regular file");
}

struct TestCase {
    std::string_view name;
    void (*run)(const fs::path& reference_directory);
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "test_model: expected exactly one reference model directory argument\n";
        return 2;
    }

    const fs::path reference_directory(argv[1]);
    constexpr TestCase tests[] = {
        {"reference model views", test_reference_model},
        {"deterministic fixture and row-major mapping", test_deterministic_fixture_and_row_major_mapping},
        {"move construction", test_move_construction_getters_on_destination},
        {"per-file load failure matrix and recovery", test_file_failure_matrix},
        {"unreadable weight file and recovery", test_unreadable_weight_and_recovery},
        {"invalid model directories", test_invalid_model_directories},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.run(reference_directory);
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "test_model: " << test.name << ": " << error.what() << '\n';
            return 1;
        } catch (...) {
            std::cerr << "test_model: " << test.name << ": unknown failure\n";
            return 1;
        }
    }

    std::cout << "test_model: " << passed << " test groups passed\n";
    return 0;
}
