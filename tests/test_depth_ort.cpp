#include "miniinfer/depth/ort_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace miniinfer::depth;
void expect(bool value, const char* description) {
    if (!value) throw std::runtime_error(description);
}
template<class Callable> void fails(Callable callable, const char* description) {
    try { callable(); } catch (const std::exception&) { return; }
    throw std::runtime_error(description);
}
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        std::string pattern = (std::filesystem::temp_directory_path() / "miniinfer-depth-ort-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* created = ::mkdtemp(buffer.data());
        if (!created) throw std::runtime_error("cannot create test directory");
        path = created;
    }
    ~Temporary() { std::error_code error; std::filesystem::remove_all(path, error); }
};
// Tiny deterministic ONNX protobuf fixtures, no Python/weights/download needed.
std::string varint(std::uint64_t value) {
    std::string result;
    do {
        auto byte = static_cast<unsigned char>(value & 127);
        value >>= 7;
        if (value) byte |= 128;
        result.push_back(static_cast<char>(byte));
    } while (value);
    return result;
}
std::string integer(unsigned field, std::uint64_t value) { return varint(field * 8) + varint(value); }
std::string bytes(unsigned field, const std::string& value) { return varint(field * 8 + 2) + varint(value.size()) + value; }
std::string value_info(const std::string& name, const std::vector<std::uint64_t>& dimensions) {
    std::string shape;
    for (auto dimension : dimensions) shape += bytes(1, integer(1, dimension));
    const auto tensor = integer(1, 1) + bytes(2, shape);  // TensorProto.FLOAT
    return bytes(1, name) + bytes(2, bytes(1, tensor));
}
std::string fixture(bool valid_shape, bool valid_name = true, std::uint64_t size = 518) {
    const std::string input_name = valid_name ? "image" : "wrong_image";
    std::string node = bytes(1, input_name) + bytes(2, "depth") + bytes(4, valid_shape ? "ReduceMean" : "Identity");
    if (valid_shape) {
        node += bytes(5, bytes(1, "axes") + integer(8, 1) + integer(20, 7));  // INTS
        node += bytes(5, bytes(1, "keepdims") + integer(3, 0) + integer(20, 2));  // INT
    }
    const auto input = value_info(input_name, valid_shape ? std::vector<std::uint64_t>{1, 3, size, size} : std::vector<std::uint64_t>{1, 3, 2, 2});
    const auto output = value_info("depth", valid_shape ? std::vector<std::uint64_t>{1, size, size} : std::vector<std::uint64_t>{1, 3, 2, 2});
    const auto graph = bytes(1, node) + bytes(2, "miniinfer_depth_test") + bytes(11, input) + bytes(12, output);
    return integer(1, 9) + bytes(2, "miniinfer_test") + bytes(7, graph) + bytes(8, integer(2, 17));
}
void backend_contract() {
    Temporary temporary;
    fails([&] { OrtRunner runner(temporary.path / "missing.onnx"); }, "missing model accepted");
    const auto invalid = temporary.path / "invalid.onnx";
    write_text_exclusive(invalid, "invalid protobuf");
    fails([&] { OrtRunner runner(invalid); }, "invalid ONNX accepted");
    const auto wrong = temporary.path / "wrong.onnx";
    write_text_exclusive(wrong, fixture(false));
    fails([&] { OrtRunner runner(wrong); }, "incorrect static model shape accepted");
    const auto named = temporary.path / "named.onnx";
    write_text_exclusive(named, fixture(true, false));
    fails([&] { OrtRunner runner(named); }, "incorrect model input name accepted");
    const auto valid = temporary.path / "valid.onnx";
    write_text_exclusive(valid, fixture(true));
    fails([&] { OrtRunner runner(valid, 0); }, "zero threads accepted");
    std::optional<RawDepth> first;
    {
        OrtRunner runner(valid, 1);
        const PreparedImage input(std::vector<float>(input_elements, 3.0F), geometry(2, 2));
        first.emplace(runner.run_raw(input));
        expect(std::all_of(first->values().begin(), first->values().end(), [](float v) { return v == 3.0F; }), "fixture FP32 execution");
        const auto second = runner.run_raw(PreparedImage(std::vector<float>(input_elements, 6.0F), geometry(2, 2)));
        expect(second.values()[0] == 6.0F && first->values()[0] == 3.0F, "later inference overwrote earlier raw output");
        const auto restored = runner.run(RgbImage(2, 2, std::vector<std::uint8_t>(12, 128)));
        expect(restored.width() == 2 && restored.height() == 2 && restored.values().size() == 4, "complete ORT run restoration");
        std::promise<void> start;
        const auto ready = start.get_future().share();
        std::vector<std::future<RawDepth>> concurrent;
        for (int worker = 0; worker < 4; ++worker) {
            concurrent.push_back(std::async(std::launch::async, [&, worker, ready] {
                expect(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                       "concurrent ORT start gate timed out");
                std::optional<RawDepth> previous;
                for (int iteration = 0; iteration < 8; ++iteration) {
                    const float value = static_cast<float>(10 * worker + iteration + 1);
                    auto result = runner.run_raw(PreparedImage(std::vector<float>(input_elements, value), geometry(2, 2)));
                    expect(std::all_of(result.values().begin(), result.values().end(), [value](float v) { return v == value; }),
                           "concurrent shared ORT session mixed request outputs");
                    if (previous)
                        expect(previous->values()[0] == value - 1, "concurrent ORT inference overwrote earlier output");
                    previous.emplace(std::move(result));
                }
                return std::move(*previous);
            }));
        }
        start.set_value();
        for (int worker = 0; worker < 4; ++worker) {
            const auto result = concurrent[worker].get();
            expect(result.values()[0] == 10 * worker + 8, "concurrent ORT terminal output");
        }
    }
    expect(first->values()[0] == 3.0F, "runner destruction invalidated result");
    const auto small = temporary.path / "small.onnx";
    write_text_exclusive(small, fixture(true, true, 392));
    OrtRunner small_runner(small, 1);
    expect(small_runner.input_size() == 392, "392 model shape not inferred");
    const auto small_result = small_runner.run_raw(PreparedImage(std::vector<float>(3 * 392 * 392, 4.0F), geometry(2, 2, 392)));
    expect(small_result.values()[0] == 4.0F,
           "392 execution failed");
    fails([&] { small_runner.run_raw(PreparedImage(std::vector<float>(input_elements), geometry(2, 2))); }, "cross-size input accepted");
}
double numeric(const char* text) {
    std::size_t consumed = 0;
    const double result = std::stod(text, &consumed);
    if (consumed != std::string(text).size() || !std::isfinite(result) || result < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    return result;
}
void compare(const std::vector<float>& actual, const std::vector<float>& reference,
             double atol, double rtol, double allowed, const char* name) {
    expect(actual.size() == reference.size(), "golden shape mismatch");
    std::size_t failures = 0;
    double maximum = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = std::abs(static_cast<double>(actual[index]) - reference[index]);
        maximum = std::max(maximum, error);
        if (error > atol + rtol * std::abs(static_cast<double>(reference[index]))) ++failures;
    }
    const double ratio = static_cast<double>(failures) / static_cast<double>(actual.size());
    std::cout << name << " max_abs=" << maximum << " violation_ratio=" << ratio << '\n';
    if (ratio > allowed) throw std::runtime_error(std::string(name) + " exceeds explicitly supplied FP32 tolerances");
}
void golden(const std::filesystem::path& model, const std::filesystem::path& sample,
            double atol, double rtol, double allowed) {
    if (allowed > 1) throw std::invalid_argument("allowed ratio must be <=1");
    const auto image = read_ppm(sample / "original.ppm");
    const auto input = preprocess(image);
    expect(input.values() == read_f32(sample / "preprocessed.f32", input_elements), "C++ preprocess differs from NumPy bit-exact reference");
    const RawDepth python_raw(read_f32(sample / "raw.f32", raw_elements));
    const auto reference = read_f32(sample / "restored.f32", image.width() * image.height());
    const auto python_restored = postprocess(python_raw, input.transform());
    expect(python_restored.values() == reference, "C++ restoration differs from NumPy bit-exact reference");
    OrtRunner runner(model, 4);
    const auto actual = runner.run_raw(input);
    compare(actual.values(), python_raw.values(), atol, rtol, allowed, "raw");
    const auto actual_restored = postprocess(actual, input.transform());
    compare(actual_restored.values(), reference, atol, rtol, allowed, "restored");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 6) throw std::invalid_argument("usage: test_depth_ort [MODEL SAMPLE_DIR ATOL RTOL ALLOWED_RATIO]");
        backend_contract();
        if (argc == 6) golden(argv[1], argv[2], numeric(argv[3]), numeric(argv[4]), numeric(argv[5]));
        std::cout << "Depth ORT contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Depth ORT contract failed: " << error.what() << '\n';
        return 1;
    }
}
