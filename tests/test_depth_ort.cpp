#include "miniinfer/depth/ort_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
std::string fixture(bool valid_shape, bool valid_name = true) {
    const std::string input_name = valid_name ? "image" : "wrong_image";
    std::string node = bytes(1, input_name) + bytes(2, "depth") + bytes(4, valid_shape ? "ReduceMean" : "Identity");
    if (valid_shape) {
        node += bytes(5, bytes(1, "axes") + integer(8, 1) + integer(20, 7));  // INTS
        node += bytes(5, bytes(1, "keepdims") + integer(3, 0) + integer(20, 2));  // INT
    }
    const auto input = value_info(input_name, valid_shape ? std::vector<std::uint64_t>{1, 3, 518, 518} : std::vector<std::uint64_t>{1, 3, 2, 2});
    const auto output = value_info("depth", valid_shape ? std::vector<std::uint64_t>{1, 518, 518} : std::vector<std::uint64_t>{1, 3, 2, 2});
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
    }
    expect(first->values()[0] == 3.0F, "runner destruction invalidated result");
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
