#include "miniinfer/depth/trt_runner.hpp"

#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Correctness artifact producer, not a benchmark. The Python driver validates
// frozen manifests and compares all files with the frozen precision thresholds.
// One engine/context per mode remains loaded across the entire sample list.
namespace {
using namespace miniinfer::depth;
namespace fs = std::filesystem;

struct Sample {
    std::string id;
    fs::path path;
    std::size_t width, height;
    std::string hash;
};
struct Mode { const char* name; PreprocessMode mode; };
const std::array<Mode, 4> modes{{
    {"A", PreprocessMode::Cpu},
    {"B", PreprocessMode::CudaPartial},
    {"C-basic", PreprocessMode::CudaFullBasic},
    {"C-fused", PreprocessMode::CudaFullFused}}};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::size_t number(const std::string& text) {
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("invalid unsigned integer: " + text);
    return value;
}
bool identifier(const std::string& value) {
    if (value.empty() || value.size() > 64) return false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        const bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (!alnum && (i == 0 || (c != '_' && c != '-'))) return false;
    }
    return true;
}
std::string quote(const std::string& value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += static_cast<char>(c);
    }
    return result + '"';
}
std::vector<Sample> sample_list(const fs::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read sample list: " + path.string());
    std::vector<Sample> samples;
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        require(!line.empty() && line.size() <= 16384, "invalid/empty sample-list row");
        std::array<std::string, 5> columns;
        std::istringstream row(line);
        for (auto& column : columns)
            require(static_cast<bool>(std::getline(row, column, '\t')), "sample row requires id/path/width/height/SHA256");
        std::string extra;
        require(!std::getline(row, extra, '\t') && line.back() != '\t', "extra sample-list column");
        require(identifier(columns[0]) && seen.insert(columns[0]).second, "invalid or duplicate sample id");
        const fs::path image(columns[1]);
        require(image.is_absolute() && fs::is_regular_file(image), "sample image must be an absolute file path");
        const auto width = number(columns[2]), height = number(columns[3]);
        validate_dimensions(width, height);
        require(columns[4].size() == 64 && columns[4].find_first_not_of("0123456789abcdef") == std::string::npos,
            "sample SHA256 must be lowercase hexadecimal");
        require(file_sha256(image) == columns[4], "sample image SHA256 mismatch: " + columns[0]);
        samples.push_back({columns[0], image, width, height, columns[4]});
    }
    require(input.eof() && !samples.empty(), "sample list must be nonempty and readable");
    return samples;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7)
            throw std::invalid_argument("usage: test_depth_gpu_pipeline ENGINE MODEL_SIZE PRECISION SAMPLES.tsv OUTPUT_DIR [DEVICE]");
        const fs::path engine = fs::absolute(argv[1]);
        const auto size = validate_model_size(number(argv[2]));
        const std::string precision(argv[3]);
        require(precision == "fp32" || precision == "fp16", "precision must be fp32 or fp16");
        const fs::path list = fs::absolute(argv[4]), output = fs::absolute(argv[5]);
        const auto device_value = argc == 7 ? number(argv[6]) : 0;
        require(device_value <= static_cast<std::size_t>(std::numeric_limits<int>::max()), "device index exceeds int");
        const auto samples = sample_list(list);
        require(fs::create_directory(output), "output directory must not exist: " + output.string());
        require(fs::create_directory(output / "cpu"), "cannot create CPU reference directory");
        std::vector<std::unique_ptr<TrtRunner>> runners;
        for (const auto& mode : modes) {
            require(fs::create_directory(output / mode.name), "cannot create mode directory");
            runners.push_back(std::make_unique<TrtRunner>(engine, static_cast<int>(device_value), true, mode.mode));
            require(runners.back()->input_size() == size, "engine model size differs from requested model size");
            require_equal(runners.back()->manifest(), "precision", precision);
            require_equal(runners.back()->manifest(), "contract", std::string(model_contract(size)));
        }
        std::ostringstream metadata;
        metadata << "{\"schema_version\":1,\"model_size\":" << size
            << ",\"precision\":" << quote(precision)
            << ",\"engine_sha256\":" << quote(require_field(runners.front()->manifest(), "engine_sha256"))
            << ",\"samples_list_sha256\":" << quote(file_sha256(list))
            << ",\"captured_nchw\":\"actual-device-input-readback\""
            << ",\"modes\":[\"A\",\"B\",\"C-basic\",\"C-fused\"],\"samples\":[";
        bool first = true;
        for (const auto& sample : samples) {
            const auto image = read_ppm(sample.path);
            require(image.width() == sample.width && image.height() == sample.height,
                "sample dimensions differ from list: " + sample.id);
            const auto cpu = preprocess(image, size);
            write_f32(output / "cpu" / (sample.id + ".nchw.f32"), cpu.values());
            for (std::size_t mode = 0; mode < modes.size(); ++mode) {
                std::vector<float> captured;
                const auto raw = runners[mode]->run_image_raw(image, nullptr, &captured);
                require(captured.size() == 3 * size * size, "captured NCHW shape differs: " + sample.id);
                require(raw.values().size() == size * size, "raw model output shape differs: " + sample.id);
                validate_finite(captured);
                validate_finite(raw.values());
                const auto restored = postprocess(raw, cpu.transform());
                require(restored.width() == image.width() && restored.height() == image.height(), "restored shape differs");
                validate_finite(restored.values());
                const auto prefix = output / modes[mode].name / sample.id;
                write_f32(prefix.string() + ".nchw.f32", captured);
                write_f32(prefix.string() + ".raw.f32", raw.values());
                write_f32(prefix.string() + ".restored.f32", restored.values());
            }
            if (!first) metadata << ',';
            first = false;
            metadata << "{\"id\":" << quote(sample.id) << ",\"width\":" << image.width()
                << ",\"height\":" << image.height() << ",\"source_sha256\":" << quote(sample.hash) << '}';
        }
        metadata << "]}\n";
        write_text_exclusive(output / "run.json", metadata.str());
        std::cout << "{\"completed\":true,\"model_size\":" << size
            << ",\"precision\":" << quote(precision) << ",\"samples\":" << samples.size()
            << ",\"modes\":4}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU pipeline artifact generation failed: " << error.what() << '\n';
        return 1;
    }
}
