#include "miniinfer/depth/ort_runner.hpp"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace miniinfer::depth;
struct Arguments {
    std::filesystem::path model, image, tensor, output;
    std::optional<std::size_t> width, height;
    int threads = 4;
};
void help() {
    std::cout << "Usage: depth_cli --model MODEL --image IMAGE.ppm --output PREFIX [--threads 4]\n"
                 "       depth_cli --model MODEL --tensor INPUT.f32 --width W --height H --output PREFIX [--threads 4]\n"
                 "CPU FP32 ORT, static image[1,3,518,518] -> depth[1,518,518].\n"
                 "Tensor mode expects normalized NCHW little-endian FP32; W/H are original image dimensions.\n"
                 "Writes PREFIX.preprocessed.f32, PREFIX.raw.f32, PREFIX.depth.f32, PREFIX.json.\n"
                 "All four output paths must be absent; parent directory must exist.\n"
                 "Input image is PPM P6 RGB8 maxval255. Depth values are relative; bigger means nearer.\n";
}
std::size_t positive(const std::string& value, const std::string& name) {
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0)
        throw std::invalid_argument(name + " must be a positive integer");
    return result;
}
Arguments parse(int argc, char** argv) {
    Arguments args;
    std::vector<std::string> seen;
    for (int index = 1; index < argc; index += 2) {
        const std::string name = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + name);
        for (const auto& previous : seen) if (previous == name) throw std::invalid_argument("duplicate option " + name);
        seen.push_back(name);
        const std::string value = argv[index + 1];
        if (value.empty()) throw std::invalid_argument("empty value for " + name);
        if (name == "--model") args.model = value;
        else if (name == "--image") args.image = value;
        else if (name == "--tensor") args.tensor = value;
        else if (name == "--output") args.output = value;
        else if (name == "--width") args.width = positive(value, name);
        else if (name == "--height") args.height = positive(value, name);
        else if (name == "--threads") {
            const auto threads = positive(value, name);
            if (threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("--threads out of range");
            args.threads = static_cast<int>(threads);
        } else throw std::invalid_argument("unknown option " + name);
    }
    if (args.model.empty() || args.output.empty()) throw std::invalid_argument("--model and --output are required");
    if (args.image.empty() == args.tensor.empty()) throw std::invalid_argument("specify exactly one of --image and --tensor");
    if (!args.tensor.empty()) {
        if (!args.width || !args.height) throw std::invalid_argument("--tensor requires --width and --height");
        validate_dimensions(*args.width, *args.height);
    } else if (args.width || args.height) throw std::invalid_argument("--width/--height are only valid with --tensor");
    return args;
}
std::string quoted(const std::string& text) {
    std::ostringstream stream;
    stream << '"';
    for (unsigned char value : text) {
        if (value == '"' || value == '\\') stream << '\\' << static_cast<char>(value);
        else if (value < 32) stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(value) << std::dec;
        else stream << static_cast<char>(value);
    }
    stream << '"';
    return stream.str();
}
std::filesystem::path suffixed(const std::filesystem::path& prefix, const char* suffix) {
    auto path = prefix;
    path += suffix;
    return path;
}
bool occupied(const std::filesystem::path& path) {
    return std::filesystem::symlink_status(path).type() != std::filesystem::file_type::not_found;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        help();
        return 0;
    }
    try {
        const auto args = parse(argc, argv);
        for (const char* suffix : {".preprocessed.f32", ".raw.f32", ".depth.f32", ".json"})
            if (occupied(suffixed(args.output, suffix))) throw std::runtime_error("refusing to overwrite: " + suffixed(args.output, suffix).string());
        const auto parent = args.output.parent_path().empty() ? std::filesystem::path{"."} : args.output.parent_path();
        if (!std::filesystem::is_directory(parent)) throw std::runtime_error("output parent directory does not exist: " + parent.string());
        const auto total_start = std::chrono::steady_clock::now();
        const auto prepared = [&] {
            if (!args.image.empty()) return preprocess(read_ppm(args.image));
            return PreparedImage(read_f32(args.tensor, input_elements), geometry(*args.width, *args.height));
        }();
        OrtRunner runner(args.model, args.threads);
        const auto model_start = std::chrono::steady_clock::now();
        const auto raw = runner.run_raw(prepared);
        const double model_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - model_start).count();
        const auto restored = postprocess(raw, prepared.transform());
        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
        write_f32(suffixed(args.output, ".preprocessed.f32"), prepared.values());
        write_f32(suffixed(args.output, ".raw.f32"), raw.values());
        write_f32(suffixed(args.output, ".depth.f32"), restored.values());
        const auto transform = prepared.transform();
        std::ostringstream metadata;
        metadata << std::setprecision(17)
                 << "{\n  \"contract_version\": \"depth-v1\",\n  \"backend\": \"ort_cpu_fp32\",\n"
                 << "  \"model_path\": " << quoted(std::filesystem::absolute(args.model).string()) << ",\n"
                 << "  \"source_path\": " << quoted(std::filesystem::absolute(args.image.empty() ? args.tensor : args.image).string()) << ",\n"
                 << "  \"threads\": " << runner.threads() << ",\n  \"geometry\": {"
                 << "\"original_width\": " << transform.original_width << ", \"original_height\": " << transform.original_height
                 << ", \"resized_width\": " << transform.resized_width << ", \"resized_height\": " << transform.resized_height
                 << ", \"pad_left\": " << transform.pad_left << ", \"pad_top\": " << transform.pad_top
                 << ", \"pad_right\": " << transform.pad_right << ", \"pad_bottom\": " << transform.pad_bottom << "},\n"
                 << "  \"input_shape\": [1,3,518,518],\n  \"raw_shape\": [1,518,518],\n  \"depth_shape\": ["
                 << restored.height() << ',' << restored.width() << "],\n"
                 << "  \"dtype\": \"float32\",\n  \"byte_order\": \"little-endian\",\n"
                 << "  \"semantic_direction\": \"bigger-is-nearer\",\n  \"tf32\": false,\n"
                 << "  \"model_seconds\": " << model_seconds << ",\n  \"total_seconds\": " << total_seconds << ",\n"
                 << "  \"seconds\": {\"model\": " << model_seconds << ", \"total\": " << total_seconds << "},\n"
                 << "  \"total_scope\": \"input read+preprocess+model initialization+inference+postprocess; output writes excluded\"\n}\n";
        write_text_exclusive(suffixed(args.output, ".json"), metadata.str());
        std::cout << "Saved depth-v1 artifacts to " << args.output.string() << ".*\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
