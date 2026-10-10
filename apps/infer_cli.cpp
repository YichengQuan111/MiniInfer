#include "miniinfer/cpu_executor.hpp"
#include "miniinfer/data_io.hpp"
#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/version.hpp"
#include "miniinfer/workspace.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaxBatch = 256;

struct Options final {
    std::filesystem::path model_directory;
    std::filesystem::path input_file;
    std::size_t batch = 0;
};

void print_usage() {
    std::cout << "Usage: infer_cli --model DIR --input FILE --batch N\n"
                 "       infer_cli --help | --version\n"
                 "Run digit MLP inference on N normalized 64-feature FP32 inputs.\n"
                 "N must be in [1,256].\n";
}

int report_error(std::string_view message, int exit_code = 2) {
    std::cerr << "error: " << message << '\n';
    return exit_code;
}

bool is_option(std::string_view token) {
    return token.size() >= 2 && token[0] == '-' && token[1] == '-';
}

bool parse_batch(std::string_view text, std::size_t& batch) {
    if (text.empty()) {
        return false;
    }
    for (const char character : text) {
        if (character < '0' || character > '9') {
            return false;
        }
    }

    unsigned long long parsed = 0;
    const auto conversion = std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
    if (conversion.ec != std::errc{} || conversion.ptr != text.data() + text.size() ||
        parsed < 1 || parsed > kMaxBatch) {
        return false;
    }
    batch = static_cast<std::size_t>(parsed);
    return true;
}

bool take_value(int argc, char* argv[], int& index, std::string_view option,
                std::string_view& value) {
    if (index + 1 >= argc) {
        report_error("missing value for '" + std::string(option) + "'");
        return false;
    }
    const std::string_view candidate{argv[index + 1]};
    if (is_option(candidate) || candidate == "-h") {
        report_error("missing value for '" + std::string(option) + "'");
        return false;
    }
    value = candidate;
    ++index;
    return true;
}

bool parse_options(int argc, char* argv[], Options& options, int& error_code) {
    bool have_model = false;
    bool have_input = false;
    bool have_batch = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help" || argument == "-h" || argument == "--version") {
            error_code = report_error("'" + std::string(argument) +
                                      "' must be used without other arguments");
            return false;
        }

        if (argument == "--model") {
            if (have_model) {
                error_code = report_error("duplicate option '--model'");
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value)) {
                return false;
            }
            options.model_directory = std::filesystem::path(std::string(value));
            have_model = true;
        } else if (argument == "--input") {
            if (have_input) {
                error_code = report_error("duplicate option '--input'");
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value)) {
                return false;
            }
            options.input_file = std::filesystem::path(std::string(value));
            have_input = true;
        } else if (argument == "--batch") {
            if (have_batch) {
                error_code = report_error("duplicate option '--batch'");
                return false;
            }
            std::string_view value;
            if (!take_value(argc, argv, index, argument, value)) {
                return false;
            }
            if (!parse_batch(value, options.batch)) {
                error_code = report_error("invalid value for '--batch': '" + std::string(value) +
                                          "' (expected an integer in [1,256])");
                return false;
            }
            have_batch = true;
        } else {
            error_code = report_error("unknown argument '" + std::string(argument) + "'");
            return false;
        }
    }

    if (!have_model) {
        error_code = report_error("missing required parameter '--model'");
        return false;
    }
    if (!have_input) {
        error_code = report_error("missing required parameter '--input'");
        return false;
    }
    if (!have_batch) {
        error_code = report_error("missing required parameter '--batch'");
        return false;
    }
    return true;
}

std::string serialize_result(const miniinfer::MlpResult& result) {
    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(std::numeric_limits<float>::max_digits10);
    json << "{\"batch\":" << result.batch() << ",\"labels\":[";

    const auto& labels = result.labels();
    for (std::size_t row = 0; row < labels.size(); ++row) {
        if (row != 0) {
            json << ',';
        }
        json << labels[row];
    }

    const auto logits = result.logits();
    json << "],\"logits\":[";
    for (std::size_t row = 0; row < result.batch(); ++row) {
        if (row != 0) {
            json << ',';
        }
        json << '[';
        for (std::size_t column = 0; column < miniinfer::MlpModel::output_features; ++column) {
            if (column != 0) {
                json << ',';
            }
            const float value = logits.at(row, column);
            if (!std::isfinite(value)) {
                throw std::runtime_error("inference produced a non-finite logit");
            }
            json << value;
        }
        json << ']';
    }
    json << "]}\n";
    return json.str();
}

int run_inference(const Options& options) {
    try {
        const auto model = miniinfer::MlpModel::load(options.model_directory);
        auto input_buffer = miniinfer::load_f32_file(
            options.input_file, options.batch * miniinfer::MlpModel::input_features);
        std::vector<float> input_values(input_buffer.data(),
                                        input_buffer.data() + input_buffer.size());
        const miniinfer::MlpInput input(options.batch, std::move(input_values));
        miniinfer::Workspace workspace(kMaxBatch);
        const miniinfer::CpuExecutor executor;
        const auto result = executor.run(model, input, workspace);

        const std::string output = serialize_result(result);
        std::cout << output;
        return 0;
    } catch (const std::exception& error) {
        return report_error(error.what(), 1);
    } catch (...) {
        return report_error("unknown inference failure", 1);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        const std::string_view first{argv[1]};
        if (first == "--help" || first == "-h" || first == "--version") {
            if (argc != 2) {
                return report_error("unexpected extra argument '" +
                                    std::string(argv[2]) + "'");
            }
            if (first == "--help" || first == "-h") {
                print_usage();
                return 0;
            }
            std::cout << "MiniInfer " << miniinfer::version() << '\n';
            return 0;
        }
    }

    Options options;
    int error_code = 2;
    if (!parse_options(argc, argv, options, error_code)) {
        return error_code;
    }
    return run_inference(options);
}
