#include "miniinfer/depth/ort_runner.hpp"

#include <onnxruntime_cxx_api.h>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace miniinfer::depth {
namespace {
Ort::SessionOptions options(int threads) {
    if (threads < 1) throw std::invalid_argument("ORT threads must be positive");
    Ort::SessionOptions configuration;
    configuration.SetIntraOpNumThreads(threads);
    configuration.SetInterOpNumThreads(1);
    configuration.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    configuration.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return configuration;
}

void verify_tensor(const Ort::TypeInfo& type, const std::vector<std::int64_t>& dimensions,
                   const char* description) {
    if (type.GetONNXType() != ONNX_TYPE_TENSOR) throw std::invalid_argument(std::string(description) + " must be tensor");
    const auto info = type.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != dimensions)
        throw std::invalid_argument(std::string(description) + " violates static FP32 depth-v1 shape");
}
}  // namespace

class OrtRunner::Impl final {
public:
    Impl(const std::filesystem::path& path, int threads)
        : environment(ORT_LOGGING_LEVEL_WARNING, "miniinfer_depth"), configuration(options(threads)),
          session(environment, path.c_str(), configuration), thread_count(threads) {
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::invalid_argument("depth model must have exactly one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        if (std::string(input_name.get()) != "image" || std::string(output_name.get()) != "depth")
            throw std::invalid_argument("depth model names must be image and depth");
        verify_tensor(session.GetInputTypeInfo(0), {1, 3, 518, 518}, "input image");
        verify_tensor(session.GetOutputTypeInfo(0), {1, 518, 518}, "output depth");
    }
    Ort::Env environment;
    Ort::SessionOptions configuration;
    Ort::Session session;
    int thread_count;
};

OrtRunner::OrtRunner(const std::filesystem::path& model_path, int threads) {
    if (threads < 1) throw std::invalid_argument("ORT threads must be positive");
    if (!std::filesystem::is_regular_file(model_path)) throw std::runtime_error("model file missing: " + model_path.string());
    try {
        impl_ = std::make_unique<Impl>(model_path, threads);
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(std::string("ORT model initialization failed: ") + error.what());
    }
}
OrtRunner::~OrtRunner() = default;
OrtRunner::OrtRunner(OrtRunner&&) noexcept = default;
OrtRunner& OrtRunner::operator=(OrtRunner&&) noexcept = default;

RawDepth OrtRunner::run_raw(const PreparedImage& input) const {
    if (!impl_) throw std::logic_error("ORT runner was moved from");
    validate_geometry(input.transform());
    if (input.values().size() != input_elements) throw std::invalid_argument("prepared input storage length mismatch");
    validate_finite(input.values());
    // Private copy avoids mutable ORT tensor API exposing caller storage.
    std::vector<float> storage = input.values();
    const std::array<std::int64_t, 4> shape{1, 3, 518, 518};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, storage.data(), storage.size(), shape.data(), shape.size());
    const char* inputs[] = {"image"};
    const char* outputs[] = {"depth"};
    try {
        auto results = impl_->session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        if (results.size() != 1 || !results[0].IsTensor()) throw std::runtime_error("ORT returned invalid output");
        const auto info = results[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || info.GetShape() != std::vector<std::int64_t>{1, 518, 518}
            || info.GetElementCount() != raw_elements) throw std::runtime_error("ORT output violates depth-v1 contract");
        const float* values = results[0].GetTensorData<float>();
        return RawDepth(std::vector<float>(values, values + raw_elements));
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(std::string("ORT model execution failed: ") + error.what());
    }
}
DepthMap OrtRunner::run(const RgbImage& image) const {
    const auto prepared = preprocess(image);
    return postprocess(run_raw(prepared), prepared.transform());
}
int OrtRunner::threads() const {
    if (!impl_) throw std::logic_error("ORT runner was moved from");
    return impl_->thread_count;
}
}  // namespace miniinfer::depth
