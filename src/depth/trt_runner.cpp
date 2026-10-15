#include "miniinfer/depth/trt_runner.hpp"
#include "miniinfer/depth/profile.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <sys/utsname.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace miniinfer::depth {
namespace {
void check(cudaError_t code, const char* operation) {
    if(code!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(code));
}
class Device {
    int previous=0;bool changed=false;
public:
    explicit Device(int device) {
        check(cudaGetDevice(&previous),"get CUDA device");
        if(previous!=device) { check(cudaSetDevice(device),"set CUDA device");changed=true; }
    }
    ~Device() { if(changed) (void)cudaSetDevice(previous); }
};
class Logger final:public nvinfer1::ILogger {
public:
    void log(Severity level,const char* text) noexcept override {
        if(level<=Severity::kWARNING) { try { std::cerr<<"TensorRT: "<<text<<'\n'; } catch(...) {} }
    }
};
template<class T> std::unique_ptr<T> require(T* value,const char* message) {
    if(!value) throw std::runtime_error(message);
    return std::unique_ptr<T>(value);
}
class Stream {
public:
    cudaStream_t value=nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"create CUDA stream"); }
    ~Stream() { if(value) { (void)cudaStreamSynchronize(value);(void)cudaStreamDestroy(value); } }
};
class Allocation {
public:
    void* pointer=nullptr;
    explicit Allocation(std::size_t bytes) { check(cudaMalloc(&pointer,bytes),"allocate CUDA buffer"); }
    ~Allocation() { if(pointer) (void)cudaFree(pointer); }
};
class Event {
public:
    cudaEvent_t value=nullptr;
    Event() { check(cudaEventCreate(&value),"create CUDA event"); }
    ~Event() { if(value) (void)cudaEventDestroy(value); }
};
class Pinned {
public:
    float* pointer=nullptr;
    explicit Pinned(std::size_t elements) {
        void* allocated=nullptr;
        check(cudaMallocHost(&allocated,elements*sizeof(float)),"allocate pinned buffer");
        pointer=static_cast<float*>(allocated);
    }
    ~Pinned() { if(pointer) (void)cudaFreeHost(pointer); }
};
std::size_t verify_engine(const nvinfer1::ICudaEngine& engine) {
    if(engine.getNbIOTensors()!=2) throw std::invalid_argument("engine must expose exactly two tensors");
    const auto in=engine.getTensorShape("image"),out=engine.getTensorShape("depth");
    if(in.nbDims!=4||in.d[0]!=1||in.d[1]!=3||(in.d[2]!=518&&in.d[2]!=392)||in.d[3]!=in.d[2]
       ||out.nbDims!=3||out.d[0]!=1||out.d[1]!=in.d[2]||out.d[2]!=in.d[2]
       ||engine.getTensorDataType("image")!=nvinfer1::DataType::kFLOAT
       ||engine.getTensorDataType("depth")!=nvinfer1::DataType::kFLOAT
       ||engine.getTensorIOMode("image")!=nvinfer1::TensorIOMode::kINPUT
       ||engine.getTensorIOMode("depth")!=nvinfer1::TensorIOMode::kOUTPUT
       ||engine.getTensorLocation("image")!=nvinfer1::TensorLocation::kDEVICE
       ||engine.getTensorLocation("depth")!=nvinfer1::TensorLocation::kDEVICE
       ||engine.getTensorFormat("image")!=nvinfer1::TensorFormat::kLINEAR
       ||engine.getTensorFormat("depth")!=nvinfer1::TensorFormat::kLINEAR)
        throw std::invalid_argument("engine violates static FP32 IO depth-v1 contract");
    return static_cast<std::size_t>(in.d[2]);
}
void verify_manifest(const Manifest& manifest) {
    require_equal(manifest,"schema","miniinfer-engine-1");
    const auto side=std::to_string(contract_model_size(require_field(manifest,"contract")));
    require_equal(manifest,"input","1x3x"+side+"x"+side+"-fp32");
    require_equal(manifest,"output","1x"+side+"x"+side+"-fp32");
    require_equal(manifest,"tf32","off");
    const auto precision=require_field(manifest,"precision");
    if(precision!="fp32"&&precision!="fp16") throw std::invalid_argument("unknown engine precision");
    require_equal(manifest,"precision_policy",precision=="fp32"?"fp32-v1":"backbone-fp32-head-fp16-v2");
    auto identity=manifest;
    identity.erase("cache_key");identity.erase("build_seconds");identity.erase("engine_sha256");
    require_equal(manifest,"cache_key",sha256(manifest_text(identity)));
}
void verify_environment(const Manifest& manifest,int device) {
    for(const auto& pair:trt_environment(device)) require_equal(manifest,pair.first,pair.second);
}
}

Manifest trt_environment(int device) {
    if(device<0) throw std::invalid_argument("CUDA device must be nonnegative");
    Device guard(device);
    cudaDeviceProp property{};
    check(cudaGetDeviceProperties(&property,device),"query CUDA device");
    int runtime=0,driver=0;
    check(cudaRuntimeGetVersion(&runtime),"query CUDA runtime");
    check(cudaDriverGetVersion(&driver),"query CUDA driver");
    utsname platform{};
    if(uname(&platform)!=0) throw std::runtime_error("query host platform failed");
    return {{"gpu",property.name},{"compute_capability",std::to_string(property.major)+"."+std::to_string(property.minor)},
            {"cuda_runtime",std::to_string(runtime)},{"cuda_driver",std::to_string(driver)},
            {"tensorrt",std::to_string(getInferLibVersion())},
            {"tensorrt_headers",std::to_string(NV_TENSORRT_VERSION)},
            {"platform",std::string(platform.sysname)+"-"+platform.machine}};
}

Manifest build_trt_engine(const std::filesystem::path& onnx,const std::filesystem::path& output,
                          const TrtBuildOptions& options) {
    if(!std::filesystem::is_regular_file(onnx)) throw std::invalid_argument("ONNX file missing");
    if(options.workspace_bytes<16*1024*1024||options.workspace_bytes>8ULL*1024*1024*1024)
        throw std::invalid_argument("workspace must be between 16 MiB and 8 GiB");
    for(const auto& suffix:{"", ".manifest", ".layers.json"})
        if(std::filesystem::exists(output.string()+suffix)) throw std::invalid_argument("engine output already exists");
    Device guard(options.device);
    Logger logger;
    auto builder=require(nvinfer1::createInferBuilder(logger),"create builder failed");
    auto network=require(builder->createNetworkV2(0),"create network failed");
    auto config=require(builder->createBuilderConfig(),"create build config failed");
    auto parser=require(nvonnxparser::createParser(*network,logger),"create ONNX parser failed");
    if(!parser->parseFromFile(onnx.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("ONNX parse failed; see TensorRT diagnostics");
    if(network->getNbInputs()!=1 || network->getNbOutputs()!=1)
        throw std::invalid_argument("ONNX must expose one input and output");
    const auto in=network->getInput(0)->getDimensions(), out=network->getOutput(0)->getDimensions();
    if(in.nbDims!=4 || in.d[0]!=1 || in.d[1]!=3 || (in.d[2]!=518&&in.d[2]!=392) || in.d[3]!=in.d[2]
       || out.nbDims!=3 || out.d[0]!=1 || out.d[1]!=in.d[2] || out.d[2]!=in.d[2])
        throw std::invalid_argument("ONNX must use a supported static square shape");
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,options.workspace_bytes);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    std::size_t constrained=0;
    if(options.fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        // Development-set evidence found raw-map error from the FP16 backbone.
        // Keep the transformer backbone in FP32; the convolutional depth head
        // remains eligible for FP16. Inspector records the actual mixed plan.
        for(int i=0;i<network->getNbLayers();++i) {
            auto* layer=network->getLayer(i);
            const std::string name=layer->getName();
            const auto type=layer->getType();
            bool floating=false;
            for(int j=0;j<layer->getNbOutputs();++j)
                floating=floating||layer->getOutput(j)->getType()==nvinfer1::DataType::kFLOAT;
            if(floating&&type!=nvinfer1::LayerType::kCONSTANT&&name.find("/depth_head/")==std::string::npos) {
                layer->setPrecision(nvinfer1::DataType::kFLOAT);
                for(int j=0;j<layer->getNbOutputs();++j)
                    if(layer->getOutput(j)->getType()==nvinfer1::DataType::kFLOAT)
                        layer->setOutputType(j,nvinfer1::DataType::kFLOAT);
                ++constrained;
            }
        }
    }
    const auto started=std::chrono::steady_clock::now();
    auto plan=require(builder->buildSerializedNetwork(*network,*config),"TensorRT build failed");
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    auto runtime=require(nvinfer1::createInferRuntime(logger),"create runtime failed");
    auto engine=require(runtime->deserializeCudaEngine(plan->data(),plan->size()),"verify built engine failed");
    const auto size=verify_engine(*engine);
    const auto side=std::to_string(size);
    auto inspector=require(engine->createEngineInspector(),"create inspector failed");
    const char* layers=inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    if(!layers) throw std::runtime_error("engine layer inspection failed");
    Manifest metadata=trt_environment(options.device);
    metadata.insert({{"schema","miniinfer-engine-1"},{"contract",std::string(model_contract(size))},
        {"input","1x3x"+side+"x"+side+"-fp32"},{"output","1x"+side+"x"+side+"-fp32"},
        {"precision",options.fp16?"fp16":"fp32"},{"tf32","off"},
        {"workspace_bytes",std::to_string(options.workspace_bytes)},
        {"fp32_constrained_layers",std::to_string(constrained)},
        {"precision_policy",options.fp16?"backbone-fp32-head-fp16-v2":"fp32-v1"},
        {"onnx_sha256",file_sha256(onnx)}, {"engine_sha256",sha256({static_cast<const char*>(plan->data()),plan->size()})},
        {"build_seconds",std::to_string(seconds)}});
    auto identity=metadata;
    identity.erase("build_seconds");identity.erase("engine_sha256");
    metadata["cache_key"]=sha256(manifest_text(identity));
    write_text_exclusive(output,{static_cast<const char*>(plan->data()),plan->size()});
    write_text_exclusive(output.string()+".layers.json",layers);
    write_text_exclusive(output.string()+".manifest",manifest_text(metadata));
    return metadata;
}

class TrtRunner::Impl {
public:
    int device;
    Manifest metadata;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::unique_ptr<Allocation> input,output;
    std::unique_ptr<Pinned> host_input,host_output;
    std::unique_ptr<Event> begin,h2d_end,compute_end,d2h_end;
    std::unique_ptr<Stream> stream; // destroyed first: wait before buffers/context
    std::mutex mutex;
    bool failed=false;
    std::size_t size=0;
    bool reuse_buffers=true;
    std::size_t context_required_bytes=0;
    std::atomic<std::size_t> device_bytes{0}, pinned_bytes{0}, peak_device{0}, peak_pinned{0};
    std::atomic<std::size_t> device_allocations{0}, pinned_allocations{0};
    void allocate_buffers() {
        const auto raw_elements=size*size, input_elements=3*raw_elements;
        input=std::make_unique<Allocation>(input_elements*sizeof(float));
        device_bytes += input_elements*sizeof(float); ++device_allocations;
        peak_device.store(std::max(peak_device.load(),device_bytes.load()));
        output=std::make_unique<Allocation>(raw_elements*sizeof(float));
        device_bytes += raw_elements*sizeof(float); ++device_allocations;
        peak_device.store(std::max(peak_device.load(),device_bytes.load()));
        host_input=std::make_unique<Pinned>(input_elements);
        pinned_bytes += input_elements*sizeof(float); ++pinned_allocations;
        peak_pinned.store(std::max(peak_pinned.load(),pinned_bytes.load()));
        host_output=std::make_unique<Pinned>(raw_elements);
        pinned_bytes += raw_elements*sizeof(float); ++pinned_allocations;
        peak_pinned.store(std::max(peak_pinned.load(),pinned_bytes.load()));
        if(!context->setTensorAddress("image",input->pointer)||!context->setTensorAddress("depth",output->pointer))
            throw std::runtime_error("bind engine tensor addresses failed");
    }
    void release_buffers() noexcept {
        host_output.reset();host_input.reset();output.reset();input.reset();
        pinned_bytes=0;device_bytes=0;
    }
    Impl(std::vector<std::uint8_t> plan,Manifest validated,int target,bool reuse)
        :device(target),metadata(std::move(validated)),reuse_buffers(reuse) {
        Device guard(device);
        verify_environment(metadata,device);
        runtime=require(nvinfer1::createInferRuntime(logger),"create runtime failed");
        engine=require(runtime->deserializeCudaEngine(plan.data(),plan.size()),"load engine failed");
        size=verify_engine(*engine);
        require_equal(metadata,"contract",std::string(model_contract(size)));
        context=require(engine->createExecutionContext(),"create context failed");
        const auto required=engine->getDeviceMemorySizeV2();
        if(required<0) throw std::runtime_error("invalid context memory requirement");
        context_required_bytes=static_cast<std::size_t>(required);
        begin=std::make_unique<Event>();h2d_end=std::make_unique<Event>();
        compute_end=std::make_unique<Event>();d2h_end=std::make_unique<Event>();
        stream=std::make_unique<Stream>();
        if(reuse_buffers) allocate_buffers();
    }
    ~Impl() {
        // CUDA resources belong to their original device even when the final
        // SDK owner shuts down on another host thread/device.
        int previous=device;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        stream.reset();d2h_end.reset();compute_end.reset();h2d_end.reset();begin.reset();
        release_buffers();
        context.reset();engine.reset();runtime.reset();
        if(previous!=device) (void)cudaSetDevice(previous);
    }
};
TrtRunner::TrtRunner(const std::filesystem::path& path,int device,bool reuse_buffers) {
    if(device<0) throw std::invalid_argument("CUDA device must be nonnegative");
    // Reject missing, malformed or corrupt files before initializing CUDA.
    auto metadata=read_manifest(path.string()+".manifest");
    verify_manifest(metadata);
    auto plan=read_artifact(path);
    require_equal(metadata,"engine_sha256",sha256({reinterpret_cast<const char*>(plan.data()),plan.size()}));
    // Keep the selected device active while partially built Impl members unwind.
    Device guard(device);
    impl_=std::make_unique<Impl>(std::move(plan),std::move(metadata),device,reuse_buffers);
}
TrtRunner::~TrtRunner()=default;
const Manifest& TrtRunner::manifest() const noexcept { return impl_->metadata; }
std::size_t TrtRunner::input_size() const noexcept { return impl_->size; }
TrtResources TrtRunner::resource_snapshot() const noexcept {
    return {impl_->device_bytes.load(),impl_->pinned_bytes.load(),impl_->peak_device.load(),impl_->peak_pinned.load(),
            impl_->device_allocations.load(),impl_->pinned_allocations.load(),impl_->context_required_bytes};
}
RawDepth TrtRunner::run_raw(const PreparedImage& prepared,TrtTiming* timing) {
    ProfileRange profile("depth.trt.h2d_execute_d2h");
    const auto host_start=std::chrono::steady_clock::now();
    const auto raw_elements=impl_->size*impl_->size, input_elements=3*raw_elements;
    validate_geometry(prepared.transform());
    if(prepared.values().size()!=input_elements || geometry_size(prepared.transform())!=impl_->size)
        throw std::invalid_argument("prepared input size mismatch");
    validate_finite(prepared.values());
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(impl_->failed) throw std::runtime_error("TensorRT slot failed previously; reinitialize the runner");
    Device guard(impl_->device);
    struct Release {
        Impl& impl;
        bool active;
        ~Release() { if(active) impl.release_buffers(); }
    } release{*impl_,!impl_->reuse_buffers};
    try {
        if(!impl_->reuse_buffers) impl_->allocate_buffers();
        std::memcpy(impl_->host_input->pointer,prepared.values().data(),input_elements*sizeof(float));
        if(timing) check(cudaEventRecord(impl_->begin->value,impl_->stream->value),"record input event");
        check(cudaMemcpyAsync(impl_->input->pointer,impl_->host_input->pointer,input_elements*sizeof(float),
                              cudaMemcpyHostToDevice,impl_->stream->value),"H2D copy");
        if(timing) check(cudaEventRecord(impl_->h2d_end->value,impl_->stream->value),"record H2D event");
        if(!impl_->context->enqueueV3(impl_->stream->value)) throw std::runtime_error("TensorRT enqueue failed");
        if(timing) check(cudaEventRecord(impl_->compute_end->value,impl_->stream->value),"record compute event");
        check(cudaMemcpyAsync(impl_->host_output->pointer,impl_->output->pointer,raw_elements*sizeof(float),
                              cudaMemcpyDeviceToHost,impl_->stream->value),"D2H copy");
        if(timing) check(cudaEventRecord(impl_->d2h_end->value,impl_->stream->value),"record output event");
        check(cudaStreamSynchronize(impl_->stream->value),"wait for device completion");
        if(timing) {
            float h2d=0,compute=0,d2h=0;
            check(cudaEventElapsedTime(&h2d,impl_->begin->value,impl_->h2d_end->value),"time H2D");
            check(cudaEventElapsedTime(&compute,impl_->h2d_end->value,impl_->compute_end->value),"time compute");
            check(cudaEventElapsedTime(&d2h,impl_->compute_end->value,impl_->d2h_end->value),"time D2H");
            timing->h2d_ms=h2d;timing->compute_ms=compute;timing->d2h_ms=d2h;
        }
    } catch(...) {
        impl_->failed=true;
        (void)cudaStreamSynchronize(impl_->stream->value);
        throw;
    }
    RawDepth result(std::vector<float>(impl_->host_output->pointer,impl_->host_output->pointer+raw_elements));
    if(release.active) { impl_->release_buffers(); release.active=false; }
    if(timing) timing->host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-host_start).count();
    return result;
}
DepthMap TrtRunner::run(const RgbImage& image) {
    const auto prepared=preprocess(image,input_size());
    return postprocess(run_raw(prepared),prepared.transform());
}
} // namespace miniinfer::depth
