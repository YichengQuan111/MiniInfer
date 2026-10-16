#include "miniinfer/depth/preprocess.hpp"
#include "preprocess_cuda.hpp"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace miniinfer::depth;
void check(cudaError_t error, const char* operation) {
    if(error != cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { check(cudaMalloc(&data, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if(data) (void)cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if(value) { (void)cudaStreamSynchronize(value); (void)cudaStreamDestroy(value); } }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value), "create event"); }
    ~Event() { if(value) (void)cudaEventDestroy(value); }
};
std::size_t number(const std::string& text) {
    std::size_t value=0;
    const auto r=std::from_chars(text.data(),text.data()+text.size(),value);
    if(r.ec!=std::errc{} || r.ptr!=text.data()+text.size() || !value)
        throw std::invalid_argument("positive integer required");
    return value;
}
double percentile(const std::vector<float>& sorted, double p) {
    const double index=p*static_cast<double>(sorted.size()-1);
    const auto lo=static_cast<std::size_t>(index), hi=std::min(lo+1,sorted.size()-1);
    return sorted[lo]+(sorted[hi]-sorted[lo])*(index-static_cast<double>(lo));
}
std::string measure(const std::string& name, const std::function<cudaError_t()>& launch,
                    cudaStream_t stream, float* output, const std::vector<float>& reference,
                    std::size_t warmup, std::size_t iterations) {
    for(std::size_t i=0;i<warmup;++i) check(launch(), "warmup kernel");
    check(cudaStreamSynchronize(stream), "warmup completion");
    Event begin, end;
    std::vector<float> times;
    times.reserve(iterations);
    for(std::size_t i=0;i<iterations;++i) {
        check(cudaEventRecord(begin.value,stream), "record begin");
        check(launch(), "benchmark kernel");
        check(cudaEventRecord(end.value,stream), "record end");
        check(cudaEventSynchronize(end.value), "kernel completion");
        float ms=0;
        check(cudaEventElapsedTime(&ms,begin.value,end.value), "elapsed time");
        times.push_back(ms);
    }
    std::vector<float> observed(reference.size());
    check(cudaMemcpyAsync(observed.data(),output,observed.size()*sizeof(float),cudaMemcpyDeviceToHost,stream), "verification download");
    check(cudaStreamSynchronize(stream), "verification completion");
    double max_abs=0;
    std::size_t violations=0;
    for(std::size_t i=0;i<reference.size();++i) {
        const double difference=std::abs(static_cast<double>(observed[i])-reference[i]);
        max_abs=std::max(max_abs,difference);
        if(!std::isfinite(observed[i]) || difference>2e-6+1e-6*std::abs(reference[i])) ++violations;
    }
    if(violations) throw std::runtime_error(name+" violates frozen preprocessing tolerance");
    const double mean=std::accumulate(times.begin(),times.end(),0.0)/times.size();
    std::sort(times.begin(),times.end());
    std::ostringstream out;
    out<<std::setprecision(12)<<"{\"kernel\":\""<<name<<"\",\"iterations\":"<<iterations
       <<",\"cuda_mean_ms\":"<<mean<<",\"cuda_p50_ms\":"<<percentile(times,.5)
       <<",\"cuda_p95_ms\":"<<percentile(times,.95)<<",\"cuda_p99_ms\":"<<percentile(times,.99)
       <<",\"max_abs\":"<<max_abs<<",\"violations\":"<<violations<<"}";
    return out.str();
}
}
int main(int argc,char** argv) {
    try {
        std::filesystem::path input, output;
        std::size_t size=518, warmup=20, iterations=200;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if(i+1==argc) throw std::invalid_argument("option requires a value");
            const std::string value=argv[++i];
            if(key=="--image") input=value;
            else if(key=="--output") output=value;
            else if(key=="--size") size=number(value);
            else if(key=="--warmup") warmup=number(value);
            else if(key=="--iterations") iterations=number(value);
            else throw std::invalid_argument("unknown option: "+key);
        }
        validate_model_size(size);
        if(input.empty() || output.empty() || warmup>10000 || iterations>100000)
            throw std::invalid_argument("--image/--output required; warmup<=10000 and iterations<=100000");
        const auto image=read_ppm(input);
        const auto reference=preprocess(image,size);
        const auto g=reference.transform();
        const CudaPreprocessGeometry geometry{static_cast<std::uint32_t>(image.width()),static_cast<std::uint32_t>(image.height()),
            static_cast<std::uint32_t>(g.resized_width),static_cast<std::uint32_t>(g.resized_height),
            static_cast<std::uint32_t>(g.pad_left),static_cast<std::uint32_t>(g.pad_top),static_cast<std::uint32_t>(size)};
        // Path B preparation is outside the device microbenchmark; end-to-end
        // replay separately accounts for this CPU work and its allocation.
        std::vector<float> normalized(image.pixels().size());
        for(std::size_t i=0;i<normalized.size();++i) normalized[i]=static_cast<float>(image.pixels()[i])/255.0F;
        const auto resized=bilinear_half_pixel(normalized,image.width(),image.height(),3,g.resized_width,g.resized_height);
        std::vector<float> padded(3*size*size,0.0F);
        for(std::size_t y=0;y<g.resized_height;++y)
            std::copy_n(resized.begin()+static_cast<std::ptrdiff_t>(y*g.resized_width*3),g.resized_width*3,
                        padded.begin()+static_cast<std::ptrdiff_t>(((y+g.pad_top)*size+g.pad_left)*3));
        DeviceBuffer rgb(image.pixels().size()), hwc(padded.size()*sizeof(float)), nchw(reference.values().size()*sizeof(float));
        Stream stream;
        check(cudaMemcpyAsync(rgb.data,image.pixels().data(),image.pixels().size(),cudaMemcpyHostToDevice,stream.value), "RGB8 upload");
        check(cudaMemcpyAsync(hwc.data,padded.data(),padded.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value), "HWC upload");
        check(cudaStreamSynchronize(stream.value), "upload completion");
        auto* destination=static_cast<float*>(nchw.data);
        std::ostringstream report;
        report<<"{\"schema\":\"miniinfer-preprocess-micro-v1\",\"width\":"<<image.width()<<",\"height\":"<<image.height()
              <<",\"model_size\":"<<size<<",\"warmup\":"<<warmup<<",\"rgb8_upload_bytes\":"<<image.pixels().size()
              <<",\"fp32_upload_bytes\":"<<reference.values().size()*sizeof(float)
              <<",\"scope\":\"CUDA events around kernels only; upload, CPU work and TensorRT excluded\",\"results\":[";
        report<<measure("C_scalar",[&]{return enqueue_preprocess_rgb8(static_cast<const std::uint8_t*>(rgb.data),destination,geometry,CudaPreprocessKernel::Scalar,stream.value);},stream.value,destination,reference.values(),warmup,iterations)<<",";
        report<<measure("C_fused_rgb",[&]{return enqueue_preprocess_rgb8(static_cast<const std::uint8_t*>(rgb.data),destination,geometry,CudaPreprocessKernel::FusedRgb,stream.value);},stream.value,destination,reference.values(),warmup,iterations)<<",";
        report<<measure("B_normalize",[&]{return enqueue_normalize_hwc(static_cast<const float*>(hwc.data),destination,static_cast<std::uint32_t>(size),stream.value);},stream.value,destination,reference.values(),warmup,iterations)<<"]}\n";
        write_text_exclusive(output,report.str());
        std::cout<<report.str();
        return 0;
    } catch(const std::exception& e) { std::cerr<<"depth_preprocess_benchmark: "<<e.what()<<'\n'; return 1; }
}
