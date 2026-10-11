#include "miniinfer/depth/trt_runner.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc,char** argv) {
    using namespace miniinfer::depth;
    if(argc==2&&std::string(argv[1])=="--help") {
        std::cout<<"Usage: depth_engine_builder --onnx MODEL --output ENGINE --precision fp32|fp16\n";
        return 0;
    }
    try {
        std::filesystem::path onnx,output;
        std::string precision;
        for(int i=1;i<argc;i+=2) {
            if(i+1==argc) throw std::invalid_argument("missing argument value");
            const std::string key=argv[i];
            if(key=="--onnx"&&onnx.empty()) onnx=argv[i+1];
            else if(key=="--output"&&output.empty()) output=argv[i+1];
            else if(key=="--precision"&&precision.empty()) precision=argv[i+1];
            else throw std::invalid_argument("unknown or duplicate option: "+key);
        }
        if(onnx.empty()||output.empty()||(precision!="fp32"&&precision!="fp16"))
            throw std::invalid_argument("require --onnx, --output, --precision fp32|fp16");
        TrtBuildOptions options;options.fp16=precision=="fp16";
        std::cout<<manifest_text(build_trt_engine(onnx,output,options));
        return 0;
    } catch(const std::exception& error) { std::cerr<<"error: "<<error.what()<<'\n';return 1; }
}
