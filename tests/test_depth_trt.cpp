#include "miniinfer/depth/trt_runner.hpp"
#include <filesystem>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
using namespace miniinfer::depth;
namespace {
void expect(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}
    expect(rejected,"expected a rejected artifact");
}
template<class F> void rejects_with(F action,const std::string& reason) {
    try { action(); }
    catch(const std::exception& error) {
        expect(std::string(error.what()).find(reason)!=std::string::npos,
               "artifact validation must precede CUDA device initialization");
        return;
    }
    throw std::runtime_error("expected an artifact validation failure");
}
void offline_rejection(const std::filesystem::path& dir) {
    const int unavailable_device=std::numeric_limits<int>::max();
    rejects_with([&]{TrtRunner missing(dir/"missing",unavailable_device);},"missing.manifest");
    write_text_exclusive(dir/"bad-schema.manifest","schema=wrong\n");
    rejects_with([&]{TrtRunner bad(dir/"bad-schema",unavailable_device);},"schema");
    Manifest metadata{{"schema","miniinfer-engine-1"},{"contract","depth-v1"},
        {"input","1x3x518x518-fp32"},{"output","1x518x518-fp32"},
        {"tf32","off"},{"precision","fp32"},{"precision_policy","fp32-v1"}};
    metadata["cache_key"]=sha256(manifest_text(metadata));
    metadata["engine_sha256"]=std::string(64,'0');
    write_text_exclusive(dir/"bad-hash","invalid engine bytes");
    write_text_exclusive(dir/"bad-hash.manifest",manifest_text(metadata));
    rejects_with([&]{TrtRunner bad(dir/"bad-hash",unavailable_device);},"engine_sha256");
}
}
int main(int argc,char** argv) {
    const auto dir=std::filesystem::temp_directory_path()/("miniinfer-trt-"+std::to_string(getpid()));
    try {
        std::filesystem::create_directory(dir);
        offline_rejection(dir);
        if(argc==1) {
            std::filesystem::remove_all(dir);
            std::cout<<"TRT offline-artifact contract passed (GPU test not configured)\n";return 0;
        }
        if(argc!=3) throw std::invalid_argument("usage: test_depth_trt [ENGINE IMAGE.ppm]");
        const auto image=read_ppm(argv[2]);
        std::optional<PreparedImage> input_storage;
        std::optional<RawDepth> saved;
        {
            TrtRunner runner(argv[1]);
            input_storage.emplace(preprocess(image,runner.input_size()));
            const auto& input=*input_storage;
            TrtTiming timing;
            saved=runner.run_raw(input,&timing);
            std::cout<<"CUDA timing h2d_ms="<<timing.h2d_ms<<" compute_ms="<<timing.compute_ms
                <<" d2h_ms="<<timing.d2h_ms<<" host_ms="<<timing.host_ms<<'\n';
            expect(std::isfinite(timing.compute_ms)&&std::isfinite(timing.h2d_ms)&&std::isfinite(timing.d2h_ms)
                   &&std::isfinite(timing.host_ms)&&timing.compute_ms>0&&timing.h2d_ms>=0
                   &&timing.d2h_ms>=0&&timing.host_ms>0,
                   "CUDA event timings must cover completed GPU execution");
            auto a=std::async(std::launch::async,[&]{return runner.run_raw(input);});
            auto b=std::async(std::launch::async,[&]{return runner.run_raw(input);});
            const auto result_a=a.get(),result_b=b.get();
            expect(result_a.values()==saved->values(),"concurrent output changed");
            expect(result_b.values()==saved->values(),"concurrent output changed");
            const auto stats=runner.resource_snapshot();
            expect(stats.device_allocation_calls==2&&stats.pinned_allocation_calls==2,
                   "reused slot allocated IO on the hot path");
            expect(stats.io_device_bytes==4*runner.input_size()*runner.input_size()*sizeof(float)
                   &&stats.io_pinned_bytes==stats.io_device_bytes&&stats.context_required_bytes>0,
                   "slot IO/context ledger mismatch");
        }
        const auto& input=*input_storage;
        const auto after_shutdown=postprocess(*saved,input.transform());
        expect(after_shutdown.width()==image.width(),"owned output lost after destruction");
        {
            TrtRunner allocated(argv[1],0,false);
            for(int i=0;i<2;++i) {
                const auto output=allocated.run_raw(input);
                expect(output.values()==saved->values(),"per-request allocation changed output");
            }
            const auto stats=allocated.resource_snapshot();
            expect(stats.io_device_bytes==0&&stats.io_pinned_bytes==0,"per-request IO not released");
            expect(stats.device_allocation_calls==4&&stats.pinned_allocation_calls==4,"allocation control did not allocate per request");
        }
        auto manifest=read_manifest(std::string(argv[1])+".manifest");
        manifest["contract"]="wrong-contract";
        write_text_exclusive(dir/"mismatch.manifest",manifest_text(manifest));
        rejects([&]{TrtRunner bad(dir/"mismatch");});
        manifest["contract"]="depth-v1";
        auto mislabeled=manifest;
        mislabeled["precision"]=manifest.at("precision")=="fp32"?"fp16":"fp32";
        mislabeled["precision_policy"]=mislabeled.at("precision")=="fp32"?"fp32-v1":"backbone-fp32-head-fp16-v2";
        std::filesystem::copy_file(argv[1],dir/"mislabeled");
        write_text_exclusive(dir/"mislabeled.manifest",manifest_text(mislabeled));
        rejects([&]{TrtRunner bad(dir/"mislabeled");});
        write_text_exclusive(dir/"corrupt","truncated engine");
        write_text_exclusive(dir/"corrupt.manifest",manifest_text(manifest));
        rejects([&]{TrtRunner bad(dir/"corrupt");});
        // A matching hash on invalid engine bytes reaches deserialization and
        // exercises partial initialization cleanup, then a valid reload.
        manifest["engine_sha256"]=file_sha256(dir/"corrupt");
        write_text_exclusive(dir/"invalid","truncated engine");
        write_text_exclusive(dir/"invalid.manifest",manifest_text(manifest));
        rejects([&]{TrtRunner bad(dir/"invalid");});
        {
            TrtRunner recovered(argv[1]);
            const auto recovered_raw=recovered.run_raw(input);
            expect(recovered_raw.values()==saved->values(),"reload after failure changed output");
        }
        std::filesystem::remove_all(dir);
        std::cout<<"TRT ownership, serialization, compatibility and recovery passed\n";return 0;
    }catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';std::filesystem::remove_all(dir);return 1;
    }
}
