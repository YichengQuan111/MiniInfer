#pragma once
// Optional long-run checks. Output checks run after SDK ready publication;
// their CPU cost belongs to client delivery, never to CUDA event timings.
#include "miniinfer/depth/sdk.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace miniinfer::depth::replay_detail {
inline std::uint32_t bits(float value) noexcept {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof result);
    return result;
}
inline std::uint64_t full_hash(const DepthMap& map) noexcept {
    std::uint64_t value=14695981039346656037ULL;
    for(float item:map.values()) {
        const auto word=bits(item);
        for(unsigned shift=0;shift<32;shift+=8) {value^=(word>>shift)&255U;value*=1099511628211ULL;}
    }
    return value;
}
class OutputCheck final {
    struct Expected {
        std::size_t width,height;
        std::array<std::uint32_t,64> samples;
        std::uint64_t hash;
    };
    std::vector<Expected> expected_;
    std::atomic<std::size_t> sampled_{0}, full_{0}, failures_{0};
public:
    void add(const DepthMap& map) { // Startup only, before concurrent check calls.
        Expected item{map.width(),map.height(),{},full_hash(map)};
        for(std::size_t i=0;i<64;++i) item.samples[i]=bits(map.values()[i*(map.values().size()-1)/63]);
        expected_.push_back(item);
    }
    void check(std::size_t image,std::size_t index,const DepthMap& map) {
        const auto& expected=expected_.at(image);
        bool valid=map.width()==expected.width && map.height()==expected.height;
        ++sampled_;
        if(valid) for(std::size_t i=0;i<64;++i)
            valid=valid && bits(map.values()[i*(map.values().size()-1)/63])==expected.samples[i];
        // Every image receives a full check once per 128 complete sequence cycles.
        // Using index%128 would repeatedly select only one image in an 8-image sequence.
        if((index/expected_.size())%128==0) { ++full_;valid=valid&&full_hash(map)==expected.hash; }
        if(!valid) {++failures_;throw std::runtime_error("replay output differs from its image baseline");}
    }
    std::size_t sampled() const noexcept{return sampled_.load();}
    std::size_t full() const noexcept{return full_.load();}
    std::size_t failures() const noexcept{return failures_.load();}
};

// One-second sampled SDK ledger. Current/peak IO allocations are application
// allocations; context_required_bytes is TRT's static requirement, not a sum
// of every CUDA library/driver allocation. Final row follows SDK shutdown.
class SdkSampler final {
    DepthSdk& sdk_;
    std::atomic<int>& phase_;
    FILE* file_;
    std::chrono::steady_clock::time_point origin_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_=false;
    std::atomic<bool> failed_{false};
    std::thread worker_;
    void write(bool phase_boundary=false) noexcept {
        try {
            const auto s=sdk_.snapshot();
            static constexpr const char* phases[]={"startup","warmup","drain_warmup","steady","drain_steady","teardown"};
            const auto p=phase_.load();
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-origin_).count();
            const auto result=std::fprintf(file_,"%.17g,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%d\n",
                elapsed,phases[p],s.inflight,s.queued,s.executing,s.ready,s.reserved_bytes,s.accepted,s.succeeded,s.failed,
                s.delivered,s.abandoned,s.abandoned_completed,s.io_device_bytes,s.io_pinned_bytes,
                s.host_workspace_bytes,s.peak_io_device_bytes,s.peak_io_pinned_bytes,s.device_allocation_calls,
                s.pinned_allocation_calls,s.host_workspace_allocation_calls,s.context_required_bytes,s.engine_serialized_bytes,s.gpu_slots,s.shutdown_complete?1:0,phase_boundary?1:0);
            if(result<0 || std::fflush(file_)!=0) failed_=true;
        } catch(...) {failed_=true;}
    }
public:
    SdkSampler(DepthSdk& sdk,std::atomic<int>& phase,const std::filesystem::path& path)
        :sdk_(sdk),phase_(phase),file_(std::fopen(path.c_str(),"wx")),origin_(std::chrono::steady_clock::now()) {
        if(!file_) throw std::runtime_error("cannot exclusively create SDK telemetry CSV");
        const char* header="elapsed_ms,phase,inflight,queued,executing,ready,reserved_bytes,accepted,succeeded,failed,delivered,abandoned,abandoned_completed,io_device_bytes,io_pinned_bytes,host_workspace_bytes,peak_io_device_bytes,peak_io_pinned_bytes,device_allocation_calls,pinned_allocation_calls,host_workspace_allocation_calls,context_required_bytes,engine_serialized_bytes,gpu_slots,shutdown_complete,phase_boundary\n";
        if(std::fputs(header,file_)==EOF) {std::fclose(file_);throw std::runtime_error("SDK telemetry header failed");}
        write(); // Publish startup synchronously; do not depend on thread scheduling.
        try {
            worker_=std::thread([this] {
                std::unique_lock<std::mutex> lock(mutex_);
                while(!stopping_) {write();wake_.wait_for(lock,std::chrono::seconds(1),[this]{return stopping_;});}
            });
        } catch(...) {std::fclose(file_);throw;}
    }
    // Serialize the transition with periodic writes so the first row in the
    // new phase is its baseline, before the caller can submit phase work.
    void begin_phase(int phase) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        phase_=phase;
        write(true);
    }
    void stop() noexcept {
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}
        wake_.notify_all();
        if(worker_.joinable()) {worker_.join();write();}
    }
    ~SdkSampler() noexcept {stop();if(std::fclose(file_)!=0)failed_=true;}
    bool failed() const noexcept {return failed_.load();}
    SdkSampler(const SdkSampler&)=delete;
    SdkSampler& operator=(const SdkSampler&)=delete;
};
}
