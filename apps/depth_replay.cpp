#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/artifact.hpp"
#include "miniinfer/depth/preprocess.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <set>
#include <thread>
#include <vector>

namespace {
using namespace miniinfer::depth;
using Clock = std::chrono::steady_clock;
using Counts = std::map<std::string, std::size_t>;
constexpr std::size_t max_requests = 1000000;
Counts empty_counts() { return {{"succeeded",0},{"busy",0},{"closed",0},{"invalid_input",0},
    {"failure",0},{"timeout",0},{"not_submitted_capacity",0},{"not_submitted_late",0}}; }
double since(Clock::time_point origin) {
    return std::chrono::duration<double, std::milli>(Clock::now()-origin).count();
}
std::string decimal(double value) { std::ostringstream out; out<<std::setprecision(17)<<value; return out.str(); }
std::string counts_json(const Counts& counts) {
    std::string out="{";
    for(const auto& [key,value]:counts) {
        if(out.size()>1) out+=",";
        out+='"'+key+"\":"+std::to_string(value);
    }
    return out+"}";
}
std::size_t integer(const std::string& value, bool allow_zero=false) {
    std::size_t out=0;
    auto parsed=std::from_chars(value.data(),value.data()+value.size(),out);
    if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || (!allow_zero && !out))
        throw std::invalid_argument("invalid integer: "+value);
    return out;
}
double real(const std::string& value, bool allow_zero=false) {
    std::size_t used=0;
    double out=std::stod(value,&used);
    if(used!=value.size() || !std::isfinite(out) || out<0 || (!allow_zero && !out))
        throw std::invalid_argument("invalid finite number: "+value);
    return out;
}
struct Args {
    std::filesystem::path package, output;
    std::vector<std::filesystem::path> paths;
    std::string backend;
    double duration=10, warmup=2, rate=10, max_lag_ms=100;
    std::size_t sdk_workers=4, workers=4, max_outstanding=4, queue=4, budget=128ULL*1024*1024;
    std::size_t slots=1, consumer_delay_ms=0;
    bool asynchronous=false, reuse_host=true, reuse_gpu=true;
};
Args parse(int argc,char** argv) {
    Args out; std::set<std::string> seen;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(!seen.insert(key).second) throw std::invalid_argument("duplicate option: "+key);
        if(key=="--images") {
            while(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0) out.paths.emplace_back(argv[++i]);
            if(out.paths.empty()) throw std::invalid_argument("--images needs PPM paths");
            continue;
        }
        if(++i==argc) throw std::invalid_argument("missing value: "+key);
        const std::string value=argv[i];
        if(key=="--package") out.package=value;
        else if(key=="--backend") out.backend=value;
        else if(key=="--output") out.output=value;
        else if(key=="--images-list") {
            if(std::filesystem::file_size(value)>8ULL*1024*1024) throw std::invalid_argument("image list exceeds 8MiB");
            std::ifstream stream(value);
            if(!stream) throw std::invalid_argument("cannot read image list");
            std::string line;
            while(std::getline(stream,line)) {
                if(line.size()>8192) throw std::invalid_argument("image list line exceeds 8192 characters");
                if(!line.empty() && line.back()=='\r') line.pop_back();
                if(line.empty() || line[0]=='#') continue;
                std::filesystem::path path=line;
                if(path.is_relative()) path=std::filesystem::path(value).parent_path()/path;
                out.paths.push_back(path);
                if(out.paths.size()>1024) throw std::invalid_argument("at most 1024 images");
            }
        }
        else if(key=="--duration") out.duration=real(value);
        else if(key=="--warmup") out.warmup=real(value,true);
        else if(key=="--rate") out.rate=real(value);
        else if(key=="--max-lag-ms") out.max_lag_ms=real(value);
        else if(key=="--sdk-workers") out.sdk_workers=integer(value);
        else if(key=="--workers") out.workers=integer(value);
        else if(key=="--max-outstanding") out.max_outstanding=integer(value);
        else if(key=="--queue") out.queue=integer(value);
        else if(key=="--budget") out.budget=integer(value);
        else if(key=="--slots") out.slots=integer(value);
        else if(key=="--consumer-delay-ms") out.consumer_delay_ms=integer(value,true);
        else if(key=="--async-sdk" || key=="--reuse-host" || key=="--reuse-gpu") {
            const auto flag=integer(value,true);
            if(flag>1) throw std::invalid_argument("boolean options require 0 or 1");
            if(key=="--async-sdk") out.asynchronous=flag;
            else if(key=="--reuse-host") out.reuse_host=flag;
            else out.reuse_gpu=flag;
        }
        else throw std::invalid_argument("unknown option: "+key);
    }
    if(out.package.empty() || out.output.empty() || out.paths.empty())
        throw std::invalid_argument("--package, --output and --images/--images-list required");
    if(seen.count("--images") && seen.count("--images-list")) throw std::invalid_argument("choose one image source");
    if(!out.backend.empty() && out.backend!="ort" && out.backend!="trt") throw std::invalid_argument("backend must be ort/trt");
    if((out.slots!=1&&out.slots!=2)||out.consumer_delay_ms>60000 || (out.consumer_delay_ms&&!out.asynchronous))
        throw std::invalid_argument("slots must be 1/2; consumer delay <=60000 and requires async SDK");
    if(out.duration>86400 || out.warmup>3600 || out.rate>100000 || out.sdk_workers>256 || out.workers>256 ||
       out.max_outstanding>100000 || out.queue>100000 || out.paths.size()>1024 ||
       std::ceil(out.rate*out.duration)+std::ceil(out.rate*out.warmup)>max_requests)
        throw std::invalid_argument("replay exceeds bounded duration/rate/threads/requests/images");
    return out;
}
std::string error_status(const SdkError& error) {
    switch(error.code()) {
    case SdkErrorCode::Busy: return "busy";
    case SdkErrorCode::Closed: return "closed";
    case SdkErrorCode::InvalidInput: return "invalid_input";
    default: return "failure";
    }
}
// Synchronous control retains the V2.1 observation boundary. Async mode submits
// directly on the generator thread and records SDK terminal publication time.
struct SyncAdapter {
    DepthSdk& sdk;
    std::string execute(const RgbImage& image, DepthTiming& timing, Clock::time_point origin, double& ready) {
        try {
            auto result=sdk.run(image,&timing);
            ready=since(origin); // Result still owns its output: exclude destruction.
            return "succeeded";
        }
        catch(const SdkError& error) {
            ready=since(origin);
            return error_status(error);
        }
        catch(const std::exception&) { ready=since(origin); return "failure"; }
    }
    std::string collect(DepthRequest& request, DepthTiming& timing, Clock::time_point origin,
                        double& ready, std::size_t delay_ms) {
        auto set_ready=[&] {
            ready=timing.ready_at==Clock::time_point{} ? since(origin) :
                std::chrono::duration<double,std::milli>(timing.ready_at-origin).count();
        };
        try {
            request.wait();
            if(delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            auto result=request.get(&timing);
            set_ready();
            return "succeeded";
        } catch(const SdkError& error) { set_ready(); return error_status(error); }
        catch(const std::exception&) { set_ready(); return "failure"; }
    }
};
struct Record {
    std::size_t id, index, image;
    double planned, generated;
    std::optional<double> submit, ready, delivered;
    std::string status;
    DepthTiming timing;
    std::optional<DepthRequest> request;
    std::optional<std::uint64_t> sdk_id;
};
struct Phase {
    Counts counts=empty_counts();
    std::size_t planned=0, submitted=0, succeeded_in_window=0, completed_in_window=0, drain_completed=0, peak_outstanding=0, h2d_bytes=0;
    std::size_t undelivered_at_window_end=0;
    double duration_ms=0, drain_ms=0;
    std::string json() const {
        return "{\"planned\":"+std::to_string(planned)+",\"submitted\":"+std::to_string(submitted)+
            ",\"counts\":"+counts_json(counts)+",\"completed_in_window\":"+std::to_string(completed_in_window)+
            ",\"succeeded_in_window\":"+std::to_string(succeeded_in_window)+
            ",\"outstanding_at_window_end\":"+std::to_string(undelivered_at_window_end)+
            ",\"drain_completed\":"+std::to_string(drain_completed)+
            ",\"peak_outstanding\":"+std::to_string(peak_outstanding)+
            ",\"h2d_bytes\":"+std::to_string(h2d_bytes)+
            ",\"rgb8_h2d_bytes\":0,\"fp32_h2d_bytes\":"+std::to_string(h2d_bytes)+
            ",\"duration_ms\":"+decimal(duration_ms)+",\"drain_ms\":"+decimal(drain_ms)+"}";
    }
};
Phase replay(const std::string& phase, double seconds, const Args& args,
             const std::vector<RgbImage>& images, SyncAdapter& adapter, FILE* csv, std::size_t& next_id) {
    Phase result; result.duration_ms=seconds*1000;
    result.planned=static_cast<std::size_t>(std::ceil(seconds*args.rate));
    if(!result.planned) return result;
    std::mutex mutex; std::condition_variable available, drained;
    std::deque<Record> pending; std::size_t outstanding=0;
    bool stopping=false, write_failed=false;
    Clock::time_point origin;
    auto emit=[&](const Record& record) {
        // Called under mutex. Stream rows immediately: no per-request row cache.
        int ok=std::fprintf(csv,"%s,%zu,%zu,%zu,%.17g,%.17g,",phase.c_str(),record.id,record.index,record.image,record.planned,record.generated);
        for(const auto& stamp:{record.submit,record.ready,record.delivered}) {
            if(stamp) ok=std::fprintf(csv,"%.17g",*stamp)<0 ? -1:ok;
            if(std::fputc(',',csv)==EOF) ok=-1;
        }
        if(std::fprintf(csv,"%s,",record.status.c_str())<0) ok=-1;
        if(record.submit) {
            const auto& t=record.timing;
            if(std::fprintf(csv,"%d,%.17g,%.17g,%.17g,%.17g,%.17g,",t.accepted?1:0,
                t.admission_ms,t.queue_ms,t.preprocess_ms,t.backend_ms,t.postprocess_ms)<0) ok=-1;
        } else if(std::fprintf(csv,",,,,,,")<0) ok=-1;
        if(record.sdk_id && std::fprintf(csv,"%llu",static_cast<unsigned long long>(*record.sdk_id))<0) ok=-1;
        if(std::fputc('\n',csv)==EOF) ok=-1;
        if(ok<0) write_failed=true;
        ++result.counts.at(record.status);
        result.h2d_bytes += record.timing.h2d_bytes;
        if(record.submit) {
            if(*record.delivered>=result.duration_ms) ++result.undelivered_at_window_end;
            if(*record.ready<result.duration_ms) {
                ++result.completed_in_window;
                if(record.status=="succeeded") ++result.succeeded_in_window;
            } else ++result.drain_completed;
        }
    };
    std::vector<std::thread> callers;
    auto worker=[&] {
        for(;;) {
            Record record;
            {
                std::unique_lock<std::mutex> lock(mutex);
                available.wait(lock,[&]{return stopping || !pending.empty();});
                if(pending.empty()) return;
                record=std::move(pending.front()); pending.pop_front();
            }
            double ready=0;
            if(args.asynchronous) {
                record.status=adapter.collect(*record.request,record.timing,origin,ready,args.consumer_delay_ms);
            } else {
                record.submit=since(origin);
                record.status=adapter.execute(images.at(record.image),record.timing,origin,ready);
            }
            record.ready=ready;
            record.delivered=since(origin);
            {
                std::lock_guard<std::mutex> lock(mutex);
                emit(record); --outstanding;
            }
            drained.notify_one();
        }
    };
    try {
        for(std::size_t i=0;i<args.workers;++i) callers.emplace_back(worker);
        origin=Clock::now(); // Decode/model and caller thread creation precede schedule.
        const auto end=origin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        for(std::size_t i=0;i<result.planned;++i) {
            const double planned=1000*static_cast<double>(i)/args.rate;
            std::this_thread::sleep_until(origin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(planned)));
            Record record{++next_id,i,i%images.size(),planned,since(origin),{},{},{},"",{},{},{}};
            std::lock_guard<std::mutex> lock(mutex);
            if(record.generated>=result.duration_ms || record.generated-planned>args.max_lag_ms) {
                record.status="not_submitted_late"; emit(record);
            } else if(outstanding>=args.max_outstanding) {
                record.status="not_submitted_capacity"; emit(record);
            } else {
                ++result.submitted;
                if(args.asynchronous) {
                    record.submit=since(origin);
                    try {
                        record.request.emplace(adapter.sdk.submit(images.at(record.image)));
                        record.sdk_id=record.request->id();
                        record.timing.accepted=true;
                    } catch(const SdkError& error) { record.status=error_status(error); }
                    catch(const std::exception&) { record.status="failure"; }
                    if(!record.status.empty()) {
                        record.ready=since(origin);record.delivered=record.ready;
                        emit(record);
                        continue;
                    }
                }
                pending.push_back(std::move(record)); ++outstanding;
                result.peak_outstanding=std::max(result.peak_outstanding,outstanding);
                available.notify_one();
            }
        }
        std::this_thread::sleep_until(end);
        std::cout<<"{\"event\":\"phase\",\"phase\":\"drain\",\"source\":\""<<phase<<"\"}"<<std::endl;
        {
            std::unique_lock<std::mutex> lock(mutex);
            drained.wait(lock,[&]{return outstanding==0;});
            stopping=true;
        }
        available.notify_all();
        for(auto& thread:callers) thread.join();
    } catch(...) {
        { std::lock_guard<std::mutex> lock(mutex); stopping=true; }
        available.notify_all();
        for(auto& thread:callers) thread.join();
        throw;
    }
    result.drain_ms=std::max(0.0,since(origin)-result.duration_ms);
    if(write_failed || std::fflush(csv)!=0) throw std::runtime_error("CSV write/flush failed");
    return result;
}
struct FileCloser { void operator()(FILE* file) const noexcept { if(file) std::fclose(file); } };
}

int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--help") {
        std::cout<<"depth_replay --package DIR --images PPM...|--images-list FILE --output NEW.csv\n"
            "[--backend ort|trt --duration 10 --warmup 2 --rate 10 --workers 4 --max-outstanding 4\n"
            "--async-sdk 1 --slots 1 --reuse-host 1 --reuse-gpu 1 --consumer-delay-ms 0\n"
            " --sdk-workers 4 --queue 4 --budget 134217728 --max-lag-ms 100]\n"
            "Fixed open-loop schedule; async SDK mode records actual terminal publication and request handles.\n";
        return 0;
    }
    try {
        const auto main_start=Clock::now(); const auto args=parse(argc,argv);
        std::unique_ptr<FILE,FileCloser> csv(std::fopen(args.output.c_str(),"wx"));
        if(!csv) throw std::runtime_error("cannot exclusively create CSV");
        if(std::fprintf(csv.get(),"phase,request_id,index,image_index,planned_ms,generated_ms,submit_ms,ready_ms,delivered_ms,status,accepted,admission_ms,queue_ms,preprocess_ms,backend_ms,postprocess_ms,request_handle_id\n")<0)
            throw std::runtime_error("CSV header write failed");
        std::vector<RgbImage> images; std::size_t image_bytes=0;
        for(const auto& path:args.paths) {
            images.push_back(read_ppm(path)); image_bytes+=images.back().pixels().size();
            if(image_bytes>256ULL*1024*1024) throw std::invalid_argument("decoded sequence exceeds 256MiB");
        }
        SdkConfig config; config.package_dir=args.package; config.queue_capacity=args.queue; config.byte_budget=args.budget; config.worker_count=args.sdk_workers;
        config.gpu_slots=args.slots;config.reuse_host_workspace=args.reuse_host;config.reuse_gpu_buffers=args.reuse_gpu;
        DepthSdk sdk(config);
        if(!args.backend.empty() && sdk.snapshot().backend!=args.backend) throw std::invalid_argument("package backend mismatch");
        const auto input_size=contract_model_size(require_field(read_manifest(args.package/"manifest.txt"),"contract"));
        SyncAdapter adapter{sdk}; std::size_t next_id=0;
        std::cout<<std::setprecision(17)<<"{\"event\":\"ready\",\"startup_ms\":"<<since(main_start)
            <<",\"decoded_image_bytes\":"<<image_bytes<<",\"image_count\":"<<images.size()<<",\"sdk_worker_count\":"<<config.worker_count
            <<",\"fp32_model_input_bytes\":"<<(3*input_size*input_size*sizeof(float))<<",\"rgb8_input_bytes\":[";
        for(std::size_t i=0;i<images.size();++i) { if(i) std::cout<<","; std::cout<<images[i].pixels().size(); }
        std::cout<<"]}"<<std::endl;
        std::cout<<"{\"event\":\"phase\",\"phase\":\"warmup\"}"<<std::endl;
        const auto warmup=replay("warmup",args.warmup,args,images,adapter,csv.get(),next_id);
        std::cout<<"{\"event\":\"phase\",\"phase\":\"steady\"}"<<std::endl;
        const auto steady=replay("steady",args.duration,args,images,adapter,csv.get(),next_id);
        sdk.shutdown(); const auto snapshot=sdk.snapshot();
        std::cout<<"{\"event\":\"summary\",\"schema\":\"miniinfer-depth-replay-v2\",\"rate\":"<<args.rate
            <<",\"max_lag_ms\":"<<args.max_lag_ms<<",\"image_count\":"<<images.size()
            <<",\"async_sdk\":"<<(args.asynchronous?"true":"false")<<",\"slots\":"<<args.slots
            <<",\"reuse_host\":"<<(args.reuse_host?"true":"false")<<",\"reuse_gpu\":"<<(args.reuse_gpu?"true":"false")
            <<",\"consumer_delay_ms\":"<<args.consumer_delay_ms
            <<",\"warmup\":"<<warmup.json()<<",\"steady\":"<<steady.json()
            <<",\"sdk_snapshot\":{\"accepted\":"<<snapshot.accepted<<",\"succeeded\":"<<snapshot.succeeded
            <<",\"failed\":"<<snapshot.failed<<",\"delivered\":"<<snapshot.delivered
            <<",\"rejected_busy\":"<<snapshot.rejected_busy<<",\"rejected_closed\":"<<snapshot.rejected_closed
            <<",\"rejected_invalid\":"<<snapshot.rejected_invalid<<",\"inflight\":"<<snapshot.inflight
            <<",\"reserved_bytes\":"<<snapshot.reserved_bytes<<",\"ready\":"<<snapshot.ready
            <<",\"queued\":"<<snapshot.queued<<",\"executing\":"<<snapshot.executing
            <<",\"abandoned\":"<<snapshot.abandoned<<",\"abandoned_completed\":"<<snapshot.abandoned_completed
            <<",\"fatal_backend\":"<<(snapshot.fatal_backend?"true":"false")
            <<",\"gpu_slots\":"<<snapshot.gpu_slots<<",\"host_workspace_bytes\":"<<snapshot.host_workspace_bytes
            <<",\"host_workspace_allocation_calls\":"<<snapshot.host_workspace_allocation_calls
            <<",\"io_device_bytes\":"<<snapshot.io_device_bytes<<",\"io_pinned_bytes\":"<<snapshot.io_pinned_bytes
            <<",\"peak_io_device_bytes\":"<<snapshot.peak_io_device_bytes<<",\"peak_io_pinned_bytes\":"<<snapshot.peak_io_pinned_bytes
            <<",\"device_allocation_calls\":"<<snapshot.device_allocation_calls<<",\"pinned_allocation_calls\":"<<snapshot.pinned_allocation_calls
            <<",\"context_required_bytes\":"<<snapshot.context_required_bytes<<"}}"<<std::endl;
        return 0;
    } catch(const std::exception& error) { std::cerr<<"depth_replay: "<<error.what()<<'\n';return 1; }
}
