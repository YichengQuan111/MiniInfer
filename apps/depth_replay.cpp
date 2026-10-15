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
        else throw std::invalid_argument("unknown option: "+key);
    }
    if(out.package.empty() || out.output.empty() || out.paths.empty())
        throw std::invalid_argument("--package, --output and --images/--images-list required");
    if(seen.count("--images") && seen.count("--images-list")) throw std::invalid_argument("choose one image source");
    if(!out.backend.empty() && out.backend!="ort" && out.backend!="trt") throw std::invalid_argument("backend must be ort/trt");
    if(out.duration>86400 || out.warmup>3600 || out.rate>100000 || out.sdk_workers>256 || out.workers>256 ||
       out.max_outstanding>100000 || out.queue>100000 || out.paths.size()>1024 ||
       std::ceil(out.rate*out.duration)+std::ceil(out.rate*out.warmup)>max_requests)
        throw std::invalid_argument("replay exceeds bounded duration/rate/threads/requests/images");
    return out;
}
// Execution adapter is the only place coupled to V1 run(). V1 does not expose
// admission/dequeue/backend-ready times: return is the observed ready boundary.
// Keep decoded images alive until all callers join. V2.3 can replace this adapter.
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
            switch(error.code()) {
            case SdkErrorCode::Busy: return "busy";
            case SdkErrorCode::Closed: return "closed";
            case SdkErrorCode::InvalidInput: return "invalid_input";
            default: return "failure";
            }
        }
        catch(const std::exception&) { ready=since(origin); return "failure"; }
    }
};
struct Record {
    std::size_t id, index, image;
    double planned, generated;
    std::optional<double> submit, ready, delivered;
    std::string status;
    DepthTiming timing;
};
struct Phase {
    Counts counts=empty_counts();
    std::size_t planned=0, submitted=0, succeeded_in_window=0, completed_in_window=0, drain_completed=0, peak_outstanding=0, h2d_bytes=0;
    double duration_ms=0, drain_ms=0;
    std::string json() const {
        return "{\"planned\":"+std::to_string(planned)+",\"submitted\":"+std::to_string(submitted)+
            ",\"counts\":"+counts_json(counts)+",\"completed_in_window\":"+std::to_string(completed_in_window)+
            ",\"succeeded_in_window\":"+std::to_string(succeeded_in_window)+
            ",\"outstanding_at_window_end\":"+std::to_string(drain_completed)+
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
            if(std::fprintf(csv,"%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",t.accepted?1:0,
                t.admission_ms,t.queue_ms,t.preprocess_ms,t.backend_ms,t.postprocess_ms)<0) ok=-1;
        } else if(std::fprintf(csv,",,,,,\n")<0) ok=-1;
        if(ok<0) write_failed=true;
        ++result.counts.at(record.status);
        result.h2d_bytes += record.timing.h2d_bytes;
        if(record.submit) {
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
            record.submit=since(origin);
            double ready=0;
            record.status=adapter.execute(images.at(record.image),record.timing,origin,ready);
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
            Record record{++next_id,i,i%images.size(),planned,since(origin),{},{},{},"",{}};
            std::lock_guard<std::mutex> lock(mutex);
            if(record.generated>=result.duration_ms || record.generated-planned>args.max_lag_ms) {
                record.status="not_submitted_late"; emit(record);
            } else if(outstanding>=args.max_outstanding) {
                record.status="not_submitted_capacity"; emit(record);
            } else {
                pending.push_back(std::move(record)); ++outstanding; ++result.submitted;
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
            " --sdk-workers 4 --queue 4 --budget 134217728 --max-lag-ms 100]\n"
            "Fixed open-loop schedule; owned run-return is ready; acceptance and host stage timings recorded.\n";
        return 0;
    }
    try {
        const auto main_start=Clock::now(); const auto args=parse(argc,argv);
        std::unique_ptr<FILE,FileCloser> csv(std::fopen(args.output.c_str(),"wx"));
        if(!csv) throw std::runtime_error("cannot exclusively create CSV");
        if(std::fprintf(csv.get(),"phase,request_id,index,image_index,planned_ms,generated_ms,submit_ms,ready_ms,delivered_ms,status,accepted,admission_ms,queue_ms,preprocess_ms,backend_ms,postprocess_ms\n")<0)
            throw std::runtime_error("CSV header write failed");
        std::vector<RgbImage> images; std::size_t image_bytes=0;
        for(const auto& path:args.paths) {
            images.push_back(read_ppm(path)); image_bytes+=images.back().pixels().size();
            if(image_bytes>256ULL*1024*1024) throw std::invalid_argument("decoded sequence exceeds 256MiB");
        }
        SdkConfig config; config.package_dir=args.package; config.queue_capacity=args.queue; config.byte_budget=args.budget; config.worker_count=args.sdk_workers;
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
        std::cout<<"{\"event\":\"summary\",\"schema\":\"miniinfer-depth-replay-v1\",\"rate\":"<<args.rate
            <<",\"max_lag_ms\":"<<args.max_lag_ms<<",\"image_count\":"<<images.size()
            <<",\"warmup\":"<<warmup.json()<<",\"steady\":"<<steady.json()
            <<",\"sdk_snapshot\":{\"accepted\":"<<snapshot.accepted<<",\"succeeded\":"<<snapshot.succeeded
            <<",\"failed\":"<<snapshot.failed<<",\"delivered\":"<<snapshot.delivered
            <<",\"rejected_busy\":"<<snapshot.rejected_busy<<",\"rejected_closed\":"<<snapshot.rejected_closed
            <<",\"rejected_invalid\":"<<snapshot.rejected_invalid<<",\"inflight\":"<<snapshot.inflight
            <<",\"reserved_bytes\":"<<snapshot.reserved_bytes<<"}}"<<std::endl;
        return 0;
    } catch(const std::exception& error) { std::cerr<<"depth_replay: "<<error.what()<<'\n';return 1; }
}
