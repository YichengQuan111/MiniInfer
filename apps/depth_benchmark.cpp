#include "miniinfer/depth/ort_runner.hpp"
#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/net.hpp"
#include "miniinfer/net/socket.hpp"
#ifdef MINIINFER_BENCHMARK_TRT
#include "miniinfer/depth/trt_runner.hpp"
#endif

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>

namespace {
using Clock = std::chrono::steady_clock;
using namespace miniinfer::depth;
using Counts = std::map<std::string, std::size_t>;
Counts counts() { return {{"succeeded",0},{"busy",0},{"closed",0},{"failure",0},{"timeout",0}}; }
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now()-start).count(); }
struct Args {
    std::string mode, backend, host = "127.0.0.1";
    std::filesystem::path model, package, image, output;
    std::size_t iterations = 0, warmup = 0, timeout_ms = 30000;
    std::uint16_t port = 0;
};
struct FileCloser { void operator()(FILE* file) const noexcept { if(file) std::fclose(file); } };
std::size_t number(const std::string& s, bool zero = false) {
    std::size_t n = 0;
    const auto r = std::from_chars(s.data(), s.data()+s.size(), n);
    if (r.ec != std::errc{} || r.ptr != s.data()+s.size() || (!zero && n == 0))
        throw std::invalid_argument("invalid integer: " + s);
    return n;
}
Args parse(int argc, char** argv) {
    Args a; std::set<std::string> seen;
    for (int i=1; i<argc; i+=2) {
        std::string key = argv[i];
        if (i+1 == argc || !seen.insert(key).second) throw std::invalid_argument("missing/duplicate option: "+key);
        std::string v = argv[i+1];
        if (key=="--mode") a.mode=v;
        else if (key=="--backend") a.backend=v;
        else if (key=="--model") a.model=v;
        else if (key=="--package") a.package=v;
        else if (key=="--image") a.image=v;
        else if (key=="--output") a.output=v;
        else if (key=="--host") a.host=v;
        else if (key=="--port") { auto n=number(v); if(n>65535) throw std::invalid_argument("port out of range"); a.port=static_cast<std::uint16_t>(n); }
        else if (key=="--iterations") a.iterations=number(v);
        else if (key=="--warmup") a.warmup=number(v,true);
        else if (key=="--timeout-ms") { a.timeout_ms=number(v); if(a.timeout_ms>3600000) throw std::invalid_argument("timeout exceeds one hour"); }
        else throw std::invalid_argument("unknown option: "+key);
    }
    if (a.mode!="raw" && a.mode!="sdk" && a.mode!="tcp") throw std::invalid_argument("mode must be raw/sdk/tcp");
    if (a.backend!="ort" && a.backend!="trt") throw std::invalid_argument("backend must be ort/trt");
    if(a.image.empty() || a.output.empty() || !a.iterations) throw std::invalid_argument("image/output/positive iterations required");
    if(a.iterations>std::numeric_limits<std::uint32_t>::max()-1 || a.warmup>std::numeric_limits<std::uint32_t>::max()-1-a.iterations)
        throw std::invalid_argument("request count exceeds uint32 id range");
    if(a.mode=="raw" && a.model.empty()) throw std::invalid_argument("raw requires --model");
    if(a.mode=="sdk" && a.package.empty()) throw std::invalid_argument("sdk requires --package");
    if(a.mode=="tcp" && !a.port) throw std::invalid_argument("tcp requires --port (external ready server)");
    return a;
}
std::string count_json(const Counts& c) {
    std::string result="{";
    for(const auto& [key,n]:c) { if(result.size()>1) result+=","; result+='"'+key+"\":"+std::to_string(n); }
    return result+"}";
}
std::string io_status(miniinfer::net::IoStatus s) {
    using I=miniinfer::net::IoStatus;
    if(s==I::Timeout) return "timeout";
    if(s==I::Eof || s==I::Truncated || s==I::Stopped) return "closed";
    return "failure";
}
} // namespace

int main(int argc, char** argv) {
    const auto main_start=Clock::now();
    if(argc==2 && std::string(argv[1])=="--help") {
        std::cout<<"depth_benchmark --mode raw|sdk|tcp --backend ort|trt --image PPM --iterations N --warmup N --output NEW.csv\n"
                    "raw: --model ONNX_OR_ENGINE; sdk: --package DIR; tcp: --host IPv4 --port PORT [--timeout-ms 30000]\n"
                    "CPU threads=4; SDK queue=4, admission bytes=128MiB; one synchronous caller and one GPU slot.\n";
        return 0;
    }
    try {
        auto a=parse(argc,argv);
        int fd=::open(a.output.c_str(),O_WRONLY|O_CREAT|O_EXCL,0666);
        if(fd<0) throw std::system_error(errno,std::generic_category(),"exclusive CSV create");
        std::unique_ptr<FILE,FileCloser> csv(::fdopen(fd,"w"));
        if(!csv) { ::close(fd); throw std::runtime_error("fdopen failed"); }
        if(std::fprintf(csv.get(),"phase,index,status,wall_ms,h2d_ms,compute_ms,d2h_ms\n")<0) throw std::runtime_error("CSV write failed");
        const auto image=read_ppm(a.image);
        std::optional<PreparedImage> prepared;
        std::unique_ptr<OrtRunner> ort;
        std::unique_ptr<DepthSdk> sdk;
#ifdef MINIINFER_BENCHMARK_TRT
        std::unique_ptr<TrtRunner> trt;
#endif
        miniinfer::net::UniqueFd socket;
        if(a.mode=="raw") {
            if(a.backend=="ort") {
                ort=std::make_unique<OrtRunner>(a.model,4);
                prepared.emplace(preprocess(image,ort->input_size()));
            }
            else {
#ifdef MINIINFER_BENCHMARK_TRT
                trt=std::make_unique<TrtRunner>(a.model);
                prepared.emplace(preprocess(image,trt->input_size()));
#else
                throw std::invalid_argument("TRT support not compiled");
#endif
            }
        } else if(a.mode=="sdk") {
            SdkConfig config; config.package_dir=a.package;
            sdk=std::make_unique<DepthSdk>(config);
            if(sdk->snapshot().backend!=a.backend) throw std::invalid_argument("package backend differs from --backend");
        }
        const double startup_ms=ms(main_start);
        double connect_ms=0;
        if(a.mode=="tcp") {
            const auto start=Clock::now();
            socket=miniinfer::net::connect_tcp_until(start+std::chrono::milliseconds(a.timeout_ms),a.host,a.port);
            connect_ms=ms(start);
        }
        std::cout<<std::setprecision(17)<<"{\"event\":\"ready\",\"startup_ms\":"<<startup_ms
                 <<",\"connect_ms\":"<<connect_ms<<",\"main_to_ready_ms\":"<<ms(main_start)<<"}"<<std::endl;
        Counts total=counts(), steady=counts();
        std::uint32_t request_id=0;
        double first_ms=0, elapsed_steady_ms=0;
        auto run_phase=[&](const std::string& phase,std::size_t n) {
            std::cout<<"{\"event\":\"phase\",\"phase\":\""<<phase<<"\",\"main_ms\":"<<ms(main_start)<<"}"<<std::endl;
            const auto phase_start=Clock::now();
            for(std::size_t i=0;i<n;++i) {
                ++request_id;
                std::string status="succeeded";
                std::optional<double> h2d,compute,d2h;
                std::optional<RawDepth> raw_result;
                std::optional<DepthMap> depth_result;
                miniinfer::depth::net::Response response;
                const auto start=Clock::now();
                try {
                    if(a.mode=="raw") {
                        if(ort) raw_result.emplace(ort->run_raw(*prepared));
#ifdef MINIINFER_BENCHMARK_TRT
                        else { TrtTiming timing; raw_result.emplace(trt->run_raw(*prepared,&timing));
                            h2d=timing.h2d_ms;compute=timing.compute_ms;d2h=timing.d2h_ms; }
#endif
                    } else if(sdk) depth_result.emplace(sdk->run(image));
                    else if(!socket) status="closed";
                    else {
                        const auto deadline=start+std::chrono::milliseconds(a.timeout_ms);
                        auto sent=miniinfer::depth::net::send_request(socket.get(),request_id,image,deadline);
                        if(sent.status!=miniinfer::net::IoStatus::Ok) { status=io_status(sent.status);socket.reset(); }
                        else {
                            auto received=miniinfer::depth::net::recv_response(socket.get(),response,request_id,image.width(),image.height(),deadline);
                            if(received.status!=miniinfer::net::IoStatus::Ok) { status=io_status(received.status);socket.reset(); }
                            else {
                                using S=miniinfer::net::Status;
                                if(response.status==S::Busy) { status="busy";socket.reset(); }
                                else if(response.status==S::ShuttingDown) status="closed";
                                else if(response.status!=S::Ok || !response.depth) status="failure";
                            }
                        }
                    }
                } catch(const SdkError& e) {
                    if(e.code()==SdkErrorCode::Busy) status="busy";
                    else if(e.code()==SdkErrorCode::Closed) status="closed";
                    else status="failure";
                } catch(const std::exception& e) {
                    status="failure"; if(a.mode=="tcp") socket.reset();
                    std::cerr<<"request "<<request_id<<": "<<e.what()<<'\n';
                }
                const auto wall=ms(start);
                ++total.at(status); if(phase=="steady") ++steady.at(status);
                if(phase=="first") first_ms=wall;
                if(std::fprintf(csv.get(),"%s,%zu,%s,%.17g,",phase.c_str(),i,status.c_str(),wall)<0) throw std::runtime_error("CSV write failed");
                for(const auto& value:{h2d,compute,d2h}) {
                    if(value && (!std::isfinite(*value) || *value<0)) throw std::runtime_error("invalid device timer");
                }
                if(h2d) std::fprintf(csv.get(),"%.17g",*h2d);
                std::fputc(',',csv.get());
                if(compute) std::fprintf(csv.get(),"%.17g",*compute);
                std::fputc(',',csv.get());
                if(d2h) std::fprintf(csv.get(),"%.17g",*d2h);
                std::fputc('\n',csv.get());
            }
            if(phase=="steady") elapsed_steady_ms=ms(phase_start);
        };
        run_phase("first",1);
        run_phase("warmup",a.warmup);
        run_phase("steady",a.iterations);
        if(std::fflush(csv.get())!=0 || std::ferror(csv.get())) throw std::runtime_error("CSV flush failed");
        std::cout<<"{\"event\":\"summary\",\"startup_ms\":"<<startup_ms<<",\"connect_ms\":"<<connect_ms
                 <<",\"first_ms\":"<<first_ms<<",\"warmup\":"<<a.warmup<<",\"iterations\":"<<a.iterations
                 <<",\"counts\":"<<count_json(total)<<",\"steady_counts\":"<<count_json(steady)
                 <<",\"elapsed_steady_ms\":"<<elapsed_steady_ms;
        if(sdk) { auto s=sdk->snapshot(); std::cout<<",\"sdk_snapshot\":{\"accepted\":"<<s.accepted<<",\"succeeded\":"<<s.succeeded
            <<",\"failed\":"<<s.failed<<",\"reserved_bytes\":"<<s.reserved_bytes<<",\"reservation_bytes_per_request\":"
            <<DepthSdk::reservation_bytes(image.width(),image.height())<<"}"; }
        std::cout<<"}"<<std::endl;
        return 0;
    } catch(const std::exception& e) { std::cerr<<"depth_benchmark: "<<e.what()<<'\n';return 1; }
}
