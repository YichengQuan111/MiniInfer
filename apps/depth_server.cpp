#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/net.hpp"
#include "miniinfer/net/server.hpp"
#include <pthread.h>
#include <unistd.h>
#include <charconv>
#include <cerrno>
#include <csignal>
#include <iostream>
#include <set>
#include <system_error>

namespace {
std::size_t number(const std::string& text,std::size_t maximum,bool zero=false) {
    std::size_t value=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+text.size()||(!zero&&value==0)||value>maximum)
        throw std::invalid_argument("invalid numeric option: "+text);
    return value;
}
}
int main(int argc,char** argv) {
    using namespace miniinfer;
    namespace wire=depth::net;
    const auto started=std::chrono::steady_clock::now();
    if(argc==2&&std::string(argv[1])=="--help") {
        std::cout<<"depth_server --package DIR [--host IPv4] [--port 9001] [--connections 0]\n"
          " [--connection-threads 4] [--waiting-connections 8] [--queue-capacity 4]\n"
          " [--sdk-bytes 134217728] [--network-requests 4] [--network-bytes 134217728]\n"
          " [--read-timeout-ms 5000] [--write-timeout-ms 5000]\n"
          " [--sdk-workers 4] [--gpu-slots 1] [--preprocess cpu|cuda-partial|cuda-full-basic|cuda-full-fused]\n"
          "MINI v2 RGB8 -> relative FP32; local IPv4 by default. SIGINT/SIGTERM drains accepted work.\n";
        return 0;
    }
    try {
        depth::SdkConfig config;net::ServerConfig server_config;server_config.port=9001;
        std::size_t network_count=4,network_bytes=128ULL*1024*1024;
        std::set<std::string> seen;
        for(int i=1;i<argc;i+=2) {
            const std::string key=argv[i];
            if(i+1==argc||!seen.insert(key).second) throw std::invalid_argument("missing/duplicate option: "+key);
            const std::string value=argv[i+1];
            if(key=="--package") config.package_dir=value;
            else if(key=="--preprocess") config.preprocess_mode=depth::parse_preprocess_mode(value);
            else if(key=="--gpu-slots") config.gpu_slots=number(value,2);
            else if(key=="--sdk-workers") config.worker_count=number(value,256);
            else if(key=="--host") server_config.host=value;
            else if(key=="--port") server_config.port=static_cast<std::uint16_t>(number(value,65535,true));
            else if(key=="--connections") server_config.max_connections=number(value,1000000,true);
            else if(key=="--connection-threads") server_config.connection_threads=number(value,128);
            else if(key=="--waiting-connections") server_config.waiting_connections=number(value,1024);
            else if(key=="--queue-capacity") config.queue_capacity=number(value,1024);
            else if(key=="--sdk-bytes") config.byte_budget=number(value,1024ULL*1024*1024);
            else if(key=="--network-requests") network_count=number(value,1024);
            else if(key=="--network-bytes") network_bytes=number(value,1024ULL*1024*1024);
            else if(key=="--read-timeout-ms") server_config.session.read_timeout=std::chrono::milliseconds(number(value,600000));
            else if(key=="--write-timeout-ms") server_config.session.write_timeout=std::chrono::milliseconds(number(value,600000));
            else throw std::invalid_argument("unknown option: "+key);
        }
        if(config.package_dir.empty()) throw std::invalid_argument("--package is required");
        sigset_t signals;sigemptyset(&signals);sigaddset(&signals,SIGINT);sigaddset(&signals,SIGTERM);
        const int error=pthread_sigmask(SIG_BLOCK,&signals,nullptr);
        if(error) throw std::system_error(error,std::generic_category(),"block stop signals");
        depth::DepthSdk sdk(config);
        wire::Budget budget(network_count,network_bytes);
        wire::RunDepth run=[&](const depth::RgbImage& image) {
            try{return sdk.run(image);}
            catch(const depth::SdkError& failure) {
                const auto code=failure.code();
                const auto status=code==depth::SdkErrorCode::Busy?wire::Status::Busy:
                    code==depth::SdkErrorCode::Closed?wire::Status::ShuttingDown:
                    wire::Status::InferenceFailed;
                throw wire::DepthNetError(status,failure.what());
            }
        };
        net::TcpServer server(server_config,net::RunSession{[&](int fd,const net::SessionConfig& session,const std::atomic<bool>* stop) {
            return wire::serve_connection(fd,run,budget,session,stop);
        }},[&]{sdk.shutdown();});
        const auto ready=sdk.snapshot();
        std::cout<<"{\"event\":\"ready\",\"pid\":"<<getpid()<<",\"port\":"<<server.port()
            <<",\"startup_ms\":"<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()
            <<",\"backend\":\""<<ready.backend<<"\",\"precision\":\""<<ready.precision<<"\""
            <<",\"preprocess_mode\":\""<<depth::preprocess_mode_name(config.preprocess_mode)<<"\""
            <<",\"gpu_slots\":"<<config.gpu_slots<<",\"sdk_workers\":"<<config.worker_count<<"}"<<std::endl;
        while(!server.finished()) {
            const timespec wait{0,20000000};const int signal=sigtimedwait(&signals,nullptr,&wait);
            if(signal==SIGINT||signal==SIGTERM) break;
            if(signal<0&&errno!=EAGAIN&&errno!=EINTR) throw std::system_error(errno,std::generic_category(),"wait signal");
        }
        server.shutdown();
        const auto stats=server.snapshot();const auto bytes=budget.snapshot();const auto final=sdk.snapshot();
        std::cout<<"{\"event\":\"stopped\",\"requests\":"<<stats.requests<<",\"responses\":"<<stats.responses
            <<",\"accepted\":"<<stats.accepted<<",\"completed\":"<<stats.completed
            <<",\"active\":"<<stats.active<<",\"pending\":"<<stats.pending
            <<",\"rejected\":"<<stats.rejected<<",\"expired\":"<<stats.expired<<",\"discarded\":"<<stats.discarded
            <<",\"bad_requests\":"<<stats.bad_requests<<",\"protocol_errors\":"<<stats.protocol_errors
            <<",\"session_failures\":"<<stats.session_failures<<",\"accept_errors\":"<<stats.accept_errors
            <<",\"busy\":"<<stats.busy<<",\"inference_failures\":"<<stats.inference_failures
            <<",\"io_failures\":"<<stats.io_failures<<",\"network_bytes\":"<<bytes.bytes
            <<",\"network_peak_bytes\":"<<bytes.peak_bytes<<",\"network_inflight\":"<<bytes.inflight
            <<",\"sdk_bytes\":"<<final.reserved_bytes<<",\"sdk_inflight\":"<<final.inflight
            <<",\"sdk_queued\":"<<final.queued<<",\"sdk_executing\":"<<final.executing
            <<",\"sdk_accepted\":"<<final.accepted<<",\"sdk_succeeded\":"<<final.succeeded
            <<",\"sdk_failed\":"<<final.failed<<",\"sdk_delivered\":"<<final.delivered
            <<",\"sdk_rejected_busy\":"<<final.rejected_busy<<",\"sdk_rejected_closed\":"<<final.rejected_closed
            <<",\"sdk_rejected_invalid\":"<<final.rejected_invalid
            <<",\"sdk_accepting\":"<<(final.accepting?"true":"false")
            <<",\"sdk_shutdown_complete\":"<<(final.shutdown_complete?"true":"false")<<"}\n";
        return stats.accept_errors||stats.session_failures?1:0;
    }catch(const std::exception& error){std::cerr<<"depth_server: "<<error.what()<<'\n';return 1;}
}
