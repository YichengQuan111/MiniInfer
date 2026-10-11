#include "miniinfer/depth/net.hpp"
#include "miniinfer/net/socket.hpp"
#include <charconv>
#include <iostream>
#include <set>

int main(int argc,char** argv) {
    namespace wire=miniinfer::depth::net;
    namespace io=miniinfer::net;
    using namespace miniinfer::depth;
    if(argc==2&&std::string(argv[1])=="--help") {
        std::cout<<"depth_client --image INPUT.ppm --output DEPTH.f32 [--host 127.0.0.1] [--port 9001] [--timeout-ms 5000]\n";
        return 0;
    }
    try {
        std::filesystem::path image_path,output;std::string host="127.0.0.1";
        unsigned port=9001,timeout=5000;std::set<std::string> seen;
        for(int i=1;i<argc;i+=2) {
            const std::string key=argv[i];
            if(i+1==argc||!seen.insert(key).second) throw std::invalid_argument("missing/duplicate option");
            const std::string value=argv[i+1];
            if(key=="--image") image_path=value;
            else if(key=="--output") output=value;
            else if(key=="--host") host=value;
            else if(key=="--port"||key=="--timeout-ms") {
                unsigned number=0;const auto parsed=std::from_chars(value.data(),value.data()+value.size(),number);
                if(parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size()||number==0||number>(key=="--port"?65535U:600000U))
                    throw std::invalid_argument("invalid number");
                if(key=="--port") port=number;else timeout=number;
            }else throw std::invalid_argument("unknown option: "+key);
        }
        if(image_path.empty()||output.empty()) throw std::invalid_argument("--image and --output required");
        if(std::filesystem::exists(output)) throw std::invalid_argument("output already exists");
        const auto image=read_ppm(image_path);
        const auto deadline=io::Clock::now()+std::chrono::milliseconds(timeout);
        auto socket=io::connect_tcp_until(deadline,host,static_cast<std::uint16_t>(port));
        const auto sent=wire::send_request(socket.get(),1,image,deadline);
        wire::Response response;
        // Busy may arrive before the server consumes RGB. Read an available
        // error response even if sending the remainder encountered shutdown.
        const auto received=wire::recv_response(socket.get(),response,1,image.width(),image.height(),deadline);
        if(received.status!=io::IoStatus::Ok) throw std::runtime_error("depth response incomplete, timed out, or malformed");
        if(response.status!=wire::Status::Ok) throw std::runtime_error("server status="+std::to_string(static_cast<unsigned>(response.status)));
        if(sent.status!=io::IoStatus::Ok||!response.depth) throw std::runtime_error("request transmission failed");
        write_f32(output,response.depth->values());
        std::cout<<response.depth->width()<<'x'<<response.depth->height()<<" relative depth saved\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"depth_client: "<<error.what()<<'\n';return 1;}
}
