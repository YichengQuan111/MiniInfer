#include "miniinfer/depth/artifact.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace miniinfer::depth {
namespace {
constexpr std::array<std::uint32_t,64> constants{
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
std::uint32_t rotate(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32-n)); }
class Hash {
    std::array<std::uint32_t,8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<std::uint8_t,64> buffer{};
    std::size_t used = 0;
    std::uint64_t length = 0;
    void block() {
        std::array<std::uint32_t,64> w{};
        for (std::size_t i=0;i<16;++i)
            w[i]=(std::uint32_t(buffer[i*4])<<24)|(std::uint32_t(buffer[i*4+1])<<16)
                |(std::uint32_t(buffer[i*4+2])<<8)|buffer[i*4+3];
        for (std::size_t i=16;i<64;++i) {
            const auto s0=rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3);
            const auto s1=rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        auto a=state[0],b=state[1],c=state[2],d=state[3];
        auto e=state[4],f=state[5],g=state[6],h=state[7];
        for (std::size_t i=0;i<64;++i) {
            const auto s1=rotate(e,6)^rotate(e,11)^rotate(e,25);
            const auto t1=h+s1+((e&f)^(~e&g))+constants[i]+w[i];
            const auto s0=rotate(a,2)^rotate(a,13)^rotate(a,22);
            const auto t2=s0+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
        state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }
public:
    void update(std::string_view bytes) {
        length+=bytes.size();
        for (unsigned char byte:bytes) {
            buffer[used++]=byte;
            if (used==64) { block();used=0; }
        }
    }
    std::string finish() {
        const std::uint64_t bits=length*8;
        buffer[used++]=0x80;
        if (used>56) { while(used<64) buffer[used++]=0;block();used=0; }
        while(used<56) buffer[used++]=0;
        for (int i=7;i>=0;--i) buffer[used++]=static_cast<std::uint8_t>(bits>>(i*8));
        block();
        std::ostringstream out;
        out<<std::hex<<std::setfill('0');
        for(auto value:state) out<<std::setw(8)<<value;
        return out.str();
    }
};
void validate_entry(const std::string& key,const std::string& value) {
    if(key.empty()||key.size()>80||key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos
       ||value.empty()||value.size()>4096||value.find_first_of("\r\n\0",0,3)!=std::string::npos)
        throw std::invalid_argument("invalid manifest entry: "+key);
}
}
std::string sha256(std::string_view bytes) { Hash hash;hash.update(bytes);return hash.finish(); }
std::string file_sha256(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("cannot hash artifact: "+path.string());
    Hash hash;
    std::array<char,65536> buffer{};
    while(file) { file.read(buffer.data(),buffer.size());hash.update({buffer.data(),static_cast<std::size_t>(file.gcount())}); }
    if(!file.eof()) throw std::runtime_error("artifact hash read failed");
    return hash.finish();
}
std::vector<std::uint8_t> read_artifact(const std::filesystem::path& path,std::size_t max_bytes) {
    const auto count=std::filesystem::file_size(path);
    if(count==0||count>max_bytes) throw std::invalid_argument("artifact size outside limit");
    std::ifstream file(path,std::ios::binary);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
    file.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(count));
    if(file.gcount()!=static_cast<std::streamsize>(count)||file.peek()!=std::char_traits<char>::eof()||file.bad())
        throw std::runtime_error("artifact incomplete or changed during read");
    return result;
}
Manifest read_manifest(const std::filesystem::path& path) {
    const auto bytes=read_artifact(path,64*1024);
    std::istringstream input(std::string(bytes.begin(),bytes.end()));
    Manifest result;std::string line;
    while(std::getline(input,line)) {
        const auto pos=line.find('=');
        if(pos==std::string::npos) throw std::invalid_argument("manifest line requires key=value");
        auto key=line.substr(0,pos),value=line.substr(pos+1);
        validate_entry(key,value);
        if(!result.emplace(key,value).second) throw std::invalid_argument("duplicate manifest field: "+key);
    }
    return result;
}
std::string manifest_text(const Manifest& manifest) {
    std::string text;
    for(const auto& item:manifest) { validate_entry(item.first,item.second);text+=item.first+"="+item.second+"\n"; }
    return text;
}
const std::string& require_field(const Manifest& manifest,const std::string& key) {
    auto it=manifest.find(key);
    if(it==manifest.end()) throw std::invalid_argument("missing manifest field: "+key);
    return it->second;
}
void require_equal(const Manifest& manifest,const std::string& key,const std::string& expected) {
    if(require_field(manifest,key)!=expected) throw std::invalid_argument("incompatible manifest field: "+key);
}
} // namespace miniinfer::depth
