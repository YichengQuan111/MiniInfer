#include "miniinfer/depth/artifact.hpp"
#include "miniinfer/depth/image.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace miniinfer::depth;
void expect(bool value) { if(!value) throw std::runtime_error("artifact contract assertion failed"); }
template<class F> void rejects(F operation) { bool thrown=false;try{operation();}catch(const std::exception&){thrown=true;}expect(thrown); }
int main() {
    const auto dir=std::filesystem::temp_directory_path()/("miniinfer-artifact-"+std::to_string(getpid()));
    try {
        expect(sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        expect(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        expect(sha256(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        std::filesystem::create_directory(dir);
        const Manifest good{{"a","value"},{"b","value=with=equals"}};
        write_text_exclusive(dir/"good",manifest_text(good));
        expect(read_manifest(dir/"good")==good);
        expect(file_sha256(dir/"good")==sha256(manifest_text(good)));
        write_text_exclusive(dir/"duplicate","a=x\na=y\n");
        write_text_exclusive(dir/"missing","missing\n");
        rejects([&]{read_manifest(dir/"duplicate");});
        rejects([&]{read_manifest(dir/"missing");});
        rejects([&]{read_artifact(dir/"good",1);});
        rejects([&]{require_equal(good,"a","mismatch");});
        rejects([&]{require_field(good,"absent");});
        rejects([&]{manifest_text({{"bad key","value"}});});
        rejects([&]{manifest_text({{"valid","value\ninjection"}});});
        std::filesystem::remove_all(dir);
        std::cout<<"artifact hash and manifest contract passed\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';std::filesystem::remove_all(dir);return 1;}
}
