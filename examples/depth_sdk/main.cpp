#include <miniinfer/depth/sdk.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: depth_sdk_example PACKAGE_DIR IMAGE.ppm OUTPUT.f32\n";
        return 2;
    }
    try {
        miniinfer::depth::SdkConfig config;
        config.package_dir = argv[1];
        miniinfer::depth::DepthSdk sdk(config);
        const auto input = miniinfer::depth::read_ppm(argv[2]);
        const auto result = sdk.run(input);
        sdk.shutdown();
        // Output owns its storage after SDK shutdown.
        miniinfer::depth::write_f32(argv[3], result.values());
        const auto status = sdk.snapshot();
        std::cout << result.width() << 'x' << result.height() << " relative depth, "
                  << status.backend << '/' << status.precision << ", delivered=" << status.delivered << '\n';
        return 0;
    } catch (const miniinfer::depth::SdkError& error) {
        std::cerr << "SDK error code=" << static_cast<int>(error.code()) << ": " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
