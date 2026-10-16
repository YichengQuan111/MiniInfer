#include <miniinfer/depth/sdk.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: depth_sdk_example PACKAGE_DIR IMAGE.ppm OUTPUT.f32 [PREPROCESS [GPU_SLOTS]]\n";
        return 2;
    }
    try {
        miniinfer::depth::SdkConfig config;
        config.package_dir = argv[1];
        if (argc >= 5) config.preprocess_mode = miniinfer::depth::parse_preprocess_mode(argv[4]);
        if (argc == 6) {
            const std::string slots = argv[5];
            if (slots != "1" && slots != "2") throw std::invalid_argument("GPU_SLOTS must be 1 or 2");
            config.gpu_slots = slots == "1" ? 1 : 2;
        }
        miniinfer::depth::DepthSdk sdk(config);
        const auto input = miniinfer::depth::read_ppm(argv[2]);
        const auto result = sdk.run(input);
        sdk.shutdown();
        // Output owns its storage after SDK shutdown.
        miniinfer::depth::write_f32(argv[3], result.values());
        const auto status = sdk.snapshot();
        std::cout << result.width() << 'x' << result.height() << " relative depth, "
                  << status.backend << '/' << status.precision << ", "
                  << miniinfer::depth::preprocess_mode_name(config.preprocess_mode)
                  << ", slots=" << config.gpu_slots << ", delivered=" << status.delivered << '\n';
        return 0;
    } catch (const miniinfer::depth::SdkError& error) {
        std::cerr << "SDK error code=" << static_cast<int>(error.code()) << ": " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
