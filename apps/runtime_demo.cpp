#include "miniinfer/data_io.hpp"
#include "miniinfer/runtime.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// A small executable walkthrough of ownership, error delivery and drain.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: runtime_demo REFERENCE_DIR\n";
        return 2;
    }
    try {
        const std::filesystem::path directory(argv[1]);
        const auto inputs = miniinfer::load_f32_file(directory / "inputs.f32", 64 * 64);
        const auto reference = miniinfer::load_f32_file(directory / "logits.f32", 64 * 10);
        auto make_input = [&](std::size_t batch) {
            return miniinfer::MlpInput(batch, std::vector<float>(inputs.data(), inputs.data() + batch * 64));
        };
        std::vector<miniinfer::MlpResult> retained;
        retained.reserve(4);
        bool failure_delivered = false, closed_rejected = false;
        {
            miniinfer::Runtime runtime(miniinfer::MlpModel::load(directory), {2, 8, 32});
            std::vector<std::future<miniinfer::MlpResult>> futures;
            futures.reserve(3);
            for (const std::size_t batch : {1U, 8U, 32U}) {
                auto owned_input = make_input(batch);
                futures.push_back(runtime.submit(owned_input));
                owned_input = make_input(1);  // The accepted task owns an independent copy.
            }
            auto failed = runtime.submit(miniinfer::MlpInput(1,
                std::vector<float>(64, std::numeric_limits<float>::max())));
            for (auto& future : futures) retained.push_back(future.get());
            try { (void)failed.get(); }
            catch (const std::overflow_error&) { failure_delivered = true; }
            retained.push_back(runtime.run(make_input(1)));  // Worker remains usable.
            runtime.shutdown();
            runtime.shutdown();
            try { (void)runtime.submit(make_input(1)); }
            catch (const miniinfer::PoolClosed&) { closed_rejected = true; }
        }
        // Access and validate all outputs after Runtime/model/workspace destruction.
        if (!failure_delivered || !closed_rejected) throw std::runtime_error("error contract failed");
        for (const auto& result : retained) {
            const auto logits = result.logits();
            for (std::size_t row = 0; row < result.batch(); ++row) {
                std::size_t best = 0;
                for (std::size_t column = 0; column < 10; ++column) {
                    const float expected = reference.data()[row * 10 + column];
                    const float actual = logits.at(row, column);
                    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-4F + 1e-4F * std::abs(expected)) {
                        throw std::runtime_error("demo logits differ from FP32 reference");
                    }
                    if (expected > reference.data()[row * 10 + best]) best = column;
                }
                if (result.labels()[row] != static_cast<std::int32_t>(best)) {
                    throw std::runtime_error("demo class differs from FP32 reference");
                }
            }
        }
        std::cout << "{\"workers\":2,\"accepted\":5,\"successes\":4,\"task_failures\":1,"
                     "\"closed_rejected\":true,\"retained_after_destruction\":true,\"batches\":[1,8,32,1],\"labels\":[";
        for (std::size_t index = 0; index < retained.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << '[';
            const auto& labels = retained[index].labels();
            for (std::size_t row = 0; row < labels.size(); ++row) {
                if (row) std::cout << ',';
                std::cout << labels[row];
            }
            std::cout << ']';
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
