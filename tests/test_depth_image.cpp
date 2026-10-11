#include "miniinfer/depth/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using namespace miniinfer::depth;
void expect(bool value, const char* description) {
    if (!value) throw std::runtime_error(description);
}
template<class Callable> void invalid(Callable callable, const char* description) {
    try { callable(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(description);
}
template<class Callable> void fails(Callable callable, const char* description) {
    try { callable(); } catch (const std::exception&) { return; }
    throw std::runtime_error(description);
}
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        std::string pattern = (std::filesystem::temp_directory_path() / "miniinfer-depth-image-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* created = ::mkdtemp(buffer.data());
        if (!created) throw std::runtime_error("cannot create test directory");
        path = created;
    }
    ~Temporary() { std::error_code error; std::filesystem::remove_all(path, error); }
};
void dimensions_and_geometry() {
    expect(validate_dimensions(2048, 1024) == max_pixels, "maximum pixel boundary");
    expect(validate_dimensions(4096, 1) == 4096, "maximum side boundary");
    for (const auto& shape : std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {1, 0}, {4097, 1}, {2049, 1024}, {std::numeric_limits<std::size_t>::max(), 2}})
        invalid([&] { validate_dimensions(shape.first, shape.second); }, "invalid dimensions accepted");
    const auto landscape = geometry(4, 3);
    expect(landscape.resized_height == 389 && landscape.pad_top == 64 && landscape.pad_bottom == 65, "round half up and odd vertical padding");
    const auto portrait = geometry(3, 4);
    expect(portrait.resized_width == 389 && portrait.pad_left == 64 && portrait.pad_right == 65, "odd horizontal padding");
    expect(geometry(1, 4096).resized_width == 1, "minimum one resized pixel");
    auto wrong = landscape;
    ++wrong.pad_left;
    invalid([&] { validate_geometry(wrong); }, "malformed geometry accepted");
    invalid([] { RgbImage(2, 1, std::vector<std::uint8_t>(5)); }, "short RGB accepted");
    invalid([] { RgbImage(2, 1, std::vector<std::uint8_t>(7)); }, "long RGB accepted");
    invalid([] { RawDepth(std::vector<float>(raw_elements - 1)); }, "short raw accepted");
    invalid([] { PreparedImage(std::vector<float>(input_elements - 1), geometry(2, 2)); }, "short prepared accepted");
    invalid([] { DepthMap(2, 2, std::vector<float>(5)); }, "long depth accepted");
}
void interpolation_and_extremes() {
    const std::vector<float> source{0, 10, 20, 30};
    const std::vector<float> expected{0, 2.5F, 7.5F, 10, 5, 7.5F, 12.5F, 15, 15, 17.5F, 22.5F, 25, 20, 22.5F, 27.5F, 30};
    expect(bilinear_half_pixel(source, 2, 2, 1, 4, 4) == expected, "half pixel clamp hand computation");
    expect(bilinear_half_pixel(source, 2, 2, 1, 1, 1) == std::vector<float>{15}, "single target pixel");
    expect(bilinear_half_pixel({0, 4}, 2, 1, 1, 4, 3) == std::vector<float>{0, 1, 3, 4, 0, 1, 3, 4, 0, 1, 3, 4}, "singleton source axis");
    invalid([&] { bilinear_half_pixel(source, 2, 2, 3, 4, 4); }, "interpolation length mismatch accepted");
    const RgbImage rgb(4, 1, {0, 255, 128, 0, 255, 128, 0, 255, 128, 0, 255, 128});
    const auto prepared = preprocess(rgb);
    const std::size_t pixel = prepared.transform().pad_top * model_size;
    expect(prepared.values()[pixel] == (0.0F - 0.485F) / 0.229F, "black pixel normalization");
    expect(prepared.values()[raw_elements + pixel] == (1.0F - 0.456F) / 0.224F, "white channel normalization");
    expect(prepared.values()[2 * raw_elements + pixel] == (128.0F / 255.0F - 0.406F) / 0.225F, "RGB channel order");
    expect(prepared.values()[raw_elements] == -0.456F / 0.224F, "padding normalized after fill");
    std::vector<std::uint8_t> pixels(1036 * 518 * 3);
    for (std::size_t y = 0; y < 518; ++y)
        for (std::size_t x = 1; x < 1036; x += 2)
            for (std::size_t c = 0; c < 3; ++c) pixels[(y * 1036 + x) * 3 + c] = 1;
    const auto fractional = preprocess(RgbImage(1036, 518, std::move(pixels)));
    expect(fractional.values()[fractional.transform().pad_top * model_size] == (0.5F / 255.0F - 0.485F) / 0.229F, "float32 before resize");
}
void restoration_and_ownership() {
    const auto transform = geometry(4, 2);
    std::vector<float> values(raw_elements, -1000.0F);
    for (std::size_t y = 0; y < 259; ++y)
        for (std::size_t x = 0; x < 518; ++x)
            values[(y + transform.pad_top) * 518 + x] = static_cast<float>(y) * 10.0F + static_cast<float>(x);
    const RawDepth raw(values);
    const auto restored = postprocess(raw, transform);
    const std::vector<float> expected{706.75F, 836.25F, 965.75F, 1095.25F, 2001.75F, 2131.25F, 2260.75F, 2390.25F};
    expect(restored.values() == expected, "crop before restore hand computation");
    values.assign(raw_elements, 0.0F);
    expect(restored.values() == expected && raw.values()[transform.pad_top * 518 + 1] == 1.0F, "owned output and raw storage");
    const auto copied = restored;
    expect(copied.values().data() != restored.values().data(), "depth copy independent");
    const auto constant = postprocess(RawDepth(std::vector<float>(raw_elements, 42.25F)), geometry(3, 2));
    expect(constant.values() == std::vector<float>(6, 42.25F), "no per-image minmax");
}
void nonfinite() {
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        invalid([&] { DepthMap(1, 1, {value}); }, "nonfinite depth accepted");
        invalid([&] { std::vector<float> raw(raw_elements); raw[0] = value; RawDepth result(std::move(raw)); }, "nonfinite raw padding accepted");
        invalid([&] { std::vector<float> data(input_elements); data[0] = value; PreparedImage result(std::move(data), geometry(1, 1)); }, "nonfinite input accepted");
        invalid([&] { bilinear_half_pixel({value}, 1, 1, 1, 2, 2); }, "nonfinite bilinear accepted");
    }
}
void artifacts() {
    Temporary temporary;
    const auto ppm = temporary.path / "image.ppm";
    const RgbImage image(2, 1, {10, 13, 32, 35, 0, 255});
    write_ppm(ppm, image);
    const auto decoded = read_ppm(ppm);
    expect(decoded.pixels() == image.pixels(), "PPM roundtrip raster whitespace");
    fails([&] { write_ppm(ppm, image); }, "PPM overwrite accepted");
    const auto crlf = temporary.path / "crlf.ppm";
    write_text_exclusive(crlf, std::string("P6\r\n# comment\r\n2 1\r\n255\r\n") + std::string(reinterpret_cast<const char*>(image.pixels().data()), 6));
    const auto decoded_crlf = read_ppm(crlf);
    expect(decoded_crlf.pixels() == image.pixels(), "PPM comments CRLF");
    unsigned index = 0;
    for (const auto& text : {std::string("P6\n2 1\n255\n12345"), std::string("P6\n2 1\n255\n1234567"),
                             std::string("P3\n2 1\n255\n123456"), std::string("P6\n2 1\n256\n123456"),
                             std::string("P6\n4097 1\n255\n"), std::string("P6\n")}) {
        const auto path = temporary.path / ("invalid" + std::to_string(index++) + ".ppm");
        write_text_exclusive(path, text);
        fails([&] { read_ppm(path); }, "invalid PPM accepted");
    }
    const auto floats = temporary.path / "values.f32";
    write_f32(floats, {1.0F, -2.0F});
    std::ifstream stream(floats, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    expect(bytes == std::string("\0\0\x80\x3f\0\0\0\xc0", 8), "little-endian float32 encoding");
    expect(read_f32(floats, 2) == std::vector<float>{1, -2}, "float32 roundtrip");
    invalid([&] { read_f32(floats, 1); }, "long float32 file accepted");
    invalid([&] { read_f32(floats, 3); }, "short float32 file accepted");
    fails([&] { write_f32(floats, {3, 4}); }, "float32 overwrite accepted");
    expect(read_f32(floats, 2) == std::vector<float>{1, -2}, "overwrite altered existing file");
    const auto nan = temporary.path / "nan.f32";
    write_text_exclusive(nan, std::string("\0\0\xc0\x7f", 4));
    invalid([&] { read_f32(nan, 1); }, "nonfinite artifact accepted");
    fails([&] { read_ppm(temporary.path / "missing.ppm"); }, "missing image accepted");
}
}  // namespace

int main() {
    try {
        dimensions_and_geometry();
        interpolation_and_extremes();
        restoration_and_ownership();
        nonfinite();
        artifacts();
        std::cout << "Depth image contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Depth image contract failed: " << error.what() << '\n';
        return 1;
    }
}
