#include "texture_input_handler.hpp"

#include <godot_cpp/classes/image.hpp>

namespace ml {

bool TextureInputHandler::upload(
    const std::unique_ptr<InputDesc::BaseData>& desc,
    const InputHandlerContext& ctx) {
    InputDesc::Texture* texture_desc =
        dynamic_cast<InputDesc::Texture*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !texture_desc,
        false,
        "InferenceEngine: Failed to cast InputDesc to Texture.");

    ERR_FAIL_COND_V_MSG(
        texture_desc->texture.is_null(),
        false,
        "Texture2DInputHandler: null texture.");

    Ref<Image> img = texture_desc->texture->get_image();
    ERR_FAIL_COND_V_MSG(
        img.is_null(),
        {},
        "Texture2DInputHandler: null image.");

    if (img->is_compressed()) {
        img->decompress();
    }

    // mode values: 0=RGB, 1=RGBA, 2=RED, 3=GREEN, 4=BLUE, 5=ALPHA, 6=GRAYSCALE
    uint32_t mode = texture_desc->load_mode;
    bool need_alpha = (mode == 1 || mode == 5); // RGBA or ALPHA
    img->convert(need_alpha ? Image::FORMAT_RGBAF : Image::FORMAT_RGBF);

    uint32_t w = img->get_width();
    uint32_t h = img->get_height();
    uint32_t channels = texture_desc->channels;
    uint32_t pixels = w * h;
    uint32_t src_ch = need_alpha ? 4u : 3u;

    PackedByteArray raw = img->get_data();
    const float* src = reinterpret_cast<const float*>(raw.ptr());

    // BCHW layout: each channel plane is contiguous.
    std::vector<float> floats(channels * pixels);
    for (uint32_t i = 0; i < pixels; i++) {
        const float* p = src + i * src_ch;
        switch (mode) {
        case 0: // RGB
            floats[0 * pixels + i] = p[0];
            floats[1 * pixels + i] = p[1];
            floats[2 * pixels + i] = p[2];
            break;
        case 1: // RGBA
            floats[0 * pixels + i] = p[0];
            floats[1 * pixels + i] = p[1];
            floats[2 * pixels + i] = p[2];
            floats[3 * pixels + i] = p[3];
            break;
        case 2: floats[i] = p[0]; break; // RED
        case 3: floats[i] = p[1]; break; // GREEN
        case 4: floats[i] = p[2]; break; // BLUE
        case 5: floats[i] = p[3]; break; // ALPHA
        default: // GRAYSCALE (mode 6)
            floats[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
            break;
        }
    }

    std::vector<int64_t> shape = {
        1,
        (int64_t)channels,
        (int64_t)h,
        (int64_t)w};

    ctx.activations_tm->get_or_create(
        texture_desc->tensor_name,
        shape,
        floats);

    return true;
}

std::vector<int64_t> TextureInputHandler::get_shape(const std::unique_ptr<InputDesc::BaseData>& desc) const {
    InputDesc::Texture* texture_desc =
        dynamic_cast<InputDesc::Texture*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !texture_desc,
        {},
        "InferenceEngine: Failed to cast InputDesc to Texture.");

    uint32_t w = texture_desc->texture->get_width();
    uint32_t h = texture_desc->texture->get_height();
    uint32_t channels = texture_desc->channels;

    // (channels, height, width)
    return {
        1, // Batches
        (int64_t)channels,
        (int64_t)h,
        (int64_t)w};
}

void TextureInputHandler::dispatch(const InputHandlerContext& ctx) {
}

} // namespace ml