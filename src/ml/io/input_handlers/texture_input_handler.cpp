#include "texture_input_handler.hpp"

#include <godot_cpp/classes/image.hpp>

namespace ml {

std::vector<int64_t> TextureInputHandler::upload(
    const std::unique_ptr<InputDesc::BaseData>& desc,
    const InputHandlerContext& ctx) {
    InputDesc::Texture* texture_desc =
        dynamic_cast<InputDesc::Texture*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !texture_desc,
        {},
        "InferenceEngine: Failed to cast InputDesc to Texture.");

    ERR_FAIL_COND_V_MSG(
        texture_desc->texture.is_null(),
        {},
        "Texture2DInputHandler: null texture.");

    Ref<Image> img = texture_desc->texture->get_image();
    ERR_FAIL_COND_V_MSG(
        img.is_null(),
        {},
        "Texture2DInputHandler: null image.");

    if (img->is_compressed()) {
        img->decompress();
    }
    img->convert(Image::FORMAT_RGBF);

    uint32_t w = img->get_width();
    uint32_t h = img->get_height();
    uint32_t channels = texture_desc->channels;
    uint32_t pixels = w * h;

    PackedByteArray raw = img->get_data();
    const float* src = reinterpret_cast<const float*>(raw.ptr());

    std::vector<float> floats(pixels * channels);
    for (uint32_t i = 0; i < pixels; i++) {
        for (uint32_t c = 0; c < channels; c++) {
            floats[i * channels + c] = src[i * 3 + c];
        }
    }

    std::vector<int64_t> shape = {
        (int64_t)pixels,
        (int64_t)channels};

    ctx.activations_tm->get_or_create(
        texture_desc->tensor_name,
        shape,
        floats);

    return shape;
}

void TextureInputHandler::dispatch(const InputHandlerContext& ctx) {
}

} // namespace ml