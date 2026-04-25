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

    // (channels, height, width)
    std::vector<int64_t> shape = {
        1, // Batches
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