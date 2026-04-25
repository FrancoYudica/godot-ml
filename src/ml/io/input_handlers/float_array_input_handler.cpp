#include "float_array_input_handler.hpp"

namespace ml {

bool FloatArrayInputHandler::upload(
    const std::unique_ptr<InputDesc::BaseData>& desc,
    const InputHandlerContext& ctx) {
    // Use dynamic_cast on the raw pointer
    InputDesc::FloatArray* float_array_desc =
        dynamic_cast<InputDesc::FloatArray*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !float_array_desc,
        false,
        "InferenceEngine: Failed to cast InputDesc to FloatArray.");

    ctx.activations_tm->get_or_create(
        float_array_desc->tensor_name,
        float_array_desc->shape,
        float_array_desc->data.to_byte_array());

    return true;
}

std::vector<int64_t> FloatArrayInputHandler::get_shape(
    const std::unique_ptr<InputDesc::BaseData>& desc) const {
    InputDesc::FloatArray* float_array_desc =
        dynamic_cast<InputDesc::FloatArray*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !float_array_desc,
        {},
        "InferenceEngine: Failed to cast InputDesc to FloatArray.");

    return float_array_desc->shape;
}

} // namespace ml