#include "float_array_input_handler.hpp"

std::vector<int64_t> ml::FloatArrayInputHandler::upload(
    const std::unique_ptr<InputDesc::BaseData>& desc,
    const InputHandlerContext& ctx) {
    // Use dynamic_cast on the raw pointer
    InputDesc::FloatArray* float_array_desc =
        dynamic_cast<InputDesc::FloatArray*>(desc.get());

    if (!float_array_desc) {
        ERR_PRINT("InferenceEngine: Failed to cast InputDesc to FloatArray.");
        return {};
    }

    ctx.activations_tm->get_or_create(float_array_desc->tensor_name,
                                      float_array_desc->shape,
                                      float_array_desc->data.to_byte_array());

    return float_array_desc->shape;
}