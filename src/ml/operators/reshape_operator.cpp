#include "reshape_operator.hpp"

namespace ml {

bool ReshapeOperator::init(godot::RenderingDevice* rd) {
    return true;
}

void ReshapeOperator::dispatch(
    const PhysicalNode& node,
    const OperatorContext& ctx) {

    ERR_FAIL_COND_MSG(
        node.reshape_info.get() == nullptr,
        "Reshape info is null");

    auto reshape_result = node.reshape_info->validate();

    ERR_FAIL_COND_MSG(
        !reshape_result.success,
        reshape_result.error.c_str());

    const std::string& tensor_name = node.inputs[0];
    OperationResult result = {false, "Unable to find tensor name: " + tensor_name};

    if (ctx.weights_tm->has(tensor_name)) {
        result = ctx.weights_tm->reshape(tensor_name, node.reshape_info->shape);
    }

    else if (ctx.activations_tm->has(tensor_name)) {
        result = ctx.activations_tm->reshape(tensor_name, node.reshape_info->shape);
    }

    ERR_FAIL_COND_MSG(
        !result.success,
        result.error.c_str());
}

void ReshapeOperator::destroy(godot::RenderingDevice* rd) {}

} // namespace ml