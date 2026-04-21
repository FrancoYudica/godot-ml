#include "reshape_operator.hpp"

namespace ml {

bool ReshapeOperator::init(godot::RenderingDevice* rd) {
    return true;
}

// void ReshapeOperator::dispatch(
//     const PhysicalNode& node,
//     const OperatorContext& ctx) {

//     ERR_FAIL_COND_MSG(
//         node.reshape_info.get() == nullptr,
//         "Reshape info is null");

//     auto reshape_result = node.reshape_info->validate();

//     ERR_FAIL_COND_MSG(
//         !reshape_result.success,
//         reshape_result.error.c_str());

//     const std::string& tensor_name = node.inputs[0];
//     OperationResult result = {false, "Unable to find tensor name: " + tensor_name};

//     if (ctx.weights_tm->has(tensor_name)) {
//         result = ctx.weights_tm->reshape(tensor_name, node.reshape_info->shape);
//     }

//     else if (ctx.activations_tm->has(tensor_name)) {
//         result = ctx.activations_tm->reshape(tensor_name, node.reshape_info->shape);
//     }

//     ERR_FAIL_COND_MSG(
//         !result.success,
//         result.error.c_str());
// }

void ReshapeOperator::dispatch(
    const PhysicalNode& node,
    const OperatorContext& ctx) {

    auto& attrs = std::get<ReshapeAttributes>(node.attributes);

    switch (attrs.mode) {

    case ReshapeMode::ImageToGemm: {
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "ImageToGemm: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "ImageToGemm: expected 1 output");

        auto& in_shape = ctx.activations_tm->get_tensor_shape(node.inputs[0]);
        ERR_FAIL_COND_MSG(in_shape.size() != 4, "ImageToGemm expects 4D input");

        int64_t b = in_shape[0], c = in_shape[1],
                h = in_shape[2], w = in_shape[3];

        // Zero-copy — alias reinterprets the same GPU buffer as [b*h*w, c]
        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            {b * h * w, c});

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());

        // Register alias for cleanup at end of frame
        ctx.frame_deletion_stack->push([name = node.outputs[0], tm = ctx.activations_tm]() {
            tm->remove_alias(name);
        });
        break;
    }

    case ReshapeMode::GemmToImage: {
        ERR_FAIL_COND_MSG(!node.reshape_info || node.reshape_info->shape.empty(), "GemmToImage: missing reshape_info");
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "GemmToImage: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "GemmToImage: expected 1 output");

        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            node.reshape_info->shape);

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());
        break;
    }
    }
}

void ReshapeOperator::destroy(godot::RenderingDevice* rd) {}

} // namespace ml