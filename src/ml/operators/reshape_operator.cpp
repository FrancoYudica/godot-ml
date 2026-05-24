#include "reshape_operator.hpp"

namespace ml {

bool ReshapeOperator::init(godot::RenderingDevice* rd) {
    return true;
}

void ReshapeOperator::dispatch(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& attrs = std::get<Physical::ReshapeAttrs>(node.attributes);

    switch (attrs.mode) {

    case Physical::ReshapeMode::ImageToGemm: {
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "ImageToGemm: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "ImageToGemm: expected 1 output");

        const auto& in = ctx.shape_table->at(node.inputs[0]);
        ERR_FAIL_COND_MSG(in.size() != 4, "ImageToGemm: expects 4D input");

        int64_t b = in[0], c = in[1], h = in[2], w = in[3];

        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            {b * h * w, c});

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());

        ctx.frame_deletion_stack->push([name = node.outputs[0], tm = ctx.activations_tm]() {
            tm->remove_alias(name);
        });
        break;
    }

    case Physical::ReshapeMode::GemmToImage: {
        ERR_FAIL_COND_MSG(attrs.image_shape_ref.empty(), "GemmToImage: image_shape_ref is empty");
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "GemmToImage: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "GemmToImage: expected 1 output");

        const auto& target = ctx.shape_table->at(attrs.image_shape_ref);

        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            target);

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());
        break;
    }
    }
}

void ReshapeOperator::destroy(godot::RenderingDevice* rd) {}

} // namespace ml