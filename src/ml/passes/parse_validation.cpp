#include "parse_validation.hpp"

namespace ml::passes {

static OperationResult check_kernel_pads_strides_dilations(
    const std::string& ctx,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations) {
    if (kernel_shape.size() != 2)
        return {
            false,
            ctx + "only 2D convolutions supported, got kernel_shape dims: " + std::to_string(kernel_shape.size())};
    if (kernel_shape[0] <= 0 || kernel_shape[1] <= 0)
        return {
            false,
            ctx + "kernel dimensions must be positive"};
    if (pads.size() != 4)
        return {
            false,
            ctx + "expected 4 pad values, got: " + std::to_string(pads.size())};
    if (strides.size() != 2)
        return {
            false,
            ctx + "expected 2 stride values, got: " + std::to_string(strides.size())};
    if (dilations.size() != 2)
        return {
            false,
            ctx + "expected 2 dilation values, got: " + std::to_string(dilations.size())};

    return OPERATION_OK;
}

static std::string logical_node_ctx(size_t idx, Logical::Operator op) {
    return "[node " + std::to_string(idx) + " (" + Utils::op_name(op) + ")] ";
}

static OperationResult check_logical_node_arity(size_t idx, const Logical::Node& node) {
    const std::string ctx = logical_node_ctx(idx, node.op);

    switch (node.op) {
    case Logical::Operator::Conv:
    case Logical::Operator::ConvTranspose:
        if (node.inputs.size() < 3)
            return {false, ctx + "expected at least 3 inputs (activation, weights, bias), got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Logical::Operator::Gemm:
        if (node.inputs.size() != 3)
            return {false, ctx + "expected 3 inputs, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Logical::Operator::Im2Col:
        if (node.inputs.size() != 1)
            return {false, ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Logical::Operator::ReLU:
    case Logical::Operator::Sigmoid:
    case Logical::Operator::MaxPool2D:
        if (node.inputs.size() != 1)
            return {false, ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;

    case Logical::Operator::Unknown:
        return {false, ctx + "op is Unknown"};
    }
    return OPERATION_OK;
}

OperationResult validate_parse(const Logical::Graph& graph) {
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        const std::string ctx = logical_node_ctx(i, node.op);

        auto arity = check_logical_node_arity(i, node);
        if (!arity.success) return arity;

        switch (node.op) {
        case Logical::Operator::Gemm:
            if (!std::holds_alternative<Logical::GemmAttrs>(node.attributes))
                return {false, ctx + "expected GemmAttributes variant"};
            if (!std::get<Logical::GemmAttrs>(node.attributes).transB)
                return {false, ctx + "unsupported transB value `false`"};
            break;

        case Logical::Operator::Conv:
        case Logical::Operator::Im2Col: {
            if (!std::holds_alternative<Logical::ConvAttrs>(node.attributes))
                return {false, ctx + "expected ConvAttributes variant"};
            const auto& attrs = std::get<Logical::ConvAttrs>(node.attributes);
            auto vr = check_kernel_pads_strides_dilations(ctx, attrs.kernel_shape, attrs.pads, attrs.strides, attrs.dilations);
            if (!vr.success) return {vr};
            break;
        }

        case Logical::Operator::ConvTranspose: {
            if (!std::holds_alternative<Logical::ConvTransposeAttrs>(node.attributes))
                return {false, ctx + "expected ConvTransposeAttributes variant"};
            const auto& attrs = std::get<Logical::ConvTransposeAttrs>(node.attributes);
            auto vr = check_kernel_pads_strides_dilations(ctx, attrs.kernel_shape, attrs.pads, attrs.strides, attrs.dilations);
            if (!vr.success) return {vr};
            if (attrs.output_padding.size() != 2)
                return {false, ctx + "expected 2 output padding values, got: " + std::to_string(attrs.output_padding.size())};

            if (attrs.dilations[0] != 1.0 || attrs.dilations[1] != 1.0)
                return {false, ctx + "unsupported Conv ConvTransposeAttributes values. Currently, dilations=1 is supported"};

            break;
        }

        case Logical::Operator::MaxPool2D:
            if (!std::holds_alternative<Logical::MaxPool2DAttrs>(node.attributes))
                return {false, ctx + "expected MaxPool2DAttributes variant"};
            {
                const auto& attrs = std::get<Logical::MaxPool2DAttrs>(node.attributes);
                auto vr = check_kernel_pads_strides_dilations(ctx, attrs.kernel_shape, attrs.pads, attrs.strides, attrs.dilations);
                if (!vr.success) return {vr};
            }
            break;

        case Logical::Operator::ReLU:
        case Logical::Operator::Sigmoid:
            break;

        case Logical::Operator::Unknown:
            return {false, ctx + "op is Unknown"};
        }
    }
    return OPERATION_OK;
}

} // namespace ml::passes
