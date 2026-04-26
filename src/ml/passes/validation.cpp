#include "validation.hpp"

#include "core/utils.hpp"

#include <unordered_set>

namespace ml::passes {

static std::string node_ctx(size_t idx, PhysicalOp op) {
    return "[node " + std::to_string(idx) + " (" + Utils::op_name(op) + ")] ";
}

static OperationResult check_connectivity(const PhysicalGraph& graph) {
    std::unordered_set<std::string> defined;

    for (const auto& name : graph.input_names)
        defined.insert(name);
    for (const auto& [name, _] : graph.initializers)
        defined.insert(name);

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        const std::string ctx = node_ctx(i, node.op);

        for (const auto& input : node.inputs) {
            if (defined.find(input) == defined.end())
                return {false, ctx + "input tensor '" + input + "' is not defined"};
        }

        for (const auto& output : node.outputs) {
            if (node.op != PhysicalOp::Reshape && defined.count(output))
                return {false, ctx + "output tensor '" + output + "' already defined (duplicate output)"};
            defined.insert(output);
        }
    }
    return {true, {}};
}

static OperationResult check_node_arity(size_t idx, const PhysicalNode& node) {
    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case PhysicalOp::Im2Col:
        if (node.inputs.empty())
            return {false, ctx + "expected at least 1 input, got 0"};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case PhysicalOp::Gemm:
    case PhysicalOp::Conv:
        if (node.inputs.size() != 3)
            return {false, ctx + "expected 3 inputs, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case PhysicalOp::Col2Im:
        if (node.inputs.size() != 2)
            return {false, ctx + "expected 2 inputs, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case PhysicalOp::Reshape:
    case PhysicalOp::ReLU:
    case PhysicalOp::Sigmoid:
        if (node.inputs.size() != 1)
            return {false, ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case PhysicalOp::Unknown:
        return {false, ctx + "op is Unknown - node cannot be dispatched"};
    }
    return {true, {}};
}

static OperationResult check_conv_kernel_pads_strides(
    const std::string& ctx,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& strides) {

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
    return {true, {}};
}

static OperationResult check_node_attributes(
    size_t idx,
    const PhysicalNode& node,
    const std::unordered_set<std::string>& defined) {

    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case PhysicalOp::Im2Col:
    case PhysicalOp::Conv:
        if (!std::holds_alternative<ConvAttributes>(node.attributes))
            return {false, ctx + "expected ConvAttributes variant"};
        {
            const auto& attrs = std::get<ConvAttributes>(node.attributes);
            auto vr = check_conv_kernel_pads_strides(ctx, attrs.kernel_shape, attrs.pads, attrs.strides);
            if (!vr.success) return vr;
        }
        break;

    case PhysicalOp::Gemm:
        if (!std::holds_alternative<GemmAttributes>(node.attributes))
            return {false, ctx + "expected GemmAttributes variant"};
        if (!std::get<GemmAttributes>(node.attributes).transB)
            return {false, ctx + "unsupported transB value `false`"};
        break;

    case PhysicalOp::Col2Im:
        if (!std::holds_alternative<Col2ImAttributes>(node.attributes))
            return {false, ctx + "expected Col2ImAttributes variant"};
        {
            const auto& attrs = std::get<Col2ImAttributes>(node.attributes);
            auto vr = check_conv_kernel_pads_strides(ctx, attrs.kernel_shape, attrs.pads, attrs.strides);
            if (!vr.success) return vr;
            if (attrs.output_padding.size() != 2)
                return {false, ctx + "expected 2 output padding values, got: " + std::to_string(attrs.output_padding.size())};
            if (attrs.source_activation.empty())
                return {false, ctx + "source activation name cannot be empty"};
            if (defined.find(attrs.source_activation) == defined.end())
                return {false, ctx + "source_activation tensor '" + attrs.source_activation + "' not found in defined tensor set"};
        }
        break;

    case PhysicalOp::Reshape:
        if (!std::holds_alternative<ReshapeAttributes>(node.attributes))
            return {false, ctx + "expected ReshapeAttributes variant"};
        {
            const auto& attrs = std::get<ReshapeAttributes>(node.attributes);
            if (attrs.mode == ReshapeMode::GemmToImage && attrs.image_shape_ref.empty())
                return {false, ctx + "GemmToImage mode requires non-empty image_shape_ref"};
        }
        break;

    case PhysicalOp::ReLU:
    case PhysicalOp::Sigmoid:
        break;

    case PhysicalOp::Unknown:
        return {false, ctx + "op is Unknown"};
    }

    return {true, {}};
}

static std::string logical_node_ctx(size_t idx, LogicalOp op) {
    return "[node " + std::to_string(idx) + " (" + Utils::op_name(op) + ")] ";
}

static OperationResult check_logical_node_arity(size_t idx, const LogicalNode& node) {
    const std::string ctx = logical_node_ctx(idx, node.op);

    switch (node.op) {
    case LogicalOp::Conv:
    case LogicalOp::ConvTranspose:
        if (node.inputs.size() < 3)
            return {false, ctx + "expected at least 3 inputs (activation, weights, bias), got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case LogicalOp::Gemm:
        if (node.inputs.size() != 3)
            return {false, ctx + "expected 3 inputs, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case LogicalOp::Im2Col:
        if (node.inputs.size() != 1)
            return {false, ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case LogicalOp::ReLU:
    case LogicalOp::Sigmoid:
        if (node.inputs.size() != 1)
            return {false, ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case LogicalOp::Unknown:
        return {false, ctx + "op is Unknown"};
    }
    return {true, {}};
}

ValidationResult validate(const LogicalGraph& graph) {
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        const std::string ctx = logical_node_ctx(i, node.op);

        auto arity = check_logical_node_arity(i, node);
        if (!arity.success) return {arity};

        switch (node.op) {
        case LogicalOp::Gemm:
            if (!std::holds_alternative<GemmAttributes>(node.attributes))
                return {{false, ctx + "expected GemmAttributes variant"}};
            if (!std::get<GemmAttributes>(node.attributes).transB)
                return {{false, ctx + "unsupported transB value `false`"}};
            break;

        case LogicalOp::Conv:
        case LogicalOp::Im2Col: {
            if (!std::holds_alternative<ConvAttributes>(node.attributes))
                return {{false, ctx + "expected ConvAttributes variant"}};
            const auto& attrs = std::get<ConvAttributes>(node.attributes);
            auto vr = check_conv_kernel_pads_strides(ctx, attrs.kernel_shape, attrs.pads, attrs.strides);
            if (!vr.success) return {vr};
            break;
        }

        case LogicalOp::ConvTranspose: {
            if (!std::holds_alternative<ConvTransposeAttributes>(node.attributes))
                return {{false, ctx + "expected ConvTransposeAttributes variant"}};
            const auto& attrs = std::get<ConvTransposeAttributes>(node.attributes);
            auto vr = check_conv_kernel_pads_strides(ctx, attrs.kernel_shape, attrs.pads, attrs.strides);
            if (!vr.success) return {vr};
            if (attrs.output_padding.size() != 2)
                return {{false, ctx + "expected 2 output padding values, got: " + std::to_string(attrs.output_padding.size())}};
            break;
        }

        case LogicalOp::ReLU:
        case LogicalOp::Sigmoid:
            break;

        case LogicalOp::Unknown:
            return {{false, ctx + "op is Unknown"}};
        }
    }
    return {{true, {}}};
}

ValidationResult validate(const PhysicalGraph& graph) {
    auto conn = check_connectivity(graph);
    if (!conn.success)
        return {{false, conn.error}};

    std::unordered_set<std::string> defined;
    for (const auto& name : graph.input_names)
        defined.insert(name);
    for (const auto& [name, _] : graph.initializers)
        defined.insert(name);
    for (const auto& node : graph.nodes)
        for (const auto& out : node.outputs)
            defined.insert(out);

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        auto arity = check_node_arity(i, node);
        if (!arity.success) return {{false, arity.error}};

        auto attrs = check_node_attributes(i, node, defined);
        if (!attrs.success) return {{false, attrs.error}};
    }

    return {{true, {}}};
}

} // namespace ml::passes
