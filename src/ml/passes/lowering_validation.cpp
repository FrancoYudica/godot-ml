#include "lowering_validation.hpp"

#include <unordered_set>

namespace ml::passes {

static std::string node_ctx(size_t idx, Physical::Operator op) {
    return "[node " + std::to_string(idx) + " (" + Utils::op_name(op) + ")] ";
}

static OperationResult check_connectivity(const Physical::Graph& graph) {
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
            if (node.op != Physical::Operator::Reshape && defined.count(output))
                return {false, ctx + "output tensor '" + output + "' already defined (duplicate output)"};
            defined.insert(output);
        }
    }
    return {true};
}

static OperationResult check_node_arity(size_t idx, const Physical::Node& node) {
    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case Physical::Operator::Im2Col:
        if (node.inputs.empty())
            return {false, ctx + "expected at least 1 input, got 0"};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Physical::Operator::Gemm:
    case Physical::Operator::Conv:
        if (node.inputs.size() != 3)
            return {false, ctx + "expected 3 inputs, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Physical::Operator::Col2Im:
        if (node.inputs.size() != 2)
            return {false, ctx + "expected 2 inputs, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Physical::Operator::Reshape:
    case Physical::Operator::ReLU:
    case Physical::Operator::Sigmoid:
    case Physical::Operator::MaxPool2D:
        if (node.inputs.size() != 1)
            return {false, ctx + "expected 1 input, got " + std::to_string(node.inputs.size())};
        if (node.outputs.size() != 1)
            return {false, ctx + "expected 1 output, got " + std::to_string(node.outputs.size())};
        break;
    case Physical::Operator::Unknown:
        return {false, ctx + "op is Unknown - node cannot be dispatched"};
    }
    return {true};
}

static OperationResult check_node_attributes(
    size_t idx,
    const Physical::Node& node,
    const std::unordered_set<std::string>& defined) {

    const std::string ctx = node_ctx(idx, node.op);

    switch (node.op) {
    case Physical::Operator::Im2Col:
    case Physical::Operator::Conv:
        if (!std::holds_alternative<Physical::ConvAttrs>(node.attributes))
            return {false, ctx + "expected ConvAttributes variant"};
        {
            const auto& attrs = std::get<Physical::ConvAttrs>(node.attributes);

            if (attrs.dilation_x != 1 || attrs.dilation_y != 1)
                return {false, ctx + "only unit dilations supported"};
        }

        break;

    case Physical::Operator::Gemm:
        if (!std::holds_alternative<Physical::GemmAttrs>(node.attributes))
            return {false, ctx + "expected GemmAttributes variant"};
        break;

    case Physical::Operator::Col2Im:
        if (!std::holds_alternative<Physical::Col2ImAttrs>(node.attributes))
            return {false, ctx + "expected Col2ImAttributes variant"};
        {
            const auto& attrs = std::get<Physical::Col2ImAttrs>(node.attributes);
            if (attrs.source_activation.empty())
                return {false, ctx + "source activation name cannot be empty"};
            if (defined.find(attrs.source_activation) == defined.end())
                return {false, ctx + "source_activation tensor '" + attrs.source_activation + "' not found in defined tensor set"};

            if (attrs.dilation_x != 1 || attrs.dilation_y != 1)
                return {false, ctx + "only unit dilations supported"};

            if (attrs.output_padding_x != 0 || attrs.output_padding_y != 0)
                return {false, ctx + "only zero output padding supported"};
        }
        break;

    case Physical::Operator::Reshape:
        if (!std::holds_alternative<Physical::ReshapeAttrs>(node.attributes))
            return {false, ctx + "expected ReshapeAttributes variant"};
        {
            const auto& attrs = std::get<Physical::ReshapeAttrs>(node.attributes);
            if (attrs.mode == Physical::ReshapeMode::GemmToImage && attrs.image_shape_ref.empty())
                return {false, ctx + "GemmToImage mode requires non-empty image_shape_ref"};
        }
        break;

    case ml::Physical::Operator::MaxPool2D:
        if (!std::holds_alternative<Physical::MaxPool2DAttrs>(node.attributes))
            return {false, ctx + "expected MaxPool2DAttributes variant"};
        break;

    case Physical::Operator::ReLU:
    case Physical::Operator::Sigmoid:
        break;

    case Physical::Operator::Unknown:
        return {false, ctx + "op is Unknown"};
    }

    return OPERATION_OK;
}

OperationResult lowering_validation(const Physical::Graph& graph) {
    auto conn = check_connectivity(graph);
    if (!conn.success)
        return {false, conn.error};

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
        if (!arity.success) return {false, arity.error};

        auto attrs = check_node_attributes(i, node, defined);
        if (!attrs.success) return {false, attrs.error};
    }

    return OPERATION_OK;
}
} // namespace ml::passes
