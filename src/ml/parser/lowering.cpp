#include "lowering.hpp"

namespace ml::Lowering {

static bool low_relu(
    const ml::LogicalNode& logical_node,
    ml::PhysicalGraph& physical_graph) {

    PhysicalNode physical_node;
    physical_node.inputs = logical_node.inputs;
    physical_node.outputs = logical_node.outputs;
    physical_node.op = PhysicalOp::ReLU;
    physical_graph.nodes.push_back(std::move(physical_node));
    return true;
}

static bool low_sigmoid(
    const ml::LogicalNode& logical_node,
    ml::PhysicalGraph& physical_graph) {

    PhysicalNode physical_node;
    physical_node.inputs = logical_node.inputs;
    physical_node.outputs = logical_node.outputs;
    physical_node.op = PhysicalOp::Sigmoid;
    physical_graph.nodes.push_back(std::move(physical_node));
    return true;
}

static bool low_gemm(
    const ml::LogicalNode& logical_node,
    ml::PhysicalGraph& physical_graph) {

    auto& attributes = std::get<GemmAttributes>(logical_node.attributes);
    auto result = attributes.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode physical_node;
    physical_node.inputs = logical_node.inputs;
    physical_node.outputs = logical_node.outputs;
    physical_node.attributes = logical_node.attributes;
    physical_node.op = PhysicalOp::Gemm;
    physical_graph.nodes.push_back(std::move(physical_node));
    return true;
}

static bool low_conv(
    const ml::LogicalNode& logical_node,
    ml::PhysicalGraph& physical_graph) {

    ERR_FAIL_COND_V_MSG(
        logical_node.inputs.size() < 3,
        false,
        ("Conv2D: expected at least 3 inputs (input, weights, bias), got: " + std::to_string(logical_node.inputs.size())).c_str());

    // There is no physical convolution since it performs badly, instead, the convolution
    // is split into two nodes: Im2Col + GeMM. This outputs the same result and performs
    // much better due to modern hardware architecture.

    auto& attributes = std::get<ConvAttributes>(logical_node.attributes);
    auto result = attributes.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    // Transient tensor name
    const std::string col_name = logical_node.outputs[0] + "__col";

    auto reshape_info = std::make_shared<ReshapeInfo>();

    PhysicalNode im2col_node;
    im2col_node.inputs = {logical_node.inputs}; // Input activations
    im2col_node.outputs = {col_name};
    im2col_node.attributes = logical_node.attributes;
    im2col_node.reshape_info = reshape_info;
    im2col_node.op = PhysicalOp::Im2Col;

    PhysicalNode gemm_node;
    gemm_node.inputs = {
        col_name,               // Transient tensor name
        logical_node.inputs[1], // Weights
        logical_node.inputs[2]  // Bias
    };
    gemm_node.outputs = logical_node.outputs;
    gemm_node.attributes = GemmAttributes{1.0f, 1.0f, true};
    gemm_node.op = PhysicalOp::Gemm;

    PhysicalNode reshape_node;
    reshape_node.inputs = logical_node.outputs;
    reshape_node.outputs = logical_node.outputs;
    reshape_node.reshape_info = reshape_info;
    reshape_node.op = PhysicalOp::Reshape;

    physical_graph.nodes.push_back(std::move(im2col_node));
    physical_graph.nodes.push_back(std::move(gemm_node));
    physical_graph.nodes.push_back(std::move(reshape_node));
    return true;
}

static bool low_im2col(
    const ml::LogicalNode& logical_node,
    ml::PhysicalGraph& physical_graph) {

    auto& attributes = std::get<ConvAttributes>(logical_node.attributes);
    auto result = attributes.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode physical_node;
    physical_node.inputs = logical_node.inputs;
    physical_node.outputs = logical_node.outputs;
    physical_node.attributes = attributes;
    physical_node.op = PhysicalOp::Im2Col;
    physical_graph.nodes.push_back(std::move(physical_node));
    return true;
}

bool lower(ml::LogicalGraph& logical_graph, ml::PhysicalGraph& physical_graph) {

    physical_graph.input_names = logical_graph.input_names;
    physical_graph.input_shape = logical_graph.input_shape;
    physical_graph.initializers = logical_graph.initializers;

    for (const auto& logical_node : logical_graph.nodes) {

        bool success = true;

        switch (logical_node.op) {
        case LogicalOp::ReLU:
            success = low_relu(logical_node, physical_graph);
            break;
        case LogicalOp::Sigmoid:
            success = low_sigmoid(logical_node, physical_graph);
            break;
        case LogicalOp::Gemm:
            success = low_gemm(logical_node, physical_graph);
            break;
        case LogicalOp::Conv:
            low_conv(logical_node, physical_graph);
            break;
        case LogicalOp::Im2Col:
            success = low_im2col(logical_node, physical_graph);
            break;
        case ml::LogicalOp::Unknown:
            return false;
        }

        if (!success) {
            return false;
        }
    }

    // Implementation for lowering logical graph to physical graph
    return true;
}
} // namespace ml::Lowering