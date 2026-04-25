#include "lowering.hpp"

using namespace godot;

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
    physical_node.attributes = attributes;
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
    im2col_node.attributes = attributes;
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

    ReshapeAttributes reshape_attrs{.mode = ReshapeMode::GemmToImage};
    result = reshape_attrs.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode reshape_node;
    reshape_node.inputs = logical_node.outputs;
    reshape_node.outputs = logical_node.outputs;
    reshape_node.reshape_info = reshape_info;
    reshape_node.attributes = reshape_attrs;
    reshape_node.op = PhysicalOp::Reshape;

    physical_graph.nodes.push_back(std::move(im2col_node));
    physical_graph.nodes.push_back(std::move(gemm_node));
    physical_graph.nodes.push_back(std::move(reshape_node));
    return true;
}

static bool low_conv_transpose(
    const ml::LogicalNode& logical_node,
    ml::PhysicalGraph& physical_graph) {

    ERR_FAIL_COND_V_MSG(
        logical_node.inputs.size() < 3,
        false,
        "ConvTranspose: expected at least 3 inputs, got: " +
            String(std::to_string(logical_node.inputs.size()).c_str()));

    ERR_FAIL_COND_V_MSG(
        logical_node.outputs.size() != 1,
        false,
        "ConvTranspose: expected 1 output, got: " +
            String(std::to_string(logical_node.outputs.size()).c_str()));

    auto& attributes = std::get<ConvTransposeAttributes>(logical_node.attributes);
    auto result = attributes.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    auto reshape_info = std::make_shared<ReshapeInfo>();

    const std::string flat_name = logical_node.inputs[0] + "__flat";
    const std::string gemm_name = logical_node.outputs[0] + "__gemm";
    const std::string col2im_name = logical_node.outputs[0] + "__col2im";

    // Reshape 1: [b, ic, ih, iw] -> [b*ih*iw, ic]
    // ImageToGemm mode computes this dynamically from the input shape at dispatch

    ReshapeAttributes pre_reshape_attrs;
    pre_reshape_attrs.mode = ReshapeMode::ImageToGemm;
    result = pre_reshape_attrs.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode pre_reshape;
    pre_reshape.op = PhysicalOp::Reshape;
    pre_reshape.inputs = {logical_node.inputs[0]};
    pre_reshape.outputs = {flat_name};
    pre_reshape.attributes = pre_reshape_attrs;

    // ONNX ConvTranspose weights are [ic, oc, kH, kW]. The GEMM shader always
    // computes A x B^T (accessing B as [N, K]), so transpose to [oc*kH*kW, ic]
    // at load time to match that access pattern.
    const std::string& weight_name = logical_node.inputs[1];
    ERR_FAIL_COND_V_MSG(
        physical_graph.initializers.find(weight_name) == physical_graph.initializers.end(),
        false,
        "ConvTranspose: weight tensor not found in initializers");

    // CPU transposition
    const std::string transposed_weight_name = weight_name + "__T";
    {
        const Tensor& w = physical_graph.initializers.at(weight_name);
        const int64_t K = w.shape[0]; // in_channels
        int64_t N = 1;
        for (size_t i = 1; i < w.shape.size(); ++i)
            N *= w.shape[i]; // out_channels * kH * kW

        Tensor w_T;
        w_T.name = transposed_weight_name;
        w_T.shape = {N, K};
        w_T.data.resize(static_cast<size_t>(N * K));
        for (int64_t k = 0; k < K; ++k)
            for (int64_t n = 0; n < N; ++n)
                w_T.data[n * K + k] = w.data[k * N + n];
        physical_graph.initializers[transposed_weight_name] = std::move(w_T);
    }

    // Gemm: [b*ih*iw, ic] x [oc*kh*kw, ic]^T -> [b*ih*iw, oc*kh*kw]
    GemmAttributes gemm_attrs{
        .alpha = 1.0,
        .beta = 1.0,
        .transB = true};
    result = gemm_attrs.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode gemm_node;
    gemm_node.op = PhysicalOp::Gemm;
    gemm_node.inputs = {flat_name,
                        transposed_weight_name,  // weights [oc*kh*kw, ic] (transposed)
                        logical_node.inputs[2]}; // bias [oc]
    gemm_node.outputs = {gemm_name};
    gemm_node.attributes = gemm_attrs;

    // Col2Im: [b*ih*iw, oc*kh*kw] -> writes [b, oc, oh, ow] into reshape_info
    Col2ImAttributes col2im_attrs;
    col2im_attrs.kernel_shape = attributes.kernel_shape;
    col2im_attrs.pads = attributes.pads;
    col2im_attrs.strides = attributes.strides;
    col2im_attrs.output_padding = attributes.output_padding;
    col2im_attrs.source_activation = logical_node.inputs[0]; // reach back to [b, ic, ih, iw]
    result = col2im_attrs.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode col2im_node;
    col2im_node.op = PhysicalOp::Col2Im;
    col2im_node.inputs = {
        gemm_name,
        logical_node.inputs[2] // So it can get bias/channels dimension
    };
    col2im_node.outputs = {col2im_name};
    col2im_node.attributes = col2im_attrs;
    col2im_node.reshape_info = reshape_info; // Col2Im writes oh, ow, oc, b here

    // Reshape 2: [b*oh*ow, oc] -> [b, oc, oh, ow]
    // GemmToImage mode reads shape from reshape_info written by Col2Im

    ReshapeAttributes post_reshape_attrs;
    post_reshape_attrs.mode = ReshapeMode::GemmToImage;
    result = post_reshape_attrs.validate();
    ERR_FAIL_COND_V_MSG(!result.success, false, result.error.c_str());

    PhysicalNode post_reshape;
    post_reshape.op = PhysicalOp::Reshape;
    post_reshape.inputs = {col2im_name};
    post_reshape.outputs = {logical_node.outputs[0]}; // original output name
    post_reshape.attributes = post_reshape_attrs;
    post_reshape.reshape_info = reshape_info; // reads from what Col2Im wrote

    physical_graph.nodes.push_back(std::move(pre_reshape));
    physical_graph.nodes.push_back(std::move(gemm_node));
    physical_graph.nodes.push_back(std::move(col2im_node));
    physical_graph.nodes.push_back(std::move(post_reshape));

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
            success = low_conv(logical_node, physical_graph);
            break;
        case LogicalOp::Im2Col:
            success = low_im2col(logical_node, physical_graph);
            break;
        case ml::LogicalOp::ConvTranspose:
            success = low_conv_transpose(logical_node, physical_graph);
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