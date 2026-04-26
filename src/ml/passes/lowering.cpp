#include "lowering.hpp"

namespace ml::passes {

static OperationResult low_relu(const LogicalNode& node, PhysicalGraph& graph) {
    PhysicalNode n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.op = PhysicalOp::ReLU;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_sigmoid(const LogicalNode& node, PhysicalGraph& graph) {
    PhysicalNode n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.op = PhysicalOp::Sigmoid;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_gemm(const LogicalNode& node, PhysicalGraph& graph) {
    const auto& attrs = std::get<GemmAttributes>(node.attributes);

    PhysicalNode n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.attributes = attrs;
    n.op = PhysicalOp::Gemm;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_conv(const LogicalNode& node, PhysicalGraph& graph) {
    const auto& attrs = std::get<ConvAttributes>(node.attributes);

    const std::string col_name = node.outputs[0] + "__col";

    PhysicalNode im2col;
    im2col.inputs = {node.inputs}; // {activation, weights, bias}
    im2col.outputs = {col_name};
    im2col.attributes = attrs;
    im2col.op = PhysicalOp::Im2Col;

    PhysicalNode gemm;
    gemm.inputs = {col_name, node.inputs[1], node.inputs[2]};
    gemm.outputs = node.outputs;
    gemm.attributes = GemmAttributes{1.0f, 1.0f, true};
    gemm.op = PhysicalOp::Gemm;

    ReshapeAttributes reshape_attrs{
        .mode = ReshapeMode::GemmToImage,
        .image_shape_ref = col_name + "__4d"};

    PhysicalNode reshape;
    reshape.inputs = node.outputs;
    reshape.outputs = node.outputs;
    reshape.attributes = reshape_attrs;
    reshape.op = PhysicalOp::Reshape;

    graph.nodes.push_back(std::move(im2col));
    graph.nodes.push_back(std::move(gemm));
    graph.nodes.push_back(std::move(reshape));
    return {true, {}};
}

static OperationResult low_im2col(const LogicalNode& node, PhysicalGraph& graph) {
    const auto& attrs = std::get<ConvAttributes>(node.attributes);

    PhysicalNode n;
    n.inputs = {node.inputs[0]};
    n.outputs = {node.outputs[0]};
    n.attributes = attrs;
    n.op = PhysicalOp::Im2Col;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_conv_transpose(const LogicalNode& node, PhysicalGraph& graph) {
    const auto& attrs = std::get<ConvTransposeAttributes>(node.attributes);

    const std::string flat_name = node.inputs[0] + "__flat";
    const std::string gemm_name = node.outputs[0] + "__gemm";
    const std::string col2im_name = node.outputs[0] + "__col2im";

    // Reshape 1: [b, ic, ih, iw] -> [b*ih*iw, ic]
    PhysicalNode pre_reshape;
    pre_reshape.op = PhysicalOp::Reshape;
    pre_reshape.inputs = {node.inputs[0]};
    pre_reshape.outputs = {flat_name};
    pre_reshape.attributes = ReshapeAttributes{.mode = ReshapeMode::ImageToGemm};

    // ONNX ConvTranspose weights are [ic, oc, kH, kW]. The GEMM shader always
    // computes A x B^T, so transpose to [oc*kH*kW, ic] at load time.
    const std::string& weight_name = node.inputs[1];
    if (graph.initializers.find(weight_name) == graph.initializers.end())
        return {false, "ConvTranspose: weight tensor '" + weight_name + "' not found in initializers"};

    const std::string transposed_weight_name = weight_name + "__T";
    {
        const Tensor& w = graph.initializers.at(weight_name);
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
        graph.initializers[transposed_weight_name] = std::move(w_T);
    }

    // Gemm: [b*ih*iw, ic] x [oc*kh*kw, ic]^T -> [b*ih*iw, oc*kh*kw]
    GemmAttributes gemm_attrs{.alpha = 1.0f, .beta = 1.0f, .transB = true};

    PhysicalNode gemm;
    gemm.op = PhysicalOp::Gemm;
    gemm.inputs = {flat_name, transposed_weight_name, node.inputs[2]};
    gemm.outputs = {gemm_name};
    gemm.attributes = gemm_attrs;

    // Col2Im: [b*ih*iw, oc*kh*kw] -> [out_h*out_w, oc]
    Col2ImAttributes col2im_attrs;
    col2im_attrs.kernel_shape = attrs.kernel_shape;
    col2im_attrs.pads = attrs.pads;
    col2im_attrs.strides = attrs.strides;
    col2im_attrs.output_padding = attrs.output_padding;
    col2im_attrs.source_activation = node.inputs[0];

    PhysicalNode col2im;
    col2im.op = PhysicalOp::Col2Im;
    col2im.inputs = {gemm_name, node.inputs[2]};
    col2im.outputs = {col2im_name};
    col2im.attributes = col2im_attrs;

    // Reshape 2: [b*oh*ow, oc] -> [b, oc, oh, ow]
    PhysicalNode post_reshape;
    post_reshape.op = PhysicalOp::Reshape;
    post_reshape.inputs = {col2im_name};
    post_reshape.outputs = {node.outputs[0]};
    post_reshape.attributes = ReshapeAttributes{
        .mode = ReshapeMode::GemmToImage,
        .image_shape_ref = col2im_name + "__4d"};

    graph.nodes.push_back(std::move(pre_reshape));
    graph.nodes.push_back(std::move(gemm));
    graph.nodes.push_back(std::move(col2im));
    graph.nodes.push_back(std::move(post_reshape));
    return {true, {}};
}

LoweringResult lower(const LogicalGraph& logical_graph) {
    LoweringResult result;
    result.status = {true, {}};
    auto& graph = result.graph;

    graph.input_names = logical_graph.input_names;
    graph.initializers = logical_graph.initializers;

    for (const auto& node : logical_graph.nodes) {
        OperationResult op_result;

        switch (node.op) {
        case LogicalOp::ReLU:
            op_result = low_relu(node, graph);
            break;
        case LogicalOp::Sigmoid:
            op_result = low_sigmoid(node, graph);
            break;
        case LogicalOp::Gemm:
            op_result = low_gemm(node, graph);
            break;
        case LogicalOp::Conv:
            op_result = low_conv(node, graph);
            break;
        case LogicalOp::Im2Col:
            op_result = low_im2col(node, graph);
            break;
        case LogicalOp::ConvTranspose:
            op_result = low_conv_transpose(node, graph);
            break;
        case LogicalOp::Unknown:
            result.status = {false, "lowering: encountered unknown op"};
            return result;
        }

        if (!op_result.success) {
            result.status = op_result;
            return result;
        }
    }

    return result;
}

} // namespace ml::passes
