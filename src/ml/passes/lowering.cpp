#include "lowering.hpp"

namespace ml::passes {

static OperationResult low_relu(const Logical::Node& node, Physical::Graph& graph) {
    Physical::Node n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.op = Physical::Operator::ReLU;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_sigmoid(const Logical::Node& node, Physical::Graph& graph) {
    Physical::Node n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;
    n.op = Physical::Operator::Sigmoid;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_gemm(const Logical::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<Logical::GemmAttrs>(node.attributes);

    Physical::Node n;
    n.inputs = node.inputs;
    n.outputs = node.outputs;

    Physical::GemmAttrs attrs;
    attrs.alpha = l_attrs.alpha;
    attrs.beta = l_attrs.beta;
    attrs.transB = l_attrs.transB;
    n.attributes = attrs;

    n.op = Physical::Operator::Gemm;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_conv(const Logical::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<Logical::ConvAttrs>(node.attributes);

    const std::string col_name = node.outputs[0] + "__col";

    Physical::Node im2col;
    im2col.inputs = {node.inputs}; // {activation, weights, bias}
    im2col.outputs = {col_name};

    Physical::ConvAttrs attrs;
    attrs.kernel_shape = l_attrs.kernel_shape;
    attrs.pads = l_attrs.pads;
    attrs.strides = l_attrs.strides;
    attrs.dilations = l_attrs.dilations;
    im2col.attributes = attrs;

    im2col.op = Physical::Operator::Im2Col;

    Physical::Node gemm;
    gemm.inputs = {col_name, node.inputs[1], node.inputs[2]};
    gemm.outputs = node.outputs;
    gemm.attributes = Physical::GemmAttrs{1.0f, 1.0f, true};
    gemm.op = Physical::Operator::Gemm;

    Physical::ReshapeAttrs reshape_attrs{
        .mode = Physical::ReshapeMode::GemmToImage,
        .image_shape_ref = col_name + "__4d"};

    Physical::Node reshape;
    reshape.inputs = node.outputs;
    reshape.outputs = node.outputs;
    reshape.attributes = reshape_attrs;
    reshape.op = Physical::Operator::Reshape;

    graph.nodes.push_back(std::move(im2col));
    graph.nodes.push_back(std::move(gemm));
    graph.nodes.push_back(std::move(reshape));
    return {true, {}};
}

static OperationResult low_im2col(const Logical::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<Logical::ConvAttrs>(node.attributes);

    Physical::Node n;
    n.inputs = {node.inputs[0]};
    n.outputs = {node.outputs[0]};

    Physical::ConvAttrs attrs;
    attrs.kernel_shape = l_attrs.kernel_shape;
    attrs.pads = l_attrs.pads;
    attrs.strides = l_attrs.strides;
    attrs.dilations = l_attrs.dilations;

    n.attributes = attrs;
    n.op = Physical::Operator::Im2Col;
    graph.nodes.push_back(std::move(n));
    return {true, {}};
}

static OperationResult low_conv_transpose(const Logical::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<Logical::ConvTransposeAttrs>(node.attributes);

    const std::string flat_name = node.inputs[0] + "__flat";
    const std::string gemm_name = node.outputs[0] + "__gemm";
    const std::string col2im_name = node.outputs[0] + "__col2im";

    // Reshape 1: [b, ic, ih, iw] -> [b*ih*iw, ic]
    Physical::Node pre_reshape;
    pre_reshape.op = Physical::Operator::Reshape;
    pre_reshape.inputs = {node.inputs[0]};
    pre_reshape.outputs = {flat_name};
    pre_reshape.attributes = Physical::ReshapeAttrs{.mode = Physical::ReshapeMode::ImageToGemm};

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
    Physical::GemmAttrs gemm_attrs{.alpha = 1.0f, .beta = 1.0f, .transB = true};

    Physical::Node gemm;
    gemm.op = Physical::Operator::Gemm;
    gemm.inputs = {flat_name, transposed_weight_name, node.inputs[2]};
    gemm.outputs = {gemm_name};
    gemm.attributes = gemm_attrs;

    // Col2Im: [b*ih*iw, oc*kh*kw] -> [out_h*out_w, oc]
    Physical::Col2ImAttrs col2im_attrs;
    col2im_attrs.kernel_shape = l_attrs.kernel_shape;
    col2im_attrs.pads = l_attrs.pads;
    col2im_attrs.strides = l_attrs.strides;
    col2im_attrs.output_padding = l_attrs.output_padding;
    col2im_attrs.dilations = l_attrs.dilations;
    col2im_attrs.source_activation = node.inputs[0];

    Physical::Node col2im;
    col2im.op = Physical::Operator::Col2Im;
    col2im.inputs = {gemm_name, node.inputs[2]};
    col2im.outputs = {col2im_name};
    col2im.attributes = col2im_attrs;

    // Reshape 2: [b*oh*ow, oc] -> [b, oc, oh, ow]
    Physical::Node post_reshape;
    post_reshape.op = Physical::Operator::Reshape;
    post_reshape.inputs = {col2im_name};
    post_reshape.outputs = {node.outputs[0]};
    post_reshape.attributes = Physical::ReshapeAttrs{
        .mode = Physical::ReshapeMode::GemmToImage,
        .image_shape_ref = col2im_name + "__4d"};

    graph.nodes.push_back(std::move(pre_reshape));
    graph.nodes.push_back(std::move(gemm));
    graph.nodes.push_back(std::move(col2im));
    graph.nodes.push_back(std::move(post_reshape));
    return {true, {}};
}

static OperationResult low_max_pool_2d(const Logical::Node& node, Physical::Graph& graph) {
    const auto& l_attrs = std::get<Logical::MaxPool2DAttrs>(node.attributes);

    Physical::Node max_pool;
    max_pool.op = Physical::Operator::MaxPool2D;
    max_pool.inputs = {node.inputs[0]};
    max_pool.outputs = {node.outputs[0]};

    max_pool.attributes = Physical::MaxPool2DAttrs{
        .kernel_shape = l_attrs.kernel_shape,
        .pads = l_attrs.pads,
        .strides = l_attrs.strides,
        .dilations = l_attrs.dilations};

    graph.nodes.push_back(std::move(max_pool));
    return {true, {}};
}

LoweringResult lower(const Logical::Graph& logical_graph) {
    LoweringResult result;
    result.status = {true, {}};
    auto& graph = result.graph;

    graph.input_names = logical_graph.input_names;
    graph.initializers = logical_graph.initializers;

    for (const auto& node : logical_graph.nodes) {
        OperationResult op_result;

        switch (node.op) {
        case Logical::Operator::ReLU:
            op_result = low_relu(node, graph);
            break;
        case Logical::Operator::Sigmoid:
            op_result = low_sigmoid(node, graph);
            break;
        case Logical::Operator::Gemm:
            op_result = low_gemm(node, graph);
            break;
        case Logical::Operator::Conv:
            op_result = low_conv(node, graph);
            break;
        case Logical::Operator::Im2Col:
            op_result = low_im2col(node, graph);
            break;
        case Logical::Operator::ConvTranspose:
            op_result = low_conv_transpose(node, graph);
            break;
        case Logical::Operator::MaxPool2D:
            op_result = low_max_pool_2d(node, graph);
            break;
        case Logical::Operator::Unknown:
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
