#include "shape_inference.hpp"

using namespace ml::Physical;

namespace ml::passes {

// Helper: looks up a shape and sets an error on result if missing.
static const std::vector<int64_t>* require(
    const std::string& name,
    const ShapeTable& shapes,
    ShapeInferenceResult& result) {

    auto it = shapes.find(name);
    if (it == shapes.end()) {
        result.status = {false, "shape_inference: tensor '" + name + "' not found"};
        return nullptr;
    }
    return &it->second;
}

// Im2Col: [b, c, h, w] -> [out_h*out_w, c*kH*kW]
// Also writes "<output>__4d" = [b, out_c, out_h, out_w] when weights are present
// (inputs[1] = weights, available when lowered from Conv).
static bool infer_im2col(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    if (!in) return false;

    auto& attrs = std::get<ConvAttrs>(node.attributes);
    int64_t b = (*in)[0], c = (*in)[1], h = (*in)[2], w = (*in)[3];
    int64_t kH = attrs.kernel_h, kW = attrs.kernel_w;

    int64_t kh_effective = (kH - 1) * attrs.dilation_y + 1;
    int64_t kw_effective = (kW - 1) * attrs.dilation_x + 1;

    int64_t pad_horizontal = attrs.padding_left + attrs.padding_right;
    int64_t pad_vertical = attrs.padding_top + attrs.padding_bottom;

    int64_t out_h = (h + pad_vertical - kh_effective) / attrs.stride_y + 1;
    int64_t out_w = (w + pad_horizontal - kw_effective) / attrs.stride_x + 1;

    shapes[node.outputs[0]] = {out_h * out_w, c * kH * kW};

    // Write 4D meta-shape for the downstream GemmToImage reshape.
    // weights are inputs[1] when lowered from Conv (shape [out_c, c, kH, kW]).
    if (node.inputs.size() > 1) {
        const auto* weight_shape = require(node.inputs[1], shapes, result);
        if (!weight_shape) return false;
        int64_t out_c = (*weight_shape)[0];
        shapes[node.outputs[0] + "__4d"] = {b, out_c, out_h, out_w};
    }

    return true;
}
// Gemm: [M, K] x [N, K]^T -> [M, N]
// Weights are always stored as [N, K] (transB=true).
static bool infer_gemm(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    const auto* w = require(node.inputs[1], shapes, result);
    if (!in || !w) return false;

    int64_t M = (*in)[0];
    int64_t N = (*w)[0]; // [N, K] layout - first dim is output features
    shapes[node.outputs[0]] = {M, N};
    return true;
}

// Col2Im: [b*ih*iw, oc*kH*kW] -> [out_h*out_w, oc]
// Also writes "<output>__4d" = [b, oc, out_h, out_w].
// out_c comes from bias shape (inputs[1] = bias [oc]).
// Spatial output dims come from attrs.source_activation shape [b, ic, ih, iw].
static bool infer_col2im(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto& attrs = std::get<Col2ImAttrs>(node.attributes);

    const auto* src = require(attrs.source_activation, shapes, result);
    const auto* bias = require(node.inputs[1], shapes, result);
    if (!src || !bias) return false;

    int64_t b = (*src)[0];
    int64_t ih = (*src)[2];
    int64_t iw = (*src)[3];

    int64_t kH = attrs.kernel_h, kW = attrs.kernel_w;

    int64_t kh_effective = (kH - 1) * attrs.dilation_y + 1;
    int64_t kw_effective = (kW - 1) * attrs.dilation_x + 1;

    int64_t pad_horizontal = attrs.padding_left + attrs.padding_right;
    int64_t pad_vertical = attrs.padding_top + attrs.padding_bottom;

    int64_t out_h = attrs.stride_y * (ih - 1) + attrs.output_padding_y + kh_effective - pad_vertical;
    int64_t out_w = attrs.stride_x * (iw - 1) + attrs.output_padding_x + kw_effective - pad_horizontal;

    int64_t out_c = (*bias)[0];

    shapes[node.outputs[0]] = {out_h * out_w, out_c};
    shapes[node.outputs[0] + "__4d"] = {b, out_c, out_h, out_w};
    return true;
}

// Reshape (ImageToGemm): [b, c, h, w] -> [b*h*w, c]
// Reshape (GemmToImage): reads target shape from ShapeTable[attrs.image_shape_ref]
static bool infer_reshape(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& attrs = std::get<ReshapeAttrs>(node.attributes);

    switch (attrs.mode) {
    case ReshapeMode::ImageToGemm: {
        const auto* in = require(node.inputs[0], shapes, result);
        if (!in) return false;
        int64_t b = (*in)[0], c = (*in)[1], h = (*in)[2], w = (*in)[3];
        shapes[node.outputs[0]] = {b * h * w, c};
        break;
    }
    case ReshapeMode::GemmToImage: {
        const auto* target = require(attrs.image_shape_ref, shapes, result);
        if (!target) return false;
        shapes[node.outputs[0]] = *target;
        break;
    }
    }
    return true;
}

// ReLU / Sigmoid: output shape = input shape
static bool infer_elementwise(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto* in = require(node.inputs[0], shapes, result);
    if (!in) return false;
    shapes[node.outputs[0]] = *in;
    return true;
}

// Direct Conv2D (not lowered): [b, c, h, w] -> [b, out_c, out_h, out_w]
static bool infer_conv2d(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    const auto* w = require(node.inputs[1], shapes, result);
    if (!in || !w) return false;

    const auto& attrs = std::get<ConvAttrs>(node.attributes);
    int64_t b = (*in)[0];
    int64_t h = (*in)[2];
    int64_t iw = (*in)[3];
    int64_t out_c = (*w)[0];

    int64_t kH = attrs.kernel_h;
    int64_t kW = attrs.kernel_w;

    int64_t kh_effective = (kH - 1) * attrs.dilation_y + 1;
    int64_t kw_effective = (kW - 1) * attrs.dilation_x + 1;

    int64_t pad_horizontal = attrs.padding_left + attrs.padding_right;
    int64_t pad_vertical = attrs.padding_top + attrs.padding_bottom;

    int64_t out_h = (h + pad_vertical - kh_effective) / attrs.stride_y + 1;
    int64_t out_w = (iw + pad_horizontal - kw_effective) / attrs.stride_x + 1;

    shapes[node.outputs[0]] = {b, out_c, out_h, out_w};
    return true;
}
// MaxPool2D: [b, c, h, w] -> [b, c, out_h, out_w]
static bool infer_max_pool_2d(const Node& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    if (!in) return false;

    const auto& attrs = std::get<MaxPool2DAttrs>(node.attributes);
    int64_t b = (*in)[0]; // Batches
    int64_t c = (*in)[1]; // Channels
    int64_t h = (*in)[2]; // Height
    int64_t w = (*in)[3]; // Width

    int64_t kh = attrs.kernel_h;
    int64_t kw = attrs.kernel_w;

    int64_t kh_effective = (kh - 1) * attrs.dilation_y + 1;
    int64_t kw_effective = (kw - 1) * attrs.dilation_x + 1;

    int64_t pad_horizontal = attrs.padding_left + attrs.padding_right;
    int64_t pad_vertical = attrs.padding_top + attrs.padding_bottom;

    int64_t out_h = (h + pad_vertical - kh_effective) / attrs.stride_y + 1;
    int64_t out_w = (w + pad_horizontal - kw_effective) / attrs.stride_x + 1;

    shapes[node.outputs[0]] = {b, c, out_h, out_w};
    return true;
}

ShapeInferenceResult infer_shapes(
    const Graph& graph,
    const ShapeTable& input_shapes) {

    ShapeInferenceResult result;
    result.status = {true, {}};
    auto& shapes = result.shapes;

    // Seed with caller-provided input shapes
    for (const auto& [name, shape] : input_shapes)
        shapes[name] = shape;

    // Seed with initializer shapes (weights, biases)
    for (const auto& [name, tensor] : graph.initializers)
        shapes[name] = tensor.shape;

    // Forward pass in topological order
    for (const auto& node : graph.nodes) {
        bool ok = true;
        switch (node.op) {
        case Operator::Im2Col:
            ok = infer_im2col(node, result);
            break;
        case Operator::Gemm:
            ok = infer_gemm(node, result);
            break;
        case Operator::Col2Im:
            ok = infer_col2im(node, result);
            break;
        case Operator::Reshape:
            ok = infer_reshape(node, result);
            break;
        case Operator::ReLU:
        case Operator::Sigmoid:
            ok = infer_elementwise(node, result);
            break;
        case Operator::Conv:
            ok = infer_conv2d(node, result);
            break;

        case Operator::MaxPool2D:
            ok = infer_max_pool_2d(node, result);
            break;
        default:
            result.status = {false, "shape_inference: unhandled op"};
            return result;
        }

        if (!ok) return result;
    }

    return result;
}

} // namespace ml::passes