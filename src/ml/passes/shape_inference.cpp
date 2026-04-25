#include "shape_inference.hpp"

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
static bool infer_im2col(const PhysicalNode& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    if (!in) return false;

    const auto& attrs = std::get<ConvAttributes>(node.attributes);
    int64_t b = (*in)[0], c = (*in)[1], h = (*in)[2], w = (*in)[3];
    int64_t kH = attrs.kernel_shape[0], kW = attrs.kernel_shape[1];
    int64_t out_h = (h + 2 * attrs.pads[0] - kH) / attrs.strides[0] + 1;
    int64_t out_w = (w + 2 * attrs.pads[1] - kW) / attrs.strides[1] + 1;

    shapes[node.outputs[0]] = {out_h * out_w, c * kH * kW};

    // Write 4D meta-shape for the downstream GemmToImage reshape.
    // weights are inputs[1] when lowered from Conv (shape [out_c, c, kH, kW]).
    if (node.inputs.size() > 1) {
        const auto* w = require(node.inputs[1], shapes, result);
        if (!w) return false;
        int64_t out_c = (*w)[0];
        shapes[node.outputs[0] + "__4d"] = {b, out_c, out_h, out_w};
    }

    return true;
}

// Gemm: [M, K] x [N, K]^T -> [M, N]
// Weights are always stored as [N, K] (transB=true).
static bool infer_gemm(const PhysicalNode& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    const auto* w = require(node.inputs[1], shapes, result);
    if (!in || !w) return false;

    int64_t M = (*in)[0];
    int64_t N = (*w)[0]; // [N, K] layout — first dim is output features
    shapes[node.outputs[0]] = {M, N};
    return true;
}

// Col2Im: [b*ih*iw, oc*kH*kW] -> [out_h*out_w, oc]
// Also writes "<output>__4d" = [b, oc, out_h, out_w].
// out_c comes from bias shape (inputs[1] = bias [oc]).
// Spatial output dims come from attrs.source_activation shape [b, ic, ih, iw].
static bool infer_col2im(const PhysicalNode& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto& attrs = std::get<Col2ImAttributes>(node.attributes);

    const auto* src = require(attrs.source_activation, shapes, result);
    const auto* bias = require(node.inputs[1], shapes, result);
    if (!src || !bias) return false;

    int64_t b = (*src)[0];
    int64_t ih = (*src)[2];
    int64_t iw = (*src)[3];
    int64_t kH = attrs.kernel_shape[0];
    int64_t kW = attrs.kernel_shape[1];

    // ConvTranspose output size formula (matches col2im_operator.cpp)
    int64_t out_h = (ih - 1) * attrs.strides[0] - 2 * attrs.pads[0] + kH;
    int64_t out_w = (iw - 1) * attrs.strides[1] - 2 * attrs.pads[1] + kW;

    int64_t out_c = (*bias)[0];

    shapes[node.outputs[0]] = {out_h * out_w, out_c};
    shapes[node.outputs[0] + "__4d"] = {b, out_c, out_h, out_w};
    return true;
}

// Reshape (ImageToGemm): [b, c, h, w] -> [b*h*w, c]
// Reshape (GemmToImage): reads target shape from ShapeTable[attrs.image_shape_ref]
static bool infer_reshape(const PhysicalNode& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto& attrs = std::get<ReshapeAttributes>(node.attributes);

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
static bool infer_elementwise(const PhysicalNode& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;
    const auto* in = require(node.inputs[0], shapes, result);
    if (!in) return false;
    shapes[node.outputs[0]] = *in;
    return true;
}

// Direct Conv2D (not lowered): [b, c, h, w] -> [b, out_c, out_h, out_w]
static bool infer_conv2d(const PhysicalNode& node, ShapeInferenceResult& result) {
    auto& shapes = result.shapes;

    const auto* in = require(node.inputs[0], shapes, result);
    const auto* w = require(node.inputs[1], shapes, result);
    if (!in || !w) return false;

    const auto& attrs = std::get<ConvAttributes>(node.attributes);
    int64_t b = (*in)[0], h = (*in)[2], iw = (*in)[3];
    int64_t out_c = (*w)[0];
    int64_t kH = attrs.kernel_shape[0], kW = attrs.kernel_shape[1];
    int64_t out_h = (h + 2 * attrs.pads[0] - kH) / attrs.strides[0] + 1;
    int64_t out_w = (iw + 2 * attrs.pads[1] - kW) / attrs.strides[1] + 1;

    shapes[node.outputs[0]] = {b, out_c, out_h, out_w};
    return true;
}

ShapeInferenceResult infer_shapes(
    const PhysicalGraph& graph,
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
        case PhysicalOp::Im2Col:
            ok = infer_im2col(node, result);
            break;
        case PhysicalOp::Gemm:
            ok = infer_gemm(node, result);
            break;
        case PhysicalOp::Col2Im:
            ok = infer_col2im(node, result);
            break;
        case PhysicalOp::Reshape:
            ok = infer_reshape(node, result);
            break;
        case PhysicalOp::ReLU:
        case PhysicalOp::Sigmoid:
            ok = infer_elementwise(node, result);
            break;
        case PhysicalOp::Conv:
            ok = infer_conv2d(node, result);
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