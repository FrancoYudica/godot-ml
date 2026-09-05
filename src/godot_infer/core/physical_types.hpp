#pragma once
#include "core_types.hpp"

namespace ml {

namespace Physical {

enum class Operator {
    Gemm,
    ReLU,
    Sigmoid,
    Conv,
    Im2Col,
    Col2Im,
    MaxPool2D,
    Reshape,
    Unknown
};

struct GemmAttrs {
    float alpha = 1.0f;
    float beta = 1.0f;
};

struct ConvAttrs {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

struct Col2ImAttrs {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t output_padding_y;
    uint32_t output_padding_x;
    std::string source_activation;
};

struct MaxPool2DAttrs {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
};

enum class ReshapeMode {
    // Flattens [b, c, h, w] -> [b*h*w, c]  (before Gemm in ConvTranspose)
    ImageToGemm,
    // Restores [b*h*w, oc] -> [b, oc, oh, ow]  (after Col2Im / after GEMM in Conv)
    // Reads the 4D target shape from ShapeTable using image_shape_ref.
    GemmToImage,
    // Standard ONNX Reshape: explicit target_shape from the shape initializer.
    // -1 dimensions are resolved at shape inference time.
    Standard,
};

struct ReshapeAttrs {
    ReshapeMode mode;
    // For GemmToImage: key in ShapeTable that holds the target [b, c, h, w].
    // Written by Im2Col (Conv path) or Col2Im (ConvTranspose path) during shape inference.
    // Format: <producer_output_name> + "__4d"
    std::string image_shape_ref;
    // When true: dispatches matrix_transpose.glsl instead of creating a zero-copy alias.
    // Used for BCHW bridge nodes around GEMM in Conv (post-GEMM) and ConvTranspose (pre-GEMM).
    bool is_permutation = false;
    // For Standard mode: target dimensions read from the ONNX shape initializer.
    // May contain -1 (infer) or 0 (copy from input dim).
    std::vector<int64_t> target_shape;
};

struct Node {
    Operator op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    std::variant<
        std::monostate,
        GemmAttrs,
        ConvAttrs,
        Col2ImAttrs,
        ReshapeAttrs,
        MaxPool2DAttrs>
        attributes;
};

struct Graph {
    std::vector<std::string> input_names;
    std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Tensor> initializers;
};
} // namespace Physical
} // namespace ml
