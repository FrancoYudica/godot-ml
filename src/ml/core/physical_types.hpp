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
    /**
     * GeMM (General Matrix Multiply) attributes
     * output = alpha * (A @ B^T) + beta * C
     *
     * A = input activations, B = weights, C = bias
     * transB = true means B is stored as [out_features, in_features],
     * which is what the shader always assumes.
     */
    float alpha = 1.0f;
    float beta = 1.0f;
    bool transB = false;
};

struct ConvAttrs {
    uint32_t kernel_w;
    uint32_t kernel_h;
    uint32_t padding_left;
    uint32_t padding_top;
    uint32_t padding_right;
    uint32_t padding_bottom;
    uint32_t stride_x;
    uint32_t stride_y;
    uint32_t dilation_x;
    uint32_t dilation_y;
};

struct Col2ImAttrs {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> output_padding;
    std::vector<int64_t> dilations;

    std::string source_activation;
};

struct MaxPool2DAttrs {
    uint32_t kernel_w;
    uint32_t kernel_h;
    uint32_t padding_left;
    uint32_t padding_top;
    uint32_t padding_right;
    uint32_t padding_bottom;
    uint32_t stride_x;
    uint32_t stride_y;
    uint32_t dilation_x;
    uint32_t dilation_y;
};

enum class ReshapeMode {
    // Flattens [b, c, h, w] -> [b*h*w, c]  (before Gemm in ConvTranspose)
    ImageToGemm,
    // Restores [b*h*w, oc] -> [b, oc, oh, ow]  (after Col2Im / after GEMM in Conv)
    // Reads the 4D target shape from ShapeTable using image_shape_ref.
    GemmToImage,
};

struct ReshapeAttrs {
    ReshapeMode mode;
    // For GemmToImage: key in ShapeTable that holds the target [b, c, h, w].
    // Written by Im2Col (Conv path) or Col2Im (ConvTranspose path) during shape inference.
    // Format: <producer_output_name> + "__4d"
    std::string image_shape_ref;
};

/**
 * Physical::Graph node has an operation and its tensor inputs and outputs.
 * Shapes are not stored here - they are computed per-inference by the
 * shape inference pass and stored in a ShapeTable.
 */
struct Node {
    Operator op;

    /**
     * Tensor names (activations + weights)
     */
    std::vector<std::string> inputs;

    /**
     * Intermediate output tensor names
     */
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

/**
 * Parsed graph
 */
struct Graph {
    std::vector<std::string> input_names;

    /**
     * Nodes to process in topological order
     */
    std::vector<Node> nodes;

    /**
     * Holds the trained weights and biases.
     */
    std::unordered_map<std::string, Tensor> initializers;
};
} // namespace Physical
} // namespace ml
