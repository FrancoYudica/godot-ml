#pragma once
#include "core_types.hpp"

namespace ml {

namespace Logical {

enum class Operator {
    Gemm,
    ReLU,
    Sigmoid,
    Conv,
    Im2Col,
    ConvTranspose,
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
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
};

struct Col2ImAttrs {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> output_padding;
    std::vector<int64_t> dilations;
};

struct ConvTransposeAttrs {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> output_padding;
    std::vector<int64_t> dilations;
};

struct MaxPool2DAttrs {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
};

// LOGICAL NODE AND GRAPH. MAPS ONNX FORMAT 1:1

struct Node {
    Logical::Operator op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::variant<
        std::monostate,
        GemmAttrs,
        ConvAttrs,
        ConvTransposeAttrs,
        MaxPool2DAttrs>
        attributes;
};

struct Graph {
    std::vector<std::string> input_names;
    std::vector<int64_t> input_shape;
    std::vector<Node> nodes;
    std::unordered_map<std::string, Tensor> initializers;
};

} // namespace Logical
} // namespace ml
