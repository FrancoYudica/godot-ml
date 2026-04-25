#pragma once
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/string.hpp>
#include <memory> // IWYU pragma: export
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ml {

// Maps every tensor name (inputs, initializers, intermediates) to its concrete shape.
// Populated by the shape inference pass before each inference call.
using ShapeTable = std::unordered_map<std::string, std::vector<int64_t>>;

enum class LogicalOp {
    Gemm,
    ReLU,
    Sigmoid,
    Conv,
    Im2Col,
    ConvTranspose,
    Unknown
};

enum class PhysicalOp {
    Gemm,
    ReLU,
    Sigmoid,
    Conv,
    Im2Col,
    Col2Im,
    Reshape,
    Unknown
};

struct OperationResult {
    bool success = true;
    std::string error;
};

struct GemmAttributes {
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

    OperationResult validate() const {
        if (!transB) {
            return {false, "Unsupported transB value `false`"};
        }
        return {true, {}};
    }
};

struct ConvAttributes {

    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;

    OperationResult validate() const {
        if (kernel_shape.size() != 2)
            return {
                false,
                "only 2D convolutions supported, got kernel_shape dims: " + std::to_string(kernel_shape.size())};
        if (kernel_shape[0] <= 0 || kernel_shape[1] <= 0)
            return {
                false,
                "kernel dimensions must be positive"};
        if (pads.size() != 4)
            return {
                false,
                "expected 4 pad values, got: " + std::to_string(pads.size())};
        if (strides.size() != 2)
            return {
                false,
                "expected 2 stride values, got: " + std::to_string(strides.size())};

        return {true, {}};
    }
};

struct Col2ImAttributes {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> output_padding;
    // Name of the original ConvTranspose input tensor [b, ic, ih, iw].
    // Used by shape inference to compute output spatial dimensions.
    std::string source_activation;

    OperationResult validate() const {
        if (kernel_shape.size() != 2)
            return {
                false,
                "only 2D convolutions supported, got kernel_shape dims: " + std::to_string(kernel_shape.size())};
        if (kernel_shape[0] <= 0 || kernel_shape[1] <= 0)
            return {
                false,
                "kernel dimensions must be positive"};
        if (pads.size() != 4)
            return {
                false,
                "expected 4 pad values, got: " + std::to_string(pads.size())};
        if (strides.size() != 2)
            return {
                false,
                "expected 2 stride values, got: " + std::to_string(strides.size())};

        if (output_padding.size() != 2)
            return {
                false,
                "expected 2 output padding values, got: " + std::to_string(output_padding.size())};
        if (source_activation.empty())
            return {
                false,
                "source activation name cannot be empty"};
        return {true, {}};
    }
};

enum class ReshapeMode {
    // Flattens [b, c, h, w] -> [b*h*w, c]  (before Gemm in ConvTranspose)
    ImageToGemm,
    // Restores [b*h*w, oc] -> [b, oc, oh, ow]  (after Col2Im / after GEMM in Conv)
    // Reads the 4D target shape from ShapeTable using image_shape_ref.
    GemmToImage,
};

struct ReshapeAttributes {
    ReshapeMode mode;
    // For GemmToImage: key in ShapeTable that holds the target [b, c, h, w].
    // Written by Im2Col (Conv path) or Col2Im (ConvTranspose path) during shape inference.
    // Format: <producer_output_name> + "__4d"
    std::string image_shape_ref;

    OperationResult validate() const {
        return {true, {}};
    }
};

struct ConvTransposeAttributes {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> output_padding;

    OperationResult validate() const {
        if (kernel_shape.size() != 2)
            return {
                false,
                "only 2D convolutions supported, got kernel_shape dims: " + std::to_string(kernel_shape.size())};
        if (kernel_shape[0] <= 0 || kernel_shape[1] <= 0)
            return {
                false,
                "kernel dimensions must be positive"};
        if (pads.size() != 4)
            return {
                false,
                "expected 4 pad values, got: " + std::to_string(pads.size())};
        if (strides.size() != 2)
            return {
                false,
                "expected 2 stride values, got: " + std::to_string(strides.size())};
        if (output_padding.size() != 2)
            return {
                false,
                "expected 2 output padding values, got: " + std::to_string(output_padding.size())};
        return {true, {}};
    }
};

/**
 * Tensor holds its shape and weight data
 */
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

// LOGICAL NODE AND GRAPH. MAPS ONNX FORMAT 1:1

struct LogicalNode {
    LogicalOp op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::variant<
        std::monostate,
        GemmAttributes,
        ConvAttributes,
        ConvTransposeAttributes>
        attributes;
};

struct LogicalGraph {
    std::vector<std::string> input_names;
    std::vector<int64_t> input_shape;
    std::vector<LogicalNode> nodes;
    std::unordered_map<std::string, Tensor> initializers;
};

// PHYSICAL NODE AND GRAPH. OBTAINED AFTER LOWERING AND OPTIMIZATION

/**
 * PhysicalGraph node has an operation and its tensor inputs and outputs.
 * Shapes are not stored here — they are computed per-inference by the
 * shape inference pass and stored in a ShapeTable.
 */
struct PhysicalNode {
    PhysicalOp op;

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
        GemmAttributes,
        ConvAttributes,
        Col2ImAttributes,
        ReshapeAttributes>
        attributes;
};

/**
 * Parsed graph
 */
struct PhysicalGraph {
    std::vector<std::string> input_names;
    std::vector<int64_t> input_shape;

    /**
     * Nodes to process in topological order
     */
    std::vector<PhysicalNode> nodes;

    /**
     * Holds the trained weights and biases.
     */
    std::unordered_map<std::string, Tensor> initializers;
};

} // namespace ml
