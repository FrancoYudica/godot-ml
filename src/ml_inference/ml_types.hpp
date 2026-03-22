#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace ml {

/**
 * Supported operators
 */
enum class NodeOperator : int { Gemm, ReLU, Sigmoid, Unknown };

/**
 * Tensor holds it's shape and weight data
 */
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/**
 * Graph node has an operation and it's tensor inputs and outputs
 */
struct GraphNode {
  NodeOperator op;

  /**
   * Tensor names (activations + weights)
   */
  std::vector<std::string> inputs;

  /**
   * Intermediate output tensor names
   */
  std::vector<std::string> outputs;

  /**
   * GeMM (General Matrix Multiply) attributes
   * output = alpha * (A @ B^T) + beta * C
   *
   * A = input activations, B = weights, C = bias
   * transB = true means B is already stored transposed in the file,
   * i.e. shape is [out_features, in_features] instead of [in_features,
   * out_features]
   */
  float alpha = 1.0f;
  float beta = 1.0f;
  bool transB = false;
};

/**
 * Parsed graph
 */
struct Graph {
  std::string input_name;
  std::vector<int64_t> input_shape;

  /**
   * Nodes to process in topological order
   */
  std::vector<GraphNode> nodes;

  /**
   * Holds the trained weights and biases. The
   * trained tensors
   */
  std::unordered_map<std::string, Tensor> initializers;
};

} // namespace ml