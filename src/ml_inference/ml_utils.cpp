#include "ml_utils.hpp"
#include <iostream>

void ml::Utils::print(const Graph &graph) {
  static const std::unordered_map<NodeOperator, std::string> operator_names = {
      {NodeOperator::Unknown, "Unknown"},
      {NodeOperator::Gemm, "Gemm"},
      {NodeOperator::ReLU, "ReLU"},
      {NodeOperator::Sigmoid, "Sigmoid"}};

  std::cout << "Input: " << graph.input_name << "\n";
  std::cout << "Input shape: [";

  for (const auto &i : graph.input_shape)
    std::cout << i << ",";
  std::cout << "]\n";

  for (const auto &node : graph.nodes) {
    std::cout << "Node: " << operator_names.at(node.op) << "\n";
    std::cout << "  inputs: ";
    for (const auto &i : node.inputs)
      std::cout << i << " ";
    std::cout << "\n";

    std::cout << "  outputs: ";
    for (const auto &o : node.outputs)
      std::cout << o << " ";
    std::cout << "\n";
  }

  std::cout << "Initializers:\n";
  for (const auto &[name, tensor] : graph.initializers) {
    std::cout << "  " << name << ": [";
    for (auto d : tensor.shape)
      std::cout << d << ",";
    std::cout << "]\n";
  }
}