#include "ml_parser.hpp"
#include "ml_types.hpp"
#include "onnx/onnx_pb.h"
#include <fstream>

namespace ml {

static NodeOperator parse_operator(const std::string &op_type) {
  if (op_type == "Gemm")
    return NodeOperator::Gemm;
  if (op_type == "Relu")
    return NodeOperator::ReLU;
  if (op_type == "Sigmoid")
    return NodeOperator::Sigmoid;
  return NodeOperator::Unknown;
}

// ── Parser ───────────────────────────────────────────────────────────────────
static void _parse_inputs(const onnx::GraphProto &proto, Graph &graph) {
  // First input that is NOT an initializer is the actual graph input
  const auto &init = proto.initializer();
  std::unordered_set<std::string> initializer_names;
  for (const auto &t : init) {
    initializer_names.insert(t.name());
  }

  for (const auto &input : proto.input()) {
    if (initializer_names.count(input.name()))
      continue;
    graph.input_name = input.name();
    for (const auto &dim : input.type().tensor_type().shape().dim()) {
      graph.input_shape.push_back(dim.dim_value());
    }
    break; // only one input for this model
  }
}

static void _parse_initializers(const onnx::GraphProto &proto, Graph &graph) {
  for (const auto &tensor : proto.initializer()) {
    Tensor t;
    t.name = tensor.name();
    for (auto dim : tensor.dims()) {
      t.shape.push_back(dim);
    }
    // Copy float data. raw_data is used when data isn't in float_data
    if (tensor.float_data_size() > 0) {
      t.data.assign(tensor.float_data().begin(), tensor.float_data().end());
    } else if (!tensor.raw_data().empty()) {
      const float *raw =
          reinterpret_cast<const float *>(tensor.raw_data().data());
      t.data.assign(raw, raw + tensor.raw_data().size() / sizeof(float));
    }
    graph.initializers[t.name] = std::move(t);
  }
}

static void _parse_nodes(const onnx::GraphProto &proto, Graph &graph) {
  for (const auto &node : proto.node()) {
    GraphNode n;
    n.op = parse_operator(node.op_type());

    if (n.op == NodeOperator::Unknown) {
      throw std::runtime_error("Parser: unsupported operator: " +
                               node.op_type());
    }

    for (const auto &inp : node.input())
      n.inputs.push_back(inp);
    for (const auto &out : node.output())
      n.outputs.push_back(out);

    // Parse Gemm-specific attributes
    if (n.op == NodeOperator::Gemm) {
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "alpha")
          n.alpha = attr.f();
        if (attr.name() == "beta")
          n.beta = attr.f();
        if (attr.name() == "transB")
          n.transB = attr.i() == 1;
      }
    }

    graph.nodes.push_back(std::move(n));
  }
}

namespace Parser {

Graph parse(const std::string &path) {
  // Load and deserialize the protobuf file
  onnx::ModelProto model;
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("ONNXParser: could not open file: " + path);
  }
  if (!model.ParseFromIstream(&file)) {
    throw std::runtime_error("ONNXParser: failed to parse ONNX file: " + path);
  }

  const onnx::GraphProto &proto = model.graph();
  Graph graph;

  _parse_inputs(proto, graph);
  _parse_initializers(proto, graph);
  _parse_nodes(proto, graph);

  return graph;
}
} // namespace Parser
} // namespace ml
