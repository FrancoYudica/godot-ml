#include "parser.hpp"

#include "onnx/onnx_pb.h"

#include <fstream>
#include <godot_cpp/classes/project_settings.hpp>

namespace ml {

namespace Logical {
static const std::unordered_map<std::string, Operator> operator_names = {
    {"Relu", Operator::ReLU},
    {"Sigmoid", Operator::Sigmoid},
    {"Gemm", Operator::Gemm},
    {"Conv", Operator::Conv},
    {"ConvTranspose", Operator::ConvTranspose},
    {"Im2Col", Operator::Im2Col},
    {"MaxPool", Operator::MaxPool2D},
    {"Reshape", Operator::Reshape},
    {"Unknown", Operator::Unknown}};

static void _parse_inputs(const onnx::GraphProto& proto, Graph& graph) {
    const auto& init = proto.initializer();
    std::unordered_set<std::string> initializer_names;
    for (const auto& t : init)
        initializer_names.insert(t.name());

    for (const auto& input : proto.input()) {
        if (initializer_names.count(input.name())) continue;
        graph.input_names.push_back(input.name());
        for (const auto& dim : input.type().tensor_type().shape().dim())
            graph.input_shape.push_back(dim.dim_value());
    }
}

static void _parse_initializers(const onnx::GraphProto& proto, Graph& graph) {
    for (const auto& tensor : proto.initializer()) {
        Tensor t;
        t.name = tensor.name();
        for (auto dim : tensor.dims())
            t.shape.push_back(dim);
        if (tensor.float_data_size() > 0) {
            t.data.assign(tensor.float_data().begin(), tensor.float_data().end());
        } else if (tensor.data_type() == onnx::TensorProto::INT64) {
            if (tensor.int64_data_size() > 0) {
                for (auto v : tensor.int64_data())
                    t.data.push_back(static_cast<float>(v));
            } else if (!tensor.raw_data().empty()) {
                const int64_t* raw = reinterpret_cast<const int64_t*>(tensor.raw_data().data());
                size_t count = tensor.raw_data().size() / sizeof(int64_t);
                for (size_t i = 0; i < count; ++i)
                    t.data.push_back(static_cast<float>(raw[i]));
            }
        } else if (!tensor.raw_data().empty()) {
            const float* raw = reinterpret_cast<const float*>(tensor.raw_data().data());
            t.data.assign(raw, raw + tensor.raw_data().size() / sizeof(float));
        }
        graph.initializers[t.name] = std::move(t);
    }
}

static OperationResult _parse_nodes(const onnx::GraphProto& proto, Graph& graph) {
    for (const auto& node : proto.node()) {
        if (operator_names.find(node.op_type()) == operator_names.end())
            return {false, "unsupported operator '" + node.op_type() + "'"};

        Operator operator_type = operator_names.at(node.op_type());

        Node n;
        for (const auto& inp : node.input())
            n.inputs.push_back(inp);
        for (const auto& out : node.output())
            n.outputs.push_back(out);

        if (operator_type == Operator::ReLU) {
            n.op = Operator::ReLU;
        }

        else if (operator_type == Operator::Sigmoid) {
            n.op = Operator::Sigmoid;
        }

        else if (operator_type == Operator::Gemm) {
            n.attributes.emplace<GemmAttrs>();
            auto& gemm = std::get<GemmAttrs>(n.attributes);
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "alpha") gemm.alpha = attr.f();
                if (attr.name() == "beta") gemm.beta = attr.f();
                if (attr.name() == "transB") gemm.transB = (attr.i() == 1);
            }
            n.op = Operator::Gemm;
        }

        else if (operator_type == Operator::Conv || operator_type == Operator::Im2Col) {
            n.attributes.emplace<ConvAttrs>();
            auto& conv = std::get<ConvAttrs>(n.attributes);
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "kernel_shape")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.kernel_shape.push_back(attr.ints(i));
                if (attr.name() == "pads")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.pads.push_back(attr.ints(i));
                if (attr.name() == "strides")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.strides.push_back(attr.ints(i));
                if (attr.name() == "dilations")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.dilations.push_back(attr.ints(i));
            }
            // Fallback: derive kernel_shape from weight tensor if attribute was absent
            if (conv.kernel_shape.empty() && node.input_size() > 1) {
                auto it = graph.initializers.find(node.input(1));
                if (it != graph.initializers.end())
                    for (size_t i = 2; i < it->second.shape.size(); ++i)
                        conv.kernel_shape.push_back(it->second.shape[i]);
            }
            if (conv.strides.empty())
                conv.strides.assign(conv.kernel_shape.size(), 1);
            if (conv.pads.empty())
                conv.pads.assign(conv.kernel_shape.size() * 2, 0);
            if (conv.dilations.empty())
                conv.dilations.assign(conv.kernel_shape.size(), 1);
            n.op = operator_type;
        }

        else if (operator_type == Operator::ConvTranspose) {
            n.attributes.emplace<ConvTransposeAttrs>();
            auto& conv = std::get<ConvTransposeAttrs>(n.attributes);
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "kernel_shape")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.kernel_shape.push_back(attr.ints(i));
                if (attr.name() == "pads")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.pads.push_back(attr.ints(i));
                if (attr.name() == "strides")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.strides.push_back(attr.ints(i));
                if (attr.name() == "output_padding")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.output_padding.push_back(attr.ints(i));
                if (attr.name() == "dilations")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        conv.dilations.push_back(attr.ints(i));
            }
            if (conv.kernel_shape.empty() && node.input_size() > 1) {
                auto it = graph.initializers.find(node.input(1));
                if (it != graph.initializers.end())
                    for (size_t i = 2; i < it->second.shape.size(); ++i)
                        conv.kernel_shape.push_back(it->second.shape[i]);
            }
            if (conv.strides.empty())
                conv.strides.assign(conv.kernel_shape.size(), 1);
            if (conv.pads.empty())
                conv.pads.assign(conv.kernel_shape.size() * 2, 0);
            if (conv.output_padding.empty())
                conv.output_padding.assign(conv.kernel_shape.size(), 0);
            if (conv.dilations.empty())
                conv.dilations.assign(conv.kernel_shape.size(), 1);

            n.op = Operator::ConvTranspose;
        }

        else if (operator_type == Operator::Reshape) {
            n.op = Operator::Reshape;
            // No attributes - shape comes from n.inputs[1] (the shape tensor initializer).
        }

        else if (operator_type == Operator::MaxPool2D) {
            n.attributes.emplace<MaxPool2DAttrs>();
            auto& max_pool = std::get<MaxPool2DAttrs>(n.attributes);
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "kernel_shape")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        max_pool.kernel_shape.push_back(attr.ints(i));
                else if (attr.name() == "pads")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        max_pool.pads.push_back(attr.ints(i));
                else if (attr.name() == "strides")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        max_pool.strides.push_back(attr.ints(i));
                else if (attr.name() == "dilations")
                    for (int i = 0; i < attr.ints_size(); ++i)
                        max_pool.dilations.push_back(attr.ints(i));
                else {
                    godot::UtilityFunctions::push_warning("Unhandled attribute: " + godot::String(attr.name().c_str()) + " for operator: " + godot::String(node.name().c_str()));
                }
            }
            if (max_pool.strides.empty())
                max_pool.strides.assign(max_pool.kernel_shape.size(), 1);
            if (max_pool.pads.empty())
                max_pool.pads.assign(max_pool.kernel_shape.size() * 2, 0);
            if (max_pool.dilations.empty())
                max_pool.dilations.assign(max_pool.kernel_shape.size(), 1);

            n.op = operator_type;
        }

        graph.nodes.push_back(std::move(n));
    }
    return {true, {}};
}
} // namespace Logical
namespace passes {

static ParseResult _parse_model(onnx::ModelProto& model) {
    const onnx::GraphProto& proto = model.graph();
    Logical::Graph graph;
    _parse_inputs(proto, graph);
    _parse_initializers(proto, graph);

    auto nodes_result = _parse_nodes(proto, graph);
    if (!nodes_result.success)
        return {{}, nodes_result};

    return {std::move(graph), {true, {}}};
}

ParseResult parse(const uint8_t* data, size_t size) {
    onnx::ModelProto model;
    if (!model.ParseFromArray(data, static_cast<int>(size)))
        return {{}, {false, "failed to deserialize ONNX protobuf from bytes"}};

    return _parse_model(model);
}

} // namespace passes
} // namespace ml
