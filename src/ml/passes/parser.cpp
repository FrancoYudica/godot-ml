#include "parser.hpp"

#include "onnx/onnx_pb.h"

#include <fstream>
#include <godot_cpp/classes/project_settings.hpp>

namespace ml {

static const std::unordered_map<std::string, LogicalOp> operator_names = {
    {"Relu", LogicalOp::ReLU},
    {"Sigmoid", LogicalOp::Sigmoid},
    {"Gemm", LogicalOp::Gemm},
    {"Conv", LogicalOp::Conv},
    {"ConvTranspose", LogicalOp::ConvTranspose},
    {"Im2Col", LogicalOp::Im2Col},
    {"Unknown", LogicalOp::Unknown}};

static void _parse_inputs(const onnx::GraphProto& proto, LogicalGraph& graph) {
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

static void _parse_initializers(const onnx::GraphProto& proto, LogicalGraph& graph) {
    for (const auto& tensor : proto.initializer()) {
        Tensor t;
        t.name = tensor.name();
        for (auto dim : tensor.dims())
            t.shape.push_back(dim);
        if (tensor.float_data_size() > 0) {
            t.data.assign(tensor.float_data().begin(), tensor.float_data().end());
        } else if (!tensor.raw_data().empty()) {
            const float* raw = reinterpret_cast<const float*>(tensor.raw_data().data());
            t.data.assign(raw, raw + tensor.raw_data().size() / sizeof(float));
        }
        graph.initializers[t.name] = std::move(t);
    }
}

static OperationResult _parse_nodes(const onnx::GraphProto& proto, LogicalGraph& graph) {
    for (const auto& node : proto.node()) {
        if (operator_names.find(node.op_type()) == operator_names.end())
            return {false, "unsupported operator '" + node.op_type() + "'"};

        LogicalOp operator_type = operator_names.at(node.op_type());

        LogicalNode n;
        for (const auto& inp : node.input())  n.inputs.push_back(inp);
        for (const auto& out : node.output()) n.outputs.push_back(out);

        if (operator_type == LogicalOp::ReLU) {
            n.op = LogicalOp::ReLU;
        }

        else if (operator_type == LogicalOp::Sigmoid) {
            n.op = LogicalOp::Sigmoid;
        }

        else if (operator_type == LogicalOp::Gemm) {
            n.attributes.emplace<GemmAttributes>();
            auto& gemm = std::get<GemmAttributes>(n.attributes);
            for (const auto& attr : node.attribute()) {
                if (attr.name() == "alpha")  gemm.alpha  = attr.f();
                if (attr.name() == "beta")   gemm.beta   = attr.f();
                if (attr.name() == "transB") gemm.transB = (attr.i() == 1);
            }
            if (!gemm.transB)
                return {false, "Gemm node '" + node.name() + "': only transB=1 is supported"};
            n.op = LogicalOp::Gemm;
        }

        else if (operator_type == LogicalOp::Conv || operator_type == LogicalOp::Im2Col) {
            n.attributes.emplace<ConvAttributes>();
            auto& conv = std::get<ConvAttributes>(n.attributes);
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

            auto vr = conv.validate();
            if (!vr.success)
                return {false, "Conv/Im2Col node '" + node.name() + "': " + vr.error};
            n.op = operator_type;
        }

        else if (operator_type == LogicalOp::ConvTranspose) {
            n.attributes.emplace<ConvTransposeAttributes>();
            auto& conv = std::get<ConvTransposeAttributes>(n.attributes);
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

            auto vr = conv.validate();
            if (!vr.success)
                return {false, "ConvTranspose node '" + node.name() + "': " + vr.error};
            n.op = LogicalOp::ConvTranspose;
        }

        graph.nodes.push_back(std::move(n));
    }
    return {true, {}};
}

namespace passes {

ParseResult parse(const std::string& path) {
    godot::String global_path =
        godot::ProjectSettings::get_singleton()->globalize_path(
            godot::String(path.c_str()));
    std::string absolute_path = global_path.utf8().get_data();

    onnx::ModelProto model;
    std::ifstream file(absolute_path, std::ios::binary);
    if (!file.is_open())
        return {{}, {false, "could not open file: " + absolute_path}};
    if (!model.ParseFromIstream(&file))
        return {{}, {false, "failed to deserialize ONNX protobuf: " + absolute_path}};

    const onnx::GraphProto& proto = model.graph();
    LogicalGraph graph;
    _parse_inputs(proto, graph);
    _parse_initializers(proto, graph);

    auto nodes_result = _parse_nodes(proto, graph);
    if (!nodes_result.success)
        return {{}, nodes_result};

    return {std::move(graph), {true, {}}};
}

} // namespace passes
} // namespace ml
