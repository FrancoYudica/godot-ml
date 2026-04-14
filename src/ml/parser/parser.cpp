#include "parser.hpp"

#include "onnx/onnx_pb.h"

#include <fstream>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/print_string.hpp>

namespace ml {

enum class ParserOperatorType {
    ReLu,
    Sigmoid,
    Gemm,
    Conv,
    Unknown
};

static const std::unordered_map<std::string, ParserOperatorType> operator_names = {
    {"Relu", ParserOperatorType::ReLu},
    {"Sigmoid", ParserOperatorType::Sigmoid},
    {"Gemm", ParserOperatorType::Gemm},
    {"Conv", ParserOperatorType::Conv},
    {"Unknown", ParserOperatorType::Unknown}};

// ── Parser
// ───────────────────────────────────────────────────────────────────
static void _parse_inputs(const onnx::GraphProto& proto, Graph& graph) {
    // First input that is NOT an initializer is the actual graph input
    const auto& init = proto.initializer();
    std::unordered_set<std::string> initializer_names;
    for (const auto& t : init) {
        initializer_names.insert(t.name());
    }

    for (const auto& input : proto.input()) {
        if (initializer_names.count(input.name())) continue;

        graph.input_names.push_back(input.name());
        for (const auto& dim : input.type().tensor_type().shape().dim()) {
            graph.input_shape.push_back(dim.dim_value());
        }
    }
}

static void _parse_initializers(const onnx::GraphProto& proto, Graph& graph) {
    for (const auto& tensor : proto.initializer()) {
        Tensor t;
        t.name = tensor.name();
        for (auto dim : tensor.dims()) {
            t.shape.push_back(dim);
        }
        // Copy float data. raw_data is used when data isn't in float_data
        if (tensor.float_data_size() > 0) {
            t.data.assign(tensor.float_data().begin(), tensor.float_data().end());
        } else if (!tensor.raw_data().empty()) {
            const float* raw =
                reinterpret_cast<const float*>(tensor.raw_data().data());
            t.data.assign(raw, raw + tensor.raw_data().size() / sizeof(float));
        }
        graph.initializers[t.name] = std::move(t);
    }
}

static bool _parse_nodes(const onnx::GraphProto& proto, Graph& graph) {
    for (const auto& node : proto.node()) {
        GraphNode n;

        ERR_FAIL_COND_V_MSG(
            operator_names.find(node.op_type()) == operator_names.end(),
            false,
            "Parser: unsupported operator: " + godot::String(node.op_type().c_str()));

        ParserOperatorType operator_type = operator_names.at(node.op_type());

        for (const auto& inp : node.input())
            n.inputs.push_back(inp);
        for (const auto& out : node.output())
            n.outputs.push_back(out);

        if (operator_type == ParserOperatorType::ReLu) {
            n.op = ml::NodeOperator::ReLU;
        }

        else if (operator_type == ParserOperatorType::Sigmoid) {
            n.op = ml::NodeOperator::Sigmoid;
        }

        //  Parsing Gemm
        else if (operator_type == ParserOperatorType::Gemm) {
            // Initialize the variant to GemmAttributes once
            n.attributes.emplace<GemmAttributes>();
            auto& gemm = std::get<GemmAttributes>(n.attributes);

            for (const auto& attr : node.attribute()) {
                if (attr.name() == "alpha") gemm.alpha = attr.f();
                if (attr.name() == "beta") gemm.beta = attr.f();
                if (attr.name() == "transB") gemm.transB = (attr.i() == 1);
            }

            ERR_FAIL_COND_V_MSG(
                !gemm.transB,
                false,
                "Parser: only transB = 1 is supported for GEMM.");

            n.op = ml::NodeOperator::Gemm;
        }

        // Parsing Conv (currently any shape)
        else if (operator_type == ParserOperatorType::Conv) {
            // Initialize the variant to ConvAttributes once
            n.attributes.emplace<ConvAttributes>();
            auto& conv = std::get<ConvAttributes>(n.attributes);

            for (const auto& attr : node.attribute()) {
                if (attr.name() == "kernel_shape") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        conv.kernel_shape.push_back(attr.ints(i));
                    }
                }
                if (attr.name() == "pads") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        conv.pads.push_back(attr.ints(i));
                    }
                }

                if (attr.name() == "strides") {
                    for (int i = 0; i < attr.ints_size(); ++i) {
                        conv.strides.push_back(attr.ints(i));
                    }
                }
            }

            // Fallback: derive kernel_shape from weight tensor if attribute was absent
            // Weight tensor layout: [out_channels, in_channels, kH, kW] for Conv
            //                       [in_channels, out_channels, kH, kW] for ConvTranspose
            if (conv.kernel_shape.empty() && node.input_size() > 1) {
                const std::string& weight_name = node.input(1);
                auto it = graph.initializers.find(weight_name);
                if (it != graph.initializers.end()) {
                    const auto& tensor = it->second;
                    // dims are [out, in, kH, kW] — kernel shape is everything from dim 2 onward
                    for (int i = 2; i < tensor.shape.size(); ++i)
                        conv.kernel_shape.push_back(tensor.shape[i]);
                }
            }

            // Fallback: strides default to 1 per ONNX spec if absent
            if (conv.strides.empty()) {
                int spatial_dims = conv.kernel_shape.size();
                conv.strides.assign(spatial_dims, 1);
            }

            // Fallback: pads default to 0 per ONNX spec if absent
            if (conv.pads.empty()) {
                int spatial_dims = conv.kernel_shape.size();
                conv.pads.assign(spatial_dims * 2, 0); // [x1_begin, x2_begin, x1_end, x2_end]
            }

            ERR_FAIL_COND_V_MSG(
                conv.kernel_shape.size() != 2,
                false,
                "Parser: only 2D convolutions are supported. Expected kernel size of 2 dimensions. Got " + godot::String::num(conv.kernel_shape.size()));

            ERR_FAIL_COND_V_MSG(
                conv.pads.size() != 4,
                false,
                "Parser: only 2D convolutions are supported. Expected pads of 4 dimensions. Got " + godot::String::num(conv.pads.size()));

            ERR_FAIL_COND_V_MSG(
                conv.strides.size() != 2,
                false,
                "Parser: only 2D convolutions are supported. Expected strides of 2 dimensions. Got " + godot::String::num(conv.strides.size()));

            n.op = NodeOperator::Conv2D;
        }

        graph.nodes.push_back(std::move(n));
    }
    return true;
}

namespace Parser {

bool parse(const std::string& path, Graph& graph) {
    // Convert res:// path to absolute filesystem path
    godot::String godot_path(path.c_str());
    godot::String global_path =
        godot::ProjectSettings::get_singleton()->globalize_path(
            godot_path);
    std::string absolute_path = global_path.utf8().get_data();

    // Load and deserialize the protobuf file
    onnx::ModelProto model;
    std::ifstream file(absolute_path, std::ios::binary);

    ERR_FAIL_COND_V_MSG(
        !file.is_open(),
        false,
        "ONNXParser: could not open file: " + godot::String(absolute_path.c_str()));

    ERR_FAIL_COND_V_MSG(
        !model.ParseFromIstream(&file),
        false,
        "ONNXParser: failed to parse ONNX file: " + godot::String(absolute_path.c_str()));

    const onnx::GraphProto& proto = model.graph();
    _parse_inputs(proto, graph);
    _parse_initializers(proto, graph);

    ERR_FAIL_COND_V_MSG(
        !_parse_nodes(proto, graph),
        false,
        "ONNXParser: failed to parse nodes.");

    return true;
}
} // namespace Parser
} // namespace ml
