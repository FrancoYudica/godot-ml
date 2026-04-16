#include "parser.hpp"

#include "onnx/onnx_pb.h"

#include <fstream>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/print_string.hpp>

namespace ml {

static const std::unordered_map<std::string, LogicalOp> operator_names = {
    {"Relu", LogicalOp::ReLU},
    {"Sigmoid", LogicalOp::Sigmoid},
    {"Gemm", LogicalOp::Gemm},
    {"Conv", LogicalOp::Conv},
    {"Im2Col", LogicalOp::Im2Col},
    {"Unknown", LogicalOp::Unknown}};

// ── Parser
// ───────────────────────────────────────────────────────────────────
static void _parse_inputs(const onnx::GraphProto& proto, LogicalGraph& graph) {
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

static void _parse_initializers(const onnx::GraphProto& proto, LogicalGraph& graph) {
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

static bool _parse_nodes(const onnx::GraphProto& proto, LogicalGraph& graph) {
    for (const auto& node : proto.node()) {
        LogicalNode n;

        ERR_FAIL_COND_V_MSG(
            operator_names.find(node.op_type()) == operator_names.end(),
            false,
            "Parser: unsupported operator: " + godot::String(node.op_type().c_str()));

        LogicalOp operator_type = operator_names.at(node.op_type());

        for (const auto& inp : node.input())
            n.inputs.push_back(inp);
        for (const auto& out : node.output())
            n.outputs.push_back(out);

        if (operator_type == LogicalOp::ReLU) {
            n.op = LogicalOp::ReLU;
        }

        else if (operator_type == LogicalOp::Sigmoid) {
            n.op = LogicalOp::Sigmoid;
        }

        //  Parsing Gemm
        else if (operator_type == LogicalOp::Gemm) {
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

            n.op = LogicalOp::Gemm;
        }

        // Parsing Conv (currently any shape)
        else if (operator_type == LogicalOp::Conv || operator_type == LogicalOp::Im2Col) {
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
            if (operator_type == LogicalOp::Im2Col) {
                n.op = LogicalOp::Im2Col;
            } else {
                n.op = LogicalOp::Conv;
            }
        }
        graph.nodes.push_back(std::move(n));
    }
    return true;
}

namespace Parser {

bool parse(const std::string& path, LogicalGraph& graph) {
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
