#include "utils.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

namespace ml::Utils {

godot::String get_project_relative_path(
    const godot::String& addon_relative_path) {
    static const godot::String base_path = "res://addons/ml/";
    return base_path + addon_relative_path;
}

godot::String node_operator_to_string(NodeOperator op) {
    static const std::unordered_map<NodeOperator, std::string>
        operator_names = {
            {NodeOperator::Unknown, "Unknown"},
            {NodeOperator::Gemm, "Gemm"},
            {NodeOperator::ReLU, "ReLU"},
            {NodeOperator::Conv2D, "Conv2D"},
            {NodeOperator::Sigmoid, "Sigmoid"}};
    auto it = operator_names.find(op);
    if (it != operator_names.end()) {
        return godot::String(it->second.c_str());
    }
    return godot::String("Unknown");
}

const std::vector<ml::NodeOperator>& get_node_operators() {
    static std::vector<ml::NodeOperator> operators = {
        NodeOperator::Gemm,
        NodeOperator::ReLU,
        NodeOperator::Sigmoid};

    return operators;
}

RID load_shader(RenderingDevice* rd, const godot::String& path) {
    Ref<RDShaderFile> shader_file =
        ResourceLoader::get_singleton()->load(path);

    ERR_FAIL_COND_V_MSG(
        shader_file.is_null(),
        RID(),
        "Failed to load shader file: " + path);

    Ref<RDShaderSPIRV> spirv = shader_file->get_spirv();

    ERR_FAIL_COND_V_MSG(
        spirv.is_null(),
        RID(),
        "Failed to load SPIR-V from shader file: " + path);

    return rd->shader_create_from_spirv(spirv);
}

void print(const Graph& graph) {
    auto to_gstring = [](const std::string& s) {
        return String(s.c_str());
    };

    UtilityFunctions::print("Input names: " + get_iterator_str(graph.input_names.begin(), graph.input_names.end()));
    UtilityFunctions::print("Input shape: " + get_iterator_str(graph.input_shape.begin(), graph.input_shape.end()));

    for (const auto& node : graph.nodes) {
        UtilityFunctions::print("Node: ", node_operator_to_string(node.op));
        UtilityFunctions::print(" inputs: ", get_iterator_str(node.inputs.begin(), node.inputs.end()));
        UtilityFunctions::print(" outputs: ", get_iterator_str(node.outputs.begin(), node.outputs.end()));

        std::visit([&](const auto& attr) {
            UtilityFunctions::print("  attributes:");

            // Get the type of the current attribute set
            using T = std::decay_t<decltype(attr)>;

            if constexpr (std::is_same_v<T, GemmAttributes>) {
                UtilityFunctions::print("    Type: Gemm");
                UtilityFunctions::print("    alpha: ", String::num_real(attr.alpha));
                UtilityFunctions::print("    beta: ", String::num_real(attr.beta));
                UtilityFunctions::print("    transB: ", attr.transB ? "true" : "false");
            } else if constexpr (std::is_same_v<T, ConvAttributes>) {
                UtilityFunctions::print("    Type: Conv2D");
                UtilityFunctions::print("    kernel_shape: ", get_iterator_str(attr.kernel_shape.begin(), attr.kernel_shape.end()));
                UtilityFunctions::print("    pads: ", get_iterator_str(attr.pads.begin(), attr.pads.end()));
                UtilityFunctions::print("    strides: ", get_iterator_str(attr.strides.begin(), attr.strides.end()));
            } else {
                UtilityFunctions::print("    No specific attributes for this node type.");
            }
        },
                   node.attributes);

        UtilityFunctions::print("Initializers:");
        for (const auto& [name, tensor] : graph.initializers) {
            UtilityFunctions::print(" " + to_gstring(name) + ": ", get_iterator_str(tensor.shape.begin(), tensor.shape.end()));
        }
    }
}
bool tensor_shape_matches(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}
String shape_to_str(const std::vector<int64_t>& shape) {
    String str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        str += String::num_int64(shape[i]);
        if (i < shape.size() - 1) {
            str += ",";
        }
    }
    str += "]";
    return str;
}
} // namespace ml::Utils