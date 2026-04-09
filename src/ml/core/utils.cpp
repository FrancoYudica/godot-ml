#include "utils.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

namespace ml {
namespace Utils {

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

    String inputs_str = "Inputs: ";
    for (const auto& input_name : graph.input_names) {
        inputs_str += to_gstring(input_name) + " ";
    }
    UtilityFunctions::print(inputs_str);

    String shape_str = "Input shape: [";
    for (const auto& i : graph.input_shape)
        shape_str += String::num_int64(i) + ",";
    shape_str += "]";
    UtilityFunctions::print(shape_str);

    for (const auto& node : graph.nodes) {
        UtilityFunctions::print("Node: ", node_operator_to_string(node.op));

        String inputs_str = "  inputs: ";
        for (const auto& i : node.inputs)
            inputs_str += to_gstring(i) + " ";
        UtilityFunctions::print(inputs_str);

        String outputs_str = "  outputs: ";
        for (const auto& o : node.outputs)
            outputs_str += to_gstring(o) + " ";
        UtilityFunctions::print(outputs_str);
    }

    UtilityFunctions::print("Initializers:");
    for (const auto& [name, tensor] : graph.initializers) {
        String tensor_str = "  " + to_gstring(name) + ": [";
        for (auto d : tensor.shape)
            tensor_str += String::num_int64(d) + ",";
        tensor_str += "]";
        UtilityFunctions::print(tensor_str);
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
} // namespace Utils
} // namespace ml