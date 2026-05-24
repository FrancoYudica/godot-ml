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

std::string op_name(Physical::Operator op) {
    switch (op) {
    case Physical::Operator::Gemm:
        return "Gemm";
    case Physical::Operator::ReLU:
        return "ReLU";
    case Physical::Operator::Sigmoid:
        return "Sigmoid";
    case Physical::Operator::Conv:
        return "Conv";
    case Physical::Operator::Im2Col:
        return "Im2Col";
    case Physical::Operator::Col2Im:
        return "Col2Im";
    case Physical::Operator::Reshape:
        return "Reshape";
    case Physical::Operator::MaxPool2D:
        return "MaxPool2D";
    default:
        return "Unknown";
    }
}

std::string op_name(Logical::Operator op) {
    switch (op) {
    case Logical::Operator::Gemm:
        return "Gemm";
    case Logical::Operator::ReLU:
        return "ReLU";
    case Logical::Operator::Sigmoid:
        return "Sigmoid";
    case Logical::Operator::Conv:
        return "Conv";
    case Logical::Operator::Im2Col:
        return "Im2Col";
    case Logical::Operator::ConvTranspose:
        return "ConvTranspose";
    case Logical::Operator::MaxPool2D:
        return "MaxPool2D";
    default:
        return "Unknown";
    }
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

void print(const Physical::Graph& graph) {
    auto to_gstring = [](const std::string& s) {
        return String(s.c_str());
    };

    UtilityFunctions::print("Input names: " + get_iterator_str(graph.input_names.begin(), graph.input_names.end()));
    UtilityFunctions::print("Initializers:");
    for (const auto& [name, tensor] : graph.initializers) {
        UtilityFunctions::print(" " + to_gstring(name) + ": ", get_iterator_str(tensor.shape.begin(), tensor.shape.end()));
    }
    for (const auto& node : graph.nodes) {
        UtilityFunctions::print("Node: ", String(Utils::op_name(node.op).c_str()));
        UtilityFunctions::print(" inputs: ", get_iterator_str(node.inputs.begin(), node.inputs.end()));
        UtilityFunctions::print(" outputs: ", get_iterator_str(node.outputs.begin(), node.outputs.end()));

        // std::visit([&](const auto& attr) {
        //     UtilityFunctions::print("  attributes:");

        //     // Get the type of the current attribute set
        //     using T = std::decay_t<decltype(attr)>;

        //     if constexpr (std::is_same_v<T, Physical::GemmAttrs>) {
        //         UtilityFunctions::print("    alpha: ", String::num_real(attr.alpha));
        //         UtilityFunctions::print("    beta: ", String::num_real(attr.beta));
        //         UtilityFunctions::print("    transB: ", attr.transB ? "true" : "false");
        //     } else if constexpr (std::is_same_v<T, Physical::ConvAttrs>) {
        //         UtilityFunctions::print("    kernel_shape: ", get_iterator_str(attr.kernel_shape.begin(), attr.kernel_shape.end()));
        //         UtilityFunctions::print("    pads: ", get_iterator_str(attr.pads.begin(), attr.pads.end()));
        //         UtilityFunctions::print("    strides: ", get_iterator_str(attr.strides.begin(), attr.strides.end()));
        //     } else {
        //         UtilityFunctions::print("    No specific attributes for this node type.");
        //     }
        // },
        //            node.attributes);
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

uint32_t get_tensor_floats(const std::vector<int64_t>& shape) {
    uint32_t floats = 1;
    for (const auto& dim : shape) {
        floats *= static_cast<uint32_t>(dim);
    }
    return floats;
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