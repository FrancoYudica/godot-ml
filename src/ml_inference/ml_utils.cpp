#include "ml_utils.hpp"
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

namespace ml {
    namespace Utils {

        godot::String get_project_relative_path(
            const godot::String& addon_relative_path) {
            static const godot::String base_path = "res://addons/cnn/";
            return base_path + addon_relative_path;
        }

        godot::String node_operator_to_string(NodeOperator op) {
            static const std::unordered_map<NodeOperator, std::string>
                operator_names = {{NodeOperator::Unknown, "Unknown"},
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
                NodeOperator::Gemm, NodeOperator::ReLU, NodeOperator::Sigmoid};

            return operators;
        }

        void print(const Graph& graph) {
            auto to_gstring = [](const std::string& s) {
                return String(s.c_str());
            };

            UtilityFunctions::print("Input: ", to_gstring(graph.input_name));

            String shape_str = "Input shape: [";
            for (const auto& i : graph.input_shape)
                shape_str += String::num_int64(i) + ",";
            shape_str += "]";
            UtilityFunctions::print(shape_str);

            for (const auto& node : graph.nodes) {
                UtilityFunctions::print("Node: ",
                                        node_operator_to_string(node.op));

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
    }  // namespace Utils
}  // namespace ml