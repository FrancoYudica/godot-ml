#pragma once
#include "types.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/typed_array.hpp>

namespace ml {
namespace Utils {
/**
 * Gets the project-relative path for a given addon-relative path
 */
godot::String get_project_relative_path(const godot::String& path);

godot::String node_operator_to_string(ml::NodeOperator op);

const std::vector<ml::NodeOperator>& get_node_operators();

godot::RID load_shader(godot::RenderingDevice* rd, const godot::String& path);

void print(const Graph& graph);

bool tensor_shape_matches(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2);

godot::String shape_to_str(const std::vector<int64_t>& shape);
} // namespace Utils
} // namespace ml
