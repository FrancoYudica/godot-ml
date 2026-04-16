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

godot::String node_operator_to_string(ml::PhysicalOp op);

godot::RID load_shader(godot::RenderingDevice* rd, const godot::String& path);

void print(const PhysicalGraph& graph);

bool tensor_shape_matches(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2);

godot::String shape_to_str(const std::vector<int64_t>& shape);

uint32_t get_tensor_floats(const std::vector<int64_t>& shape);

template <typename Iterator>
godot::String get_iterator_str(Iterator begin, Iterator end, const godot::String separator = ", ") {
    godot::String output = "[";

    for (Iterator it = begin; it != end; ++it) {
        // We use a helper function or a manual check to handle std::string
        // godot::Variant has a constructor for const char*, which std::string provides.
        // For other types (int, float, String), godot::Variant works directly.
        if constexpr (std::is_same_v<typename std::iterator_traits<Iterator>::value_type, std::string>) {
            // Special handling for std::string
            output += godot::Variant((*it).c_str()).stringify();
        } else {
            // Default handling for Godot types and primitives
            output += godot::Variant(*it).stringify();
        }

        Iterator next_it = it;
        if (++next_it != end) {
            output += separator;
        }
    }

    output += "]";
    return output;
}

} // namespace Utils
} // namespace ml
