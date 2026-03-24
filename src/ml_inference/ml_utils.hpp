#pragma once
#include "ml_types.hpp"

namespace ml {
    namespace Utils {
        /**
         * Gets the project-relative path for a given addon-relative path
         */
        godot::String get_project_relative_path(const godot::String& path);

        godot::String node_operator_to_string(ml::NodeOperator op);

        void print(const Graph& graph);
    }  // namespace Utils
}  // namespace ml
