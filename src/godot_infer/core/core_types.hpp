#pragma once
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/string.hpp>
#include <memory> // IWYU pragma: export
#include <string>
#include <unordered_map>
#include <variant> // IWYU pragma: export
#include <vector>

namespace ml {

// Maps every tensor name (inputs, initializers, intermediates) to its concrete shape.
// Populated by the shape inference pass before each inference call.
using ShapeTable = std::unordered_map<std::string, std::vector<int64_t>>;

struct OperationResult {
    bool success = true;
    std::string error;
};

#define OPERATION_OK \
    {                \
        true, {}     \
    }

/**
 * Tensor holds its shape and weight data
 */
struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};
} // namespace ml
