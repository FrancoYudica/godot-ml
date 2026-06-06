#pragma once
#include "core/core.hpp" // IWYU pragma: export
#include <cstdint>
#include <string>

namespace ml::passes {

struct ParseResult {
    Logical::Graph graph;
    OperationResult status;
};

ParseResult parse(const uint8_t* data, size_t size);

} // namespace ml::passes
