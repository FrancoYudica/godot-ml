#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace ml::passes {

struct ParseResult {
    Logical::Graph graph;
    OperationResult status;
};

ParseResult parse(const std::string& path);

} // namespace ml::passes
