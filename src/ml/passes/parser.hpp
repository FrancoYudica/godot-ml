#pragma once
#include "core/types.hpp" // IWYU pragma: export

namespace ml::passes {

struct ParseResult {
    LogicalGraph graph;
    OperationResult status;
};

ParseResult parse(const std::string& path);

} // namespace ml::passes
