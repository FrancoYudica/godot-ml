#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace ml::passes {

struct LoweringResult {
    Physical::Graph graph;
    OperationResult status;
};

LoweringResult lower(const Logical::Graph& logical_graph);

} // namespace ml::passes
