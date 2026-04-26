#pragma once
#include "core/types.hpp" // IWYU pragma: export

namespace ml::passes {

struct LoweringResult {
    PhysicalGraph graph;
    OperationResult status;
};

LoweringResult lower(const LogicalGraph& logical_graph);

} // namespace ml::passes
