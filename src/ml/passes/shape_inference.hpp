#pragma once
#include "core/types.hpp"

namespace ml::passes {

struct ShapeInferenceResult {
    ShapeTable shapes;
    OperationResult status;
};

// Runs a forward shape propagation pass over the physical graph.
//
// input_shapes provides the concrete runtime shape for every model input tensor.
// The pass seeds the ShapeTable with inputs + initializer shapes, then walks
// nodes in topological order applying per-operator shape rules.
//
ShapeInferenceResult infer_shapes(
    const PhysicalGraph& graph,
    const ShapeTable& input_shapes);

} // namespace ml::passes