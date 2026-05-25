#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace ml::passes {

/**
 * Validates a Logical::Graph immediately after parsing, before lowering.
 * Checks per-node arity (input/output counts) and attribute values
 * (kernel dimensions, pad/stride counts, Gemm transB requirement).
 * Call between parse() and lower().
 */
OperationResult validate(const Logical::Graph& graph);

/**
 * Validates a Physical::Graph after lowering.
 * Checks edge connectivity (every input is a defined tensor), per-node arity,
 * attribute variant types, and op-specific invariants (GemmToImage image_shape_ref,
 * Col2Im source_activation existence).
 * Call between lower() and infer_shapes().
 */
OperationResult validate(const Physical::Graph& graph);

} // namespace ml::passes
