#pragma once
#include "core/types.hpp" // IWYU pragma: export

namespace ml::passes {

struct ValidationResult {
    OperationResult status;
};

/**
 * Validates a LogicalGraph immediately after parsing, before lowering.
 * Checks per-node arity (input/output counts) and attribute values
 * (kernel dimensions, pad/stride counts, Gemm transB requirement).
 * Call between parse() and lower().
 */
ValidationResult validate(const LogicalGraph& graph);

/**
 * Validates a PhysicalGraph after lowering.
 * Checks edge connectivity (every input is a defined tensor), per-node arity,
 * attribute variant types, and op-specific invariants (GemmToImage image_shape_ref,
 * Col2Im source_activation existence).
 * Call between lower() and infer_shapes().
 */
ValidationResult validate(const PhysicalGraph& graph);

} // namespace ml::passes
