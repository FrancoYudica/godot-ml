#pragma once
#include "core/core.hpp" // IWYU pragma: export

namespace ml::passes {

/**
 * Validates a Logical::Graph immediately after parsing, before lowering.
 * Checks per-node arity (input/output counts) and attribute values
 * (kernel dimensions, pad/stride counts, Gemm transB requirement).
 * Call between parse() and lower().
 */
OperationResult validate_parse(const Logical::Graph& graph);

} // namespace ml::passes
