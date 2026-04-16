#pragma once
#include "core/core.hpp"

namespace ml::Lowering {
bool lower(ml::LogicalGraph& logical_graph, ml::PhysicalGraph& physical_graph);
}