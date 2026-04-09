// Umbrella header
#pragma once
#include "core/types.hpp" // IWYU pragma: export

namespace ml {

namespace Parser {
bool parse(const std::string& path, Graph& graph);
}

} // namespace ml