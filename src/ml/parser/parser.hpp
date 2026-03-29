#pragma once
#include "core/ml_types.hpp"

namespace ml {

    namespace Parser {
        bool parse(const std::string& path, Graph& graph);
    }

}  // namespace ml