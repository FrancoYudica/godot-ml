#pragma once
#include <unordered_map>
#include <memory>
#include "ml_operator.hpp"
#include "core/core.hpp"

namespace ml {

    class OperatorRegistry {
    public:
        bool init(godot::RenderingDevice* rd);

        IOperator* get(NodeOperator op) const;

        bool supports(NodeOperator op) const;

        void destroy(godot::RenderingDevice* rd);

    private:
        template <typename T>
        bool _register(NodeOperator op, godot::RenderingDevice* rd) {
            auto impl = std::make_unique<T>();
            if (impl->init(rd)) {
                _operators[op] = std::move(impl);
                return true;
            }
            return false;
        }
        std::unordered_map<NodeOperator, std::unique_ptr<IOperator>> _operators;
    };

}  // namespace ml