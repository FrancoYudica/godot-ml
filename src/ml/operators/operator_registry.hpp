#pragma once
#include "core/core.hpp"
#include "operator.hpp"

namespace ml {

class OperatorRegistry {
  public:
    bool init(godot::RenderingDevice* rd);

    IOperator* get(PhysicalOp op) const;

    bool supports(PhysicalOp op) const;

    void destroy(godot::RenderingDevice* rd);

  private:
    template <typename T>
    bool _register(PhysicalOp op, godot::RenderingDevice* rd) {
        auto impl = std::make_unique<T>();
        if (impl->init(rd)) {
            _operators[op] = std::move(impl);
            return true;
        }
        return false;
    }
    std::unordered_map<PhysicalOp, std::unique_ptr<IOperator>> _operators;
};

} // namespace ml