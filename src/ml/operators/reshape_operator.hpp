#include "operator.hpp"

namespace ml {
class ReshapeOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd) override;
    virtual void dispatch(
        const ml::PhysicalNode& node,
        const OperatorContext& ctx) override;
    void destroy(godot::RenderingDevice* rd) override;
};

} // namespace ml