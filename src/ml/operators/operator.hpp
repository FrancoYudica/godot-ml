#pragma once
#include "core/core.hpp"
#include "tensors/tensors.hpp"

#include <godot_cpp/classes/rendering_device.hpp>

namespace ml {
struct OperatorContext {
    RenderingDevice* rd;
    Ref<TensorResourceManager> weights_tm;
    Ref<TensorResourceManager> activations_tm;
    int64_t compute_list;
    DeletionStack* frame_deletion_stack;
};

class IOperator {
  public:
    virtual ~IOperator() = default;
    virtual bool init(godot::RenderingDevice* rd) = 0;
    virtual void dispatch(
        const ml::GraphNode& node,
        const OperatorContext& ctx) = 0;
    virtual void destroy(godot::RenderingDevice* rd) = 0;
};

} // namespace ml