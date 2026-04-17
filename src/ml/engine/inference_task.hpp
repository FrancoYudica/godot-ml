#pragma once
#include "core/core.hpp"
#include "inference_descriptor.hpp"
#include "tensors/tensors.hpp"

#include <godot_cpp/classes/ref_counted.hpp>

namespace godot {

class InferenceTask : public RefCounted {
    GDCLASS(InferenceTask, RefCounted);

  protected:
    static void _bind_methods();

  public:
    void init(
        uint32_t graph_id,
        RenderingDevice* rd,
        Ref<InferenceDescriptor> descriptor,
        ml::StorageBufferPool* buffer_pool);
    void emit_completed();
    void destroy(RenderingDevice* rd);
    bool is_completed() const {
        return _is_completed;
    }

    bool is_freed() const {
        return _freed;
    }

  public:
    Ref<ml::TensorResourceManager> activations_tm;
    uint32_t graph_id;
    Ref<InferenceDescriptor> descriptor;
    std::unordered_map<std::string, godot::Variant> outputs;

  private:
    bool _is_completed = false;
    bool _freed = false;
};

} // namespace godot