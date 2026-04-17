#include "inference_task.hpp"

#include <godot_cpp/core/error_macros.hpp>
namespace godot {
void godot::InferenceTask::_bind_methods() {
    ADD_SIGNAL(
        MethodInfo(
            "completed",
            PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "result")));
}

void InferenceTask::init(
    uint32_t graph_id,
    RenderingDevice* rd,
    Ref<InferenceDescriptor> descriptor,
    ml::StorageBufferPool* buffer_pool) {
    this->graph_id = graph_id;
    this->activations_tm.instantiate();
    this->activations_tm->init(rd, buffer_pool);
    this->descriptor = descriptor;
}

void InferenceTask::emit_completed() {
    _is_completed = true;
    emit_signal("completed");
}

void InferenceTask::destroy(RenderingDevice* rd) {
    activations_tm->destroy();
    _freed = true;
}
} // namespace godot