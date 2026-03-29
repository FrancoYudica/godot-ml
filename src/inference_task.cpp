#include "inference_task.hpp"

namespace godot {
    void godot::InferenceTask::_bind_methods() {
        ADD_SIGNAL(
            MethodInfo("completed",
                       PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "result")));
    }

    void InferenceTask::init(uint32_t graph_id, RenderingDevice* rd) {
        this->graph_id = graph_id;
        this->activations_tm.instantiate();
        this->activations_tm->init(rd);
    }

    void InferenceTask::emit_completed() {
        emit_signal("completed");
    }
}  // namespace godot