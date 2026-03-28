#include "inference_task.hpp"

namespace godot {
    void godot::InferenceTask::_bind_methods() {
        ADD_SIGNAL(
            MethodInfo("completed",
                       PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "result")));
    }

    void InferenceTask::init(uint32_t graph_id,
                             const PackedFloat32Array& input,
                             uint32_t id) {
        this->graph_id = graph_id;
        this->input = input;
        this->id = id;
    }

    void InferenceTask::emit_completed() {
        emit_signal("completed");
    }
}  // namespace godot