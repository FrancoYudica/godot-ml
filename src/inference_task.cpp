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
        _is_completed = true;
        emit_signal("completed");
    }

    void InferenceTask::destroy(RenderingDevice* rd) {
        activations_tm->destroy();

        for (auto& [_, handler] : input_handlers) {
            handler->destroy(rd);
        }

        for (auto& [_, handler] : output_handlers) {
            handler->destroy(rd);
        }

        input_handlers.clear();
        output_handlers.clear();

        _freed = true;
    }
}  // namespace godot