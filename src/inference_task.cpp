#include "inference_task.hpp"
#include <godot_cpp/core/error_macros.hpp>
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
    void InferenceTask::add_input_handler(
        const std::string& tensor_name,
        std::unique_ptr<ml::IInputHandler> handler) {
        ERR_FAIL_COND_MSG(
            input_handlers.find(tensor_name) != input_handlers.end(),
            "InferenceEngine: Input handler for tensor \"" +
                String(tensor_name.c_str()) + "\" already exists.");

        input_handlers[tensor_name] = std::move(handler);
    }

    void InferenceTask::add_output_handler(
        const std::string& output_name,
        std::unique_ptr<ml::IOutputHandler> handler) {
        ERR_FAIL_COND_MSG(
            output_handlers.find(output_name) != output_handlers.end(),
            "InferenceEngine: Output handler for output \"" +
                String(output_name.c_str()) + "\" already exists.");

        output_handlers[output_name] = std::move(handler);
    }
}  // namespace godot