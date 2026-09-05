#include "inference_task.hpp"

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/array.hpp>
namespace godot {
void godot::InferenceTask::_bind_methods() {
    ADD_SIGNAL(
        MethodInfo(
            "completed"));
    ClassDB::bind_method(
        D_METHOD("get_performance_report"),
        &InferenceTask::get_performance_report);
}

void InferenceTask::init(
    uint32_t task_id,
    uint32_t graph_id,
    RenderingDevice* rd,
    Ref<InferenceDescriptor> descriptor,
    ml::StorageBufferPool* buffer_pool) {
    this->task_id = task_id;
    this->graph_id = graph_id;
    this->activations_tm.instantiate();
    this->activations_tm->init(rd, buffer_pool);
    this->descriptor = descriptor;
}

Dictionary InferenceTask::get_performance_report() const {
    Dictionary report;
    if (!has_performance_report) {
        UtilityFunctions::print(
            "Performance report not available for task " + String::num(task_id) +
                ". This may be because the task is not yet completed or GPU timestamps were not captured.",
            __FUNCTION__);
        return report;
    }

    Array operators;
    for (const auto& entry : timestamp_entries) {
        Dictionary op;
        op["name"] = entry.name;
        op["duration_ms"] = entry.duration_ms;
        operators.push_back(op);
    }
    report["total_gpu_time_ms"] = total_gpu_time_ms;
    report["operators"] = operators;
    return report;
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