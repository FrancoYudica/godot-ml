#include "inference_task.hpp"

namespace godot {
    void godot::InferenceTask::_bind_methods() {
        ADD_SIGNAL(
            MethodInfo("completed",
                       PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "result")));

        ClassDB::bind_method(D_METHOD("get_output_data", "tensor_name"),
                             &InferenceTask::get_output_data);
    }

    void InferenceTask::init(uint32_t graph_id,
                             const PackedFloat32Array& input,
                             RenderingDevice* rd) {
        this->graph_id = graph_id;
        this->input = input;
        this->tm.instantiate();
        this->tm->init(rd);
    }

    void InferenceTask::emit_completed() {
        emit_signal("completed");
    }

    PackedFloat32Array InferenceTask::get_output_data(
        const String& output_node) {
        if (!is_done) {
            UtilityFunctions::print(
                "InferenceTask: Task not yet completed by GPU.");
            return PackedFloat32Array();
        }

        PackedByteArray byte_array = tm->get_buffer(output_node.utf8().ptr());

        // Transform byte array to float array
        PackedFloat32Array float_array;
        for (size_t i = 0; i < byte_array.size(); i += sizeof(float)) {
            float value;
            memcpy(&value, byte_array.ptr() + i, sizeof(float));
            float_array.append(value);
        }
        return float_array;
    }

}  // namespace godot