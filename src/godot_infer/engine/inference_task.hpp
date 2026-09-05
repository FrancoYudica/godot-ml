#pragma once
#include "core/core.hpp"
#include "inference_descriptor.hpp"
#include "tensors/tensors.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>

namespace godot {

class InferenceTask : public RefCounted {
    GDCLASS(InferenceTask, RefCounted);

  protected:
    static void _bind_methods();

  public:
    struct TimestampEntry {
        String name;
        float duration_ms;
    };

    void init(
        uint32_t task_id,
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

    Dictionary get_performance_report() const;

  public:
    Ref<ml::TensorResourceManager> activations_tm;
    uint32_t task_id = 0;
    uint32_t graph_id;
    Ref<InferenceDescriptor> descriptor;
    std::unordered_map<std::string, godot::Variant> outputs;

    std::vector<TimestampEntry> timestamp_entries;
    float total_gpu_time_ms = 0.0f;
    bool has_performance_report = false;

  private:
    bool _is_completed = false;
    bool _freed = false;
};

} // namespace godot