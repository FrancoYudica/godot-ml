#include <godot_cpp/classes/ref_counted.hpp>
#include "ml_inference/ml_tensor_resource_manager.hpp"
#include "ml_inference/ml_types.hpp"

namespace godot {

    class InferenceTask : public RefCounted {
        GDCLASS(InferenceTask, RefCounted);

    protected:
        static void _bind_methods();

    public:
        bool is_done = false;
        PackedFloat32Array input;
        Ref<ml::TensorResourceManager> tm;
        uint32_t graph_id;

    public:
        void init(uint32_t graph_id,
                  const PackedFloat32Array& input,
                  RenderingDevice* rd);
        void emit_completed();
        PackedFloat32Array get_output_data(const String& output_node);
    };

}  // namespace godot