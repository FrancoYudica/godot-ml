#include <godot_cpp/classes/ref_counted.hpp>
#include "ml_inference/ml_tensor_resource_manager.hpp"
#include "ml_inference/ml_types.hpp"

namespace godot {

    class InferenceTask : public RefCounted {
        GDCLASS(InferenceTask, RefCounted);

    protected:
        static void _bind_methods();

    public:
        uint32_t id = 0;
        bool is_done = false;
        bool freed = false;
        PackedFloat32Array input;
        uint32_t graph_id;

    public:
        void init(uint32_t graph_id,
                  const PackedFloat32Array& input,
                  uint32_t id);
        void emit_completed();
    };

}  // namespace godot