#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include "ml_inference/ml_tensor_resource_manager.hpp"
#include "ml_inference/ml_types.hpp"
#include "ml_inference/ml_output_handler.hpp"
#include "ml_inference/ml_input_handler.hpp"

namespace godot {

    class InferenceTask : public RefCounted {
        GDCLASS(InferenceTask, RefCounted);

    protected:
        static void _bind_methods();

    public:
        bool is_done = false;
        bool freed = false;
        Ref<ml::TensorResourceManager> activations_tm;
        uint32_t graph_id;

        std::unordered_map<std::string, std::unique_ptr<ml::IInputHandler>>
            input_handlers;

        std::unordered_map<std::string, std::unique_ptr<ml::IOutputHandler>>
            output_handlers;

    public:
        void init(uint32_t graph_id, RenderingDevice* rd);
        void emit_completed();
    };

}  // namespace godot