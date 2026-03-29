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
        void init(uint32_t graph_id, RenderingDevice* rd);
        void emit_completed();
        void destroy(RenderingDevice* rd);
        bool is_completed() const {
            return _is_completed;
        }

        bool is_freed() const {
            return _freed;
        }

        void add_input_handler(const std::string& tensor_name,
                               std::unique_ptr<ml::IInputHandler> handler);

        void add_output_handler(const std::string& output_name,
                                std::unique_ptr<ml::IOutputHandler> handler);

    public:
        Ref<ml::TensorResourceManager> activations_tm;
        uint32_t graph_id;

        /**
         * Maps the tensor name to the input handler. This is done
         * to easily make sure that the user isn't defining more than one
         * input handler per tensor.
         */
        std::unordered_map<std::string, std::unique_ptr<ml::IInputHandler>>
            input_handlers;

        /**
         * Maps the output name to the output handler. Note that the output name
         * is user defined, since a single tensor could have many outputs. For
         * example, the case where the user wants to get the result as a texture
         * but also into a float array.
         */
        std::unordered_map<std::string, std::unique_ptr<ml::IOutputHandler>>
            output_handlers;

    private:
        bool _is_completed = false;
        bool _freed = false;
    };

}  // namespace godot