#ifndef ML_INFERENCE_ENGINE_H
#define ML_INFERENCE_ENGINE_H
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <queue>
#include <functional>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include "ml_inference/ml_types.hpp"
#include "ml_inference/ml_tensor_resource_manager.hpp"
#include "ml_inference/ml_parser.hpp"
#include "inference_task.hpp"
#include "ml_inference/ml_operator_registry.hpp"

namespace godot {

    struct GraphContext {
        ml::Graph graph;
        Ref<ml::TensorResourceManager> weights_tm;
    };

    class MLInferenceEngine : public RefCounted {
        GDCLASS(MLInferenceEngine, RefCounted)

    public:
        void init();
        void destroy();
        uint32_t register_model(String model_path);
        void unload_model(uint32_t model_rid);
        Ref<InferenceTask> run_async(uint32_t model_rid);
        void print_model(uint32_t model_rid);

        PackedFloat32Array pop_task_output(Ref<InferenceTask> task,
                                           const String& output_name);

        void add_float_array_input(Ref<InferenceTask> task,
                                   const String& tensor_name,
                                   const PackedFloat32Array& data,
                                   const PackedFloat64Array& shape);

        void add_float_array_output(Ref<InferenceTask> task,
                                    const String& tensor_name);

    protected:
        static void _bind_methods();

    private:
        void _process_pending_tasks();
        void _process_task(Ref<InferenceTask> task);
        void _run_node(const ml::GraphNode& node,
                       int64_t compute_list,
                       Ref<ml::TensorResourceManager> weights_tm,
                       Ref<ml::TensorResourceManager> activations_tm);

        void _free_all_resources();
        void _process_deletion_queue();
        bool _has_graph(uint32_t graph_rid);

    private:
        RenderingDevice* _rd;
        ml::OperatorRegistry _operator_registry;
        std::unordered_map<uint32_t, GraphContext> _graphs;
        std::vector<Ref<InferenceTask>> _pending_tasks;
        std::vector<Ref<InferenceTask>> _executing_tasks;
        bool _initialized = false;
        bool _destroying = false;

        std::queue<std::function<void()>> _deletion_queue;

        uint32_t _next_graph_id = 1;
    };
}  // namespace godot

#endif
