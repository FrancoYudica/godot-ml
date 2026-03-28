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
        Ref<InferenceTask> run_async(uint32_t model_rid,
                                     const PackedFloat32Array& input);
        void print_model(uint32_t model_rid);

        PackedFloat32Array pop_task_output(Ref<InferenceTask> task,
                                           const String& output_name);

    protected:
        static void _bind_methods();

    private:
        bool _setup_shaders();
        void _process_pending_tasks();
        void _process_task(Ref<InferenceTask> task);
        void _run_node(const ml::GraphNode& node,
                       int64_t compute_list,
                       Ref<ml::TensorResourceManager> weights_tm,
                       Ref<ml::TensorResourceManager> activations_tm);

        void _dispatch_gemm(int64_t compute_list,
                            RID input_sb,
                            RID weight_sb,
                            RID bias_sb,
                            RID output_sb,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K,
                            float alpha,
                            float beta);

        void _dispatch_elementwise(int64_t compute_list,
                                   RID shader,
                                   RID pipeline,
                                   RID input,
                                   RID output,
                                   uint32_t M,
                                   uint32_t K);

        RID _get_shader(ml::NodeOperator op);
        RID _get_pipeline(ml::NodeOperator op);

        void _free_all_resources();

        void _process_deletion_queue();

    private:
        RenderingDevice* _rd;
        std::unordered_map<ml::NodeOperator, RID> _operator_shader;
        std::unordered_map<ml::NodeOperator, RID> _operator_pipeline;
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
