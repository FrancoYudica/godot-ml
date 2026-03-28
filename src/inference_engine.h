#ifndef ML_INFERENCE_ENGINE_H
#define ML_INFERENCE_ENGINE_H
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include "ml_inference/ml_types.hpp"
#include "ml_inference/ml_tensor_resource_manager.hpp"
#include "ml_inference/ml_parser.hpp"
#include "inference_task.hpp"

namespace godot {
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

    protected:
        static void _bind_methods();

    private:
        bool _setup_shaders();
        void _process_pending_tasks();
        void _process_task(Ref<InferenceTask> task);
        void _run_node(const ml::GraphNode& node,
                       int64_t compute_list,
                       Ref<ml::TensorResourceManager> tm);

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

    private:
        RenderingDevice* _rd;
        std::unordered_map<ml::NodeOperator, RID> _operator_shader;
        std::unordered_map<ml::NodeOperator, RID> _operator_pipeline;
        std::unordered_map<uint32_t, ml::Graph> _graphs;
        std::vector<RID> _transient_uniform_sets;
        std::vector<Ref<InferenceTask>> _pending_tasks;
        std::vector<Ref<InferenceTask>> _executing_tasks;
        bool _initialized = false;

        uint32_t _graph_next_rid = 1;
    };
}  // namespace godot

#endif
