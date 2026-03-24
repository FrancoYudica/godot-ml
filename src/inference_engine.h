#ifndef ML_INFERENCE_ENGINE_H
#define ML_INFERENCE_ENGINE_H
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include "ml_inference/ml_types.hpp"
#include "ml_inference/ml_tensor_resource_manager.hpp"
#include "ml_inference/ml_parser.hpp"
namespace godot {
    class MLInferenceEngine : public RefCounted {
        GDCLASS(MLInferenceEngine, RefCounted)

    public:
        bool load(String model_path);
        bool run(const std::vector<float>& input,
                 uint32_t width,
                 uint32_t height);

    protected:
        static void _bind_methods();

    private:
        bool _setup_shaders();
        void _dispatch_gemm(RID input_sb,
                            RID weight_sb,
                            RID bias_sb,
                            RID output_sb,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K,
                            float alpha,
                            float beta);

        RID _get_shader(ml::NodeOperator op);

    private:
        Ref<ml::TensorResourceManager> _tm;
        ml::Graph _graph;
        RenderingDevice* _rd;
        std::unordered_map<ml::NodeOperator, RID> _operator_shader;
        bool _load_success;
    };
}  // namespace godot

#endif
