#include "inference_engine.h"
#include "ml_inference/ml_parser.hpp"
#include "ml_inference/ml_utils.hpp"
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

struct GemmPushConstants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float alpha;
    float beta;
};

struct ElementwisePushConstants {
    uint32_t M;
    uint32_t K;
};

namespace godot {

    void MLInferenceEngine::_bind_methods() {
        ClassDB::bind_method(D_METHOD("load", "model_path"),
                             &MLInferenceEngine::load);
    }

    bool MLInferenceEngine::load(String model_path) {
        _load_success = false;

        print_line("Loading ML inference engine with model: " + model_path);
        bool success = ml::Parser::parse(model_path.utf8().ptr(), _graph);

        if (!success) return false;

        UtilityFunctions::print("InferenceEngine: model parsed successfully.");

        ml::Utils::print(_graph);

        _rd = RenderingServer::get_singleton()->get_rendering_device();

        _tm.instantiate();
        _tm->init(_rd);
        for (const auto& [name, tensor] : _graph.initializers) {
            _tm->get_or_create(name, tensor.shape, tensor.data);
        }

        UtilityFunctions::print("InferenceEngine: initializers uploaded.");

        if (!_setup_shaders()) {
            return false;
        }

        UtilityFunctions::print("InferenceEngine: shaders setup successfully.");

        _load_success = true;
        return true;
    }

    bool MLInferenceEngine::run(const std::vector<float>& input,
                                uint32_t width,
                                uint32_t height) {
        if (!_load_success) {
            UtilityFunctions::print("InferenceEngine: model not loaded.");
            return false;
        }

        uint32_t num_pixels = width * height;

        // Upload input buffer
        std::vector<int64_t> dim = {num_pixels, _graph.input_shape[1]};
        _tm->get_or_create(_graph.input_name, dim, input);

        for (const ml::GraphNode& node : _graph.nodes) {
            switch (node.op) {
                case ml::NodeOperator::Gemm: {
                    // input[0] activations, input[1] weights, input[2] bias
                    RID input_sb = _tm->get_or_create(node.inputs[0]);
                    RID weight_sb = _tm->get_or_create(node.inputs[1]);
                    RID bias_sb = _tm->get_or_create(node.inputs[2]);

                    // Derive output shape from weight matrix: (M, N)
                    // weights stored as (N, K) with transB=true
                    auto& w_shape =
                        _graph.initializers.at(node.inputs[1]).shape;
                    uint32_t M = num_pixels;
                    uint32_t N = static_cast<uint32_t>(w_shape[0]);
                    uint32_t K = static_cast<uint32_t>(w_shape[1]);

                    RID out_buf =
                        _tm->get_or_create(node.outputs[0], {M, N}, {});
                } break;

                // Element wise operations
                case ml::NodeOperator::ReLU:
                case ml::NodeOperator::Sigmoid: {
                } break;
            }
        }

        return false;
    }

    bool MLInferenceEngine::_setup_shaders() {
        std::unordered_map<ml::NodeOperator, String> operator_to_path = {
            {ml::NodeOperator::Gemm, "shaders/gemm.glsl"},
            {ml::NodeOperator::ReLU, "shaders/relu.glsl"},
            {ml::NodeOperator::Sigmoid, "shaders/sigmoid.glsl"}};

        auto load_shader = [&](const String& path) -> RID {
            Ref<RDShaderFile> shader_file =
                ResourceLoader::get_singleton()->load(path);

            if (shader_file.is_null()) {
                ERR_PRINT("Failed to load path tracer shader file. " + path);
                return RID();
            }
            Ref<RDShaderSPIRV> spirv = shader_file->get_spirv();
            if (spirv.is_null()) {
                ERR_PRINT("Shader  not contain SPIR-V : " + path);
                return RID();
            }
            return _rd->shader_create_from_spirv(spirv);
        };

        // Setup shaders for each operator
        for (const auto& [op, path] : operator_to_path) {
            const String& shader_path =
                ml::Utils::get_project_relative_path(path);

            RID rid = load_shader(shader_path);
            if (!rid.is_valid()) {
                return false;
            }
            _operator_shader[op] = rid;
        }
        return true;
    }

    void MLInferenceEngine::_dispatch_gemm(RID input_sb,
                                           RID weights_sb,
                                           RID bias_sb,
                                           RID output_sb,
                                           uint32_t M,
                                           uint32_t N,
                                           uint32_t K,
                                           float alpha,
                                           float beta) {
        // Build uniform set
        TypedArray<RDUniform> uniforms;

        auto make_uniform = [&](RID rid, int binding) {
            Ref<RDUniform> u;
            u.instantiate();
            u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
            u->set_binding(binding);
            u->add_id(rid);
            return u;
        };

        uniforms.push_back(make_uniform(input_sb, 0));
        uniforms.push_back(make_uniform(weights_sb, 1));
        uniforms.push_back(make_uniform(bias_sb, 2));
        uniforms.push_back(make_uniform(output_sb, 3));

        RID uniform_set = _rd->uniform_set_create(
            uniforms, _get_shader(ml::NodeOperator::Gemm), 0);

        // Push constants
        GemmPushConstants pc{M, N, K, alpha, beta};
        PackedByteArray pc_bytes;
        pc_bytes.resize(sizeof(GemmPushConstants));
        memcpy(pc_bytes.ptrw(), &pc, sizeof(GemmPushConstants));

        // Dispatch
        int64_t compute_list = _rd->compute_list_begin();
        _rd->compute_list_bind_compute_pipeline(
            compute_list,
            _rd->compute_pipeline_create(_get_shader(ml::NodeOperator::Gemm)));
        _rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
        _rd->compute_list_set_push_constant(compute_list, pc_bytes,
                                            sizeof(GemmPushConstants));
        _rd->compute_list_dispatch(compute_list, (M + 63) / 64, 1, 1);
        _rd->compute_list_end();
    }

    RID MLInferenceEngine::_get_shader(ml::NodeOperator op) {
        auto it = _operator_shader.find(op);
        if (it != _operator_shader.end()) {
            return it->second;
        }
        ERR_PRINT("Shader not found for operator: " +
                  String(ml::Utils::node_operator_to_string(op)));
        return RID();
    }

}  // namespace godot