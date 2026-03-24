#include "inference_engine.h"
#include "ml_inference/ml_parser.hpp"
#include "ml_inference/ml_utils.hpp"
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

struct GemmPushConstants {
    uint32_t M, N, K;
    float alpha, beta;
    float padding[3];
};

struct ElementwisePushConstants {
    uint32_t M;
    uint32_t K;
    float padding[2];
};

namespace godot {

    void MLInferenceEngine::_bind_methods() {
        ClassDB::bind_method(D_METHOD("load", "model_path"),
                             &MLInferenceEngine::load);

        ClassDB::bind_method(D_METHOD("run", "input"), &MLInferenceEngine::run);

        ClassDB::bind_method(D_METHOD("get_output_data", "tensor"),
                             &MLInferenceEngine::get_output_data);
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

    bool MLInferenceEngine::run(const PackedFloat32Array& input) {
        if (!_load_success) {
            UtilityFunctions::print("InferenceEngine: model not loaded.");
            return false;
        }

        // Upload input buffer
        // If we have 300 float, that means that we have 100 pixels.
        int64_t first_axis = input.size() / _graph.input_shape[1];

        std::vector<int64_t> dim = {first_axis, _graph.input_shape[1]};
        _tm->get_or_create(_graph.input_name, dim, input.to_byte_array());

        int compute_list = _rd->compute_list_begin();

        for (size_t i = 0; i < _graph.nodes.size(); ++i) {
            const ml::GraphNode& node = _graph.nodes[i];

            _run_node(node, compute_list);
            // Add a barrier unless it's the very last node
            if (i < _graph.nodes.size() - 1) {
                _rd->compute_list_add_barrier(compute_list);
            }
        }
        _rd->compute_list_end();
        return true;
    }

    PackedFloat32Array MLInferenceEngine::get_output_data(
        const String& tensor) {
        RID sb = _tm->get_or_create(tensor.utf8().ptr());

        PackedByteArray byte_array = _rd->buffer_get_data(sb);

        // Transform byte array to float array
        PackedFloat32Array float_array;
        for (size_t i = 0; i < byte_array.size(); i += sizeof(float)) {
            float value;
            memcpy(&value, byte_array.ptr() + i, sizeof(float));
            float_array.append(value);
        }
        return float_array;
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

        // All shaders loaded successfully
        // time to create the pipelines

        for (const auto& [op, shader_rid] : _operator_shader) {
            // Create pipeline for each operator
            RID rid = _rd->compute_pipeline_create(shader_rid);
            if (!rid.is_valid()) {
                return false;
            }
            _operator_pipeline[op] = rid;
        }

        return true;
    }

    void MLInferenceEngine::_run_node(const ml::GraphNode& node,
                                      int64_t compute_list) {
        UtilityFunctions::print("Running node: " +
                                ml::Utils::node_operator_to_string(node.op));

        switch (node.op) {
            case ml::NodeOperator::Gemm: {
                UtilityFunctions::print(" - Input: " +
                                        String(node.inputs[0].c_str()));
                UtilityFunctions::print(" - Weights: " +
                                        String(node.inputs[1].c_str()));
                UtilityFunctions::print(" - Bias: " +
                                        String(node.inputs[2].c_str()));

                RID input_sb = _tm->get_or_create(node.inputs[0]);
                RID weight_sb = _tm->get_or_create(node.inputs[1]);
                RID bias_sb = _tm->get_or_create(node.inputs[2]);

                // Derive full shape from stored tensors
                // input: (M, K)
                // weights: (N, K) with transB = true
                auto& in_shape = _tm->get_tensor_shape(node.inputs[0]);
                auto& w_shape = _tm->get_tensor_shape(node.inputs[1]);

                // pixels
                uint32_t M = static_cast<uint32_t>(in_shape[0]);

                // output features
                uint32_t N = static_cast<uint32_t>(w_shape[0]);

                // input features
                uint32_t K = static_cast<uint32_t>(w_shape[1]);

                RID out_buf = _tm->get_or_create(node.outputs[0], {M, N});
                _dispatch_gemm(compute_list, input_sb, weight_sb, bias_sb,
                               out_buf, M, N, K, node.alpha, node.beta);
            } break;

            case ml::NodeOperator::ReLU:
            case ml::NodeOperator::Sigmoid: {
                UtilityFunctions::print(" - Input: " +
                                        String(node.inputs[0].c_str()));

                RID input_sb = _tm->get_or_create(node.inputs[0]);

                auto& in_shape = _tm->get_tensor_shape(node.inputs[0]);
                uint32_t M = static_cast<uint32_t>(in_shape[0]);
                uint32_t K = static_cast<uint32_t>(in_shape[1]);

                RID out_buf = _tm->get_or_create(node.outputs[0], {M, K});
                _dispatch_elementwise(compute_list, _get_shader(node.op),
                                      _get_pipeline(node.op), input_sb, out_buf,
                                      M, K);
            } break;

            default:
                ERR_PRINT("MLInferenceEngine: unsupported operator.");
                break;
        }
    }
    void MLInferenceEngine::_dispatch_gemm(int64_t compute_list,
                                           RID input_sb,
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

        // Push constant
        GemmPushConstants pc = {M, N, K, alpha, beta, {0, 0, 0}};
        PackedByteArray packed_pc;
        packed_pc.resize(sizeof(GemmPushConstants));
        memcpy(packed_pc.ptrw(), &pc, sizeof(GemmPushConstants));

        // Dispatch
        _rd->compute_list_bind_compute_pipeline(
            compute_list, _get_pipeline(ml::NodeOperator::Gemm));
        _rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
        _rd->compute_list_set_push_constant(compute_list, packed_pc,
                                            sizeof(GemmPushConstants));

        _rd->compute_list_dispatch(compute_list, (M + 63) / 64, 1, 1);
    }

    void MLInferenceEngine::_dispatch_elementwise(int64_t compute_list,
                                                  RID shader,
                                                  RID pipeline,
                                                  RID input,
                                                  RID output,
                                                  uint32_t M,
                                                  uint32_t K) {
        TypedArray<RDUniform> uniforms;

        auto make_uniform = [&](RID rid, int binding) {
            Ref<RDUniform> u;
            u.instantiate();
            u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
            u->set_binding(binding);
            u->add_id(rid);
            return u;
        };

        uniforms.push_back(make_uniform(input, 0));
        uniforms.push_back(make_uniform(output, 1));

        RID uniform_set = _rd->uniform_set_create(uniforms, shader, 0);

        ElementwisePushConstants pc{M, K};
        PackedByteArray pc_bytes;
        pc_bytes.resize(sizeof(ElementwisePushConstants));
        memcpy(pc_bytes.ptrw(), &pc, sizeof(ElementwisePushConstants));

        _rd->compute_list_bind_compute_pipeline(compute_list, pipeline);
        _rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
        _rd->compute_list_set_push_constant(compute_list, pc_bytes,
                                            sizeof(ElementwisePushConstants));
        _rd->compute_list_dispatch(compute_list, (M * K + 63) / 64, 1, 1);
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

    RID MLInferenceEngine::_get_pipeline(ml::NodeOperator op) {
        auto it = _operator_pipeline.find(op);
        if (it != _operator_pipeline.end()) {
            return it->second;
        }
        ERR_PRINT("Pipeline not found for operator: " +
                  String(ml::Utils::node_operator_to_string(op)));
        return RID();
    }

}  // namespace godot