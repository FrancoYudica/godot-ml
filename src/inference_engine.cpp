#include <unordered_set>
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
        ClassDB::bind_method(D_METHOD("init"), &MLInferenceEngine::init);
        ClassDB::bind_method(D_METHOD("destroy"), &MLInferenceEngine::destroy);

        ClassDB::bind_method(D_METHOD("register_model", "model_path"),
                             &MLInferenceEngine::register_model);

        ClassDB::bind_method(D_METHOD("run_async", "model_rid", "input"),
                             &MLInferenceEngine::run_async);

        ClassDB::bind_method(D_METHOD("print_model", "model_rid"),
                             &MLInferenceEngine::print_model);

        ClassDB::bind_method(D_METHOD("unload_model", "model_rid"),
                             &MLInferenceEngine::unload_model);

        ClassDB::bind_method(D_METHOD("pop_task_output", "task", "output_name"),
                             &MLInferenceEngine::pop_task_output);

        ClassDB::bind_method(D_METHOD("_process_pending_tasks"),
                             &MLInferenceEngine::_process_pending_tasks);
    }

    void MLInferenceEngine::init() {
        if (_initialized) {
            UtilityFunctions::print("InferenceEngine: already initialized.");
            return;
        }

        _rd = RenderingServer::get_singleton()->get_rendering_device();

        if (!_rd) {
            ERR_PRINT(
                "InferenceEngine: Could not get RenderingDevice. Are you "
                "using the Compatibility renderer?");
            return;
        }

        if (!_setup_shaders()) {
            UtilityFunctions::print(
                "InferenceEngine: failed to setup shaders.");
            return;
        }

        UtilityFunctions::print("InferenceEngine: shaders setup successfully.");
        _initialized = true;

        RenderingServer::get_singleton()->connect(
            "frame_pre_draw", Callable(this, "_process_pending_tasks"));
    }

    uint32_t MLInferenceEngine::register_model(String model_path) {
        if (!_initialized) {
            UtilityFunctions::print("InferenceEngine: not initialized.");
            return 0;
        }

        print_line("Loading ML inference engine with model: " + model_path);

        GraphContext graph_context;

        bool success =
            ml::Parser::parse(model_path.utf8().ptr(), graph_context.graph);

        if (!success) return 0;
        UtilityFunctions::print("InferenceEngine: model parsed successfully.");

        graph_context.weights_tm.instantiate();
        graph_context.weights_tm->init(_rd);

        uint32_t graph_rid = _next_graph_id++;
        _graphs[graph_rid] = graph_context;

        // Makes sure that the graph initializers are loaded to the task tm
        for (const auto& [name, tensor] : graph_context.graph.initializers) {
            graph_context.weights_tm->get_or_create(name, tensor.shape,
                                                    tensor.data);
        }

        return graph_rid;
    }
    void MLInferenceEngine::unload_model(uint32_t model_rid) {
        if (_graphs.find(model_rid) == _graphs.end()) {
            UtilityFunctions::print("InferenceEngine: model " +
                                    String::num(model_rid) + " not found.");
            return;
        }

        _graphs.erase(model_rid);
    }

    void MLInferenceEngine::destroy() {
        _destroying = true;
    }

    Ref<InferenceTask> MLInferenceEngine::run_async(
        uint32_t model_rid, const PackedFloat32Array& input) {
        auto it = _graphs.find(model_rid);
        if (it == _graphs.end()) {
            UtilityFunctions::print("InferenceEngine: model not found.");
            return nullptr;
        }

        Ref<InferenceTask> task;
        task.instantiate();
        task->init(model_rid, input, _rd);
        _pending_tasks.push_back(task);
        return task;
    }

    void MLInferenceEngine::print_model(uint32_t model_rid) {
        auto it = _graphs.find(model_rid);
        if (it != _graphs.end()) {
            ml::Utils::print(it->second.graph);
        } else {
            UtilityFunctions::print("InferenceEngine: model not found.");
        }
    }

    PackedFloat32Array MLInferenceEngine::pop_task_output(
        Ref<InferenceTask> task, const String& output_name) {
        if (!task->is_done) {
            UtilityFunctions::print(
                "InferenceTask: Task not yet completed by GPU.");
            return PackedFloat32Array();
        }

        if (task->freed) {
            ERR_PRINT(
                "InferenceEngine: Task already resources freed. Unable to "
                "retrieve output");
            return PackedFloat32Array();
        }

        RID sb = task->activations_tm->get_buffer_rid(output_name.utf8().ptr());

        if (!sb.is_valid()) {
            UtilityFunctions::print("InferenceEngine: Buffer not found.");
            return PackedFloat32Array();
        }

        PackedByteArray byte_array = _rd->buffer_get_data(sb);
        PackedFloat32Array float_array;
        float_array.resize(byte_array.size() / sizeof(float));
        memcpy(float_array.ptrw(), byte_array.ptrw(), byte_array.size());

        // Frees all the resources
        task->activations_tm->clear();
        task->freed = true;
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

    void MLInferenceEngine::_process_pending_tasks() {
        if (_destroying && _pending_tasks.empty() && _executing_tasks.empty()) {
            _free_all_resources();
            RenderingServer::get_singleton()->disconnect(
                "frame_pre_draw", Callable(this, "_process_pending_tasks"));

            _destroying = false;
            return;
        }

        _process_deletion_queue();

        // Tasks that were executing now are finished
        for (const auto& task : _executing_tasks) {
            task->is_done = true;
            task->emit_completed();
        }

        _executing_tasks.clear();

        for (const auto& task : _pending_tasks) {
            _process_task(task);
            _executing_tasks.push_back(task);
        }

        _pending_tasks.clear();
    }

    void MLInferenceEngine::_process_task(Ref<InferenceTask> task) {
        auto it = _graphs.find(task->graph_id);
        if (it == _graphs.end()) {
            UtilityFunctions::print("InferenceEngine: graph not found.");
            return;
        }
        const ml::Graph& graph = it->second.graph;
        Ref<ml::TensorResourceManager> weights_tm = it->second.weights_tm;

        const PackedFloat32Array& input = task->input;

        if (graph.input_shape.size() < 2) {
            ERR_PRINT("InferenceEngine: Invalid graph input shape for model " +
                      String::num(task->graph_id) + ". Graph dims " +
                      String::num(graph.input_shape.size()) +
                      ". Expected at least 2 dimensions.");
            return;
        }

        if (graph.input_shape[1] == 0) {
            ERR_PRINT(
                "InferenceEngine: Unable to support zero-sized input "
                "dimension.");
            return;
        }

        // Upload input buffer
        // If we have 300 float, that means that we have 100 pixels.
        int64_t first_axis = input.size() / graph.input_shape[1];
        std::vector<int64_t> dim = {first_axis, graph.input_shape[1]};

        task->activations_tm->get_or_create(graph.input_name, dim,
                                            input.to_byte_array());

        int compute_list = _rd->compute_list_begin();

        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            const ml::GraphNode& node = graph.nodes[i];

            _run_node(node, compute_list, weights_tm, task->activations_tm);
            // Add a barrier unless it's the very last node
            if (i < graph.nodes.size() - 1) {
                _rd->compute_list_add_barrier(compute_list);
            }
        }
        _rd->compute_list_end();
    }

    void MLInferenceEngine::_run_node(
        const ml::GraphNode& node,
        int64_t compute_list,
        Ref<ml::TensorResourceManager> weights_tm,
        Ref<ml::TensorResourceManager> activations_tm) {
        auto resolve = [&](const std::string& name) {
            RID rid = weights_tm->get_buffer_rid(name);
            if (rid.is_valid()) return rid;
            return activations_tm->get_or_create(name);
        };

        auto resolve_shape = [&](const std::string& name) {
            std::vector<int64_t> shape = weights_tm->get_tensor_shape(name);
            if (!shape.empty()) return shape;
            return activations_tm->get_tensor_shape(name);
        };

        switch (node.op) {
            case ml::NodeOperator::Gemm: {
                RID input_sb = resolve(node.inputs[0]);
                RID weight_sb = resolve(node.inputs[1]);
                RID bias_sb = resolve(node.inputs[2]);
                // Derive full shape from stored tensors
                // input: (M, K)
                // weights: (N, K) with transB = true
                auto in_shape = resolve_shape(node.inputs[0]);
                auto w_shape = resolve_shape(node.inputs[1]);

                // pixels
                uint32_t M = static_cast<uint32_t>(in_shape[0]);

                // output features
                uint32_t N = static_cast<uint32_t>(w_shape[0]);

                // input features
                uint32_t K = static_cast<uint32_t>(w_shape[1]);

                RID out =
                    activations_tm->get_or_create(node.outputs[0], {M, N});
                _dispatch_gemm(compute_list, input_sb, weight_sb, bias_sb, out,
                               M, N, K, node.alpha, node.beta);
            } break;

            case ml::NodeOperator::ReLU:
            case ml::NodeOperator::Sigmoid: {
                RID input_sb = resolve(node.inputs[0]);
                auto in_shape = resolve_shape(node.inputs[0]);

                uint32_t M = static_cast<uint32_t>(in_shape[0]);
                uint32_t K = static_cast<uint32_t>(in_shape[1]);

                RID out =
                    activations_tm->get_or_create(node.outputs[0], {M, K});
                _dispatch_elementwise(compute_list, _get_shader(node.op),
                                      _get_pipeline(node.op), input_sb, out, M,
                                      K);
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

        _deletion_queue.push([this, uniform_set]() {
            if (uniform_set.is_valid()) {
                _rd->free_rid(uniform_set);
            }
        });

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

        _deletion_queue.push([this, uniform_set]() {
            if (uniform_set.is_valid()) {
                _rd->free_rid(uniform_set);
            }
        });

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

    void MLInferenceEngine::_free_all_resources() {
        for (auto& it : _operator_pipeline) {
            if (it.second.is_valid()) {
                _rd->free_rid(it.second);
            }
        }

        _operator_pipeline.clear();
        for (auto& it : _operator_shader) {
            if (it.second.is_valid()) {
                _rd->free_rid(it.second);
            }
        }
        _operator_shader.clear();

        for (auto& [_, graph_context] : _graphs) {
            graph_context.weights_tm->clear();
        }

        _process_deletion_queue();
    }
    void MLInferenceEngine::_process_deletion_queue() {
        while (!_deletion_queue.empty()) {
            _deletion_queue.front()();
            _deletion_queue.pop();
        }
    }
}  // namespace godot