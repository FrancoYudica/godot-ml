#include <unordered_set>
#include "inference_engine.h"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace godot {

    void MLInferenceEngine::_bind_methods() {
        ClassDB::bind_method(D_METHOD("init"), &MLInferenceEngine::init);
        ClassDB::bind_method(D_METHOD("destroy"), &MLInferenceEngine::destroy);

        ClassDB::bind_method(D_METHOD("register_model", "model_path"),
                             &MLInferenceEngine::register_model);

        ClassDB::bind_method(D_METHOD("queue_request", "model_rid", "request"),
                             &MLInferenceEngine::queue_request);

        ClassDB::bind_method(D_METHOD("print_model", "model_rid"),
                             &MLInferenceEngine::print_model);

        ClassDB::bind_method(D_METHOD("unload_model", "model_rid"),
                             &MLInferenceEngine::unload_model);

        ClassDB::bind_method(D_METHOD("destroy_task", "task"),
                             &MLInferenceEngine::destroy_task);

        ClassDB::bind_method(D_METHOD("get_task_output", "task", "output_name"),
                             &MLInferenceEngine::get_task_output);

        ClassDB::bind_method(D_METHOD("_process_pending_tasks"),
                             &MLInferenceEngine::_process_pending_tasks);
    }

    void MLInferenceEngine::init() {
        if (_initialized) {
            UtilityFunctions::print("InferenceEngine: already initialized.");
            return;
        }

        _rd = RenderingServer::get_singleton()->get_rendering_device();

        ERR_FAIL_COND_MSG(!_rd,
                          "InferenceEngine: Could not get RenderingDevice.");

        ERR_FAIL_COND_MSG(
            !_operator_registry.init(_rd),
            "InferenceEngine: failed to initialize operator registry.");

        ERR_FAIL_COND_MSG(
            !_input_registry.init(_rd),
            "InferenceEngine: failed to initialize input registry.");

        ERR_FAIL_COND_MSG(
            !_output_registry.init(_rd),
            "InferenceEngine: failed to initialize output registry.");

        RenderingServer::get_singleton()->connect(
            "frame_pre_draw", Callable(this, "_process_pending_tasks"));

        _initialized = true;
    }

    uint32_t MLInferenceEngine::register_model(String model_path) {
        ERR_FAIL_COND_V_MSG(!_initialized, 0,
                            "InferenceEngine: not initialized.");

        print_line("Loading ML inference engine with model: " + model_path);

        GraphContext graph_context;

        bool success =
            ml::Parser::parse(model_path.utf8().ptr(), graph_context.graph);

        ERR_FAIL_COND_V_MSG(!success, 0,
                            "InferenceEngine: failed to parse model.");

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
        ERR_FAIL_COND_MSG(
            !_has_graph(model_rid),
            "InferenceEngine: model " + String::num(model_rid) + " not found.");

        _graphs.erase(model_rid);
    }

    void MLInferenceEngine::destroy() {
        _destroying = true;
    }

    Ref<InferenceTask> MLInferenceEngine::queue_request(
        uint32_t model_rid, Ref<InferenceDescriptor> request) {
        auto it = _graphs.find(model_rid);

        ERR_FAIL_COND_V_MSG(
            it == _graphs.end(), nullptr,
            "InferenceEngine: model " + String::num(model_rid) + " not found.");

        ERR_FAIL_COND_V_MSG(
            !request->has_inputs_defined(it->second.graph.input_names), nullptr,
            "InferenceEngine: one or more required inputs are missing.");

        Ref<InferenceTask> task;
        task.instantiate();
        task->init(model_rid, _rd, request);
        _pending_tasks.push_back(task);
        return task;
    }

    void MLInferenceEngine::print_model(uint32_t model_rid) {
        ERR_FAIL_COND_MSG(
            !_has_graph(model_rid),
            "InferenceEngine: model " + String::num(model_rid) + " not found.");

        auto it = _graphs.find(model_rid);
        ml::Utils::print(it->second.graph);
    }

    void MLInferenceEngine::destroy_task(Ref<InferenceTask> task) {
        ERR_FAIL_COND_MSG(!task->is_completed(),
                          "InferenceTask: Task not yet completed by GPU. Too "
                          "early to destroy");

        ERR_FAIL_COND_MSG(task->is_freed(),
                          "InferenceEngine: Task already cleared");
        // Frees all the resources
        task->destroy(_rd);
    }

    godot::Variant MLInferenceEngine::get_task_output(
        Ref<InferenceTask> task, const String& output_name) {
        ERR_FAIL_COND_V_MSG(!task->is_completed(), PackedFloat32Array(),
                            "InferenceTask: Task not yet completed by GPU.");

        ERR_FAIL_COND_V_MSG(task->is_freed(), PackedFloat32Array(),
                            "InferenceEngine: Task already resources freed. "
                            "Unable to retrieve output");

        ERR_FAIL_COND_V_MSG(task->outputs.find(output_name.utf8().get_data()) ==
                                task->outputs.end(),
                            PackedFloat32Array(),
                            "InferenceEngine: Output result \"" + output_name +
                                "\" not found.");

        godot::Variant output = task->outputs[output_name.utf8().get_data()];
        return output;
    }

    void MLInferenceEngine::_process_pending_tasks() {
        if (_destroying && _pending_tasks.empty() && _executing_tasks.empty()) {
            _free_all_resources();
            RenderingServer::get_singleton()->disconnect(
                "frame_pre_draw", Callable(this, "_process_pending_tasks"));

            _destroying = false;
            return;
        }

        _frame_deletion_stack.process();

        // Tasks that were executing now are finished
        for (const auto& task : _executing_tasks) {
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

        ERR_FAIL_COND_MSG(it == _graphs.end(),
                          "InferenceEngine: graph not found.");

        const ml::Graph& graph = it->second.graph;
        Ref<ml::TensorResourceManager> weights_tm = it->second.weights_tm;

        // Loads inputs
        ml::InputHandlerContext ctx = {
            .rd = _rd,
            .activations_tm = task->activations_tm,
            .compute_list = 0,
            .frame_deletion_stack = &_frame_deletion_stack};

        // Uploads all the input handlers data
        for (auto& [tensor_name, descriptor] : task->descriptor->inputs) {
            auto& handler = _input_registry.get(descriptor->type);
            handler->upload(descriptor, ctx);
        }

        // Creates compute list
        int compute_list = _rd->compute_list_begin();

        // Dispatches the input handlers parallelly, making sure that these are
        // the first one on the compute list
        ctx.compute_list = compute_list;
        for (auto& [_, descriptor] : task->descriptor->inputs) {
            auto& handler = _input_registry.get(descriptor->type);
            handler->dispatch(ctx);
        }

        // Make sure that all the inputs where loaded
        _rd->compute_list_add_barrier(compute_list);

        // Processes the graph
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            const ml::GraphNode& node = graph.nodes[i];

            _run_node(node, compute_list, weights_tm, task->activations_tm);
            // Add a barrier (also for the last node for output dispatch)
            _rd->compute_list_add_barrier(compute_list);
        }

        // Dispatch outputs parallelly
        ml::OutputHandlerContext out_ctx = {
            .rd = _rd,
            .activations_tm = task->activations_tm,
            .compute_list = compute_list,
            .frame_deletion_stack = &_frame_deletion_stack};

        for (auto& [_, descriptor] : task->descriptor->outputs) {
            auto& handler = _output_registry.get(descriptor->type);
            handler->dispatch(out_ctx);
        }

        _rd->compute_list_end();

        // Downloads the outputs
        for (auto& [output_name, descriptor] : task->descriptor->outputs) {
            auto& handler = _output_registry.get(descriptor->type);
            auto result =
                handler->download(descriptor, _rd, task->activations_tm);
            task->outputs[output_name] = result;
        }
    }

    void MLInferenceEngine::_run_node(
        const ml::GraphNode& node,
        int64_t compute_list,
        Ref<ml::TensorResourceManager> weights_tm,
        Ref<ml::TensorResourceManager> activations_tm) {
        auto op = _operator_registry.get(node.op);

        ERR_FAIL_COND_MSG(
            op == nullptr,
            "MLInferenceEngine: unsupported operator, skipping node.");

        ml::OperatorContext ctx{.rd = _rd,
                                .weights_tm = weights_tm,
                                .activations_tm = activations_tm,
                                .compute_list = compute_list,
                                .frame_deletion_stack = &_frame_deletion_stack};

        op->dispatch(node, ctx);
    }

    void MLInferenceEngine::_free_all_resources() {
        for (auto& [_, graph_context] : _graphs) {
            graph_context.weights_tm->destroy();
        }
        _operator_registry.destroy(_rd);
        _input_registry.destroy(_rd);
        _output_registry.destroy(_rd);
        _frame_deletion_stack.process();
    }

    bool MLInferenceEngine::_has_graph(uint32_t graph_rid) {
        return _graphs.find(graph_rid) != _graphs.end();
    }
}  // namespace godot