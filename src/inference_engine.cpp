#include <unordered_set>
#include "inference_engine.h"
#include "ml_inference/ml_parser.hpp"
#include "ml_inference/ml_utils.hpp"
#include "ml_inference/input_handlers/ml_float_array_input_handler.hpp"
#include "ml_inference/output_handlers/ml_float_array_output_handler.hpp"
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

        ClassDB::bind_method(D_METHOD("run_async", "model_rid"),
                             &MLInferenceEngine::run_async);

        ClassDB::bind_method(D_METHOD("print_model", "model_rid"),
                             &MLInferenceEngine::print_model);

        ClassDB::bind_method(D_METHOD("unload_model", "model_rid"),
                             &MLInferenceEngine::unload_model);

        ClassDB::bind_method(D_METHOD("pop_task_output", "task", "output_name"),
                             &MLInferenceEngine::pop_task_output);

        ClassDB::bind_method(D_METHOD("add_float_array_input", "task",
                                      "tensor_name", "data", "shape"),
                             &MLInferenceEngine::add_float_array_input);

        ClassDB::bind_method(
            D_METHOD("add_float_array_output", "task", "tensor_name"),
            &MLInferenceEngine::add_float_array_output);

        ClassDB::bind_method(D_METHOD("_process_pending_tasks"),
                             &MLInferenceEngine::_process_pending_tasks);
    }

    void MLInferenceEngine::init() {
        if (_initialized) {
            UtilityFunctions::print("InferenceEngine: already initialized.");
            return;
        }

        _rd = RenderingServer::get_singleton()->get_rendering_device();

        ERR_FAIL_COND_EDMSG(!_rd,
                            "InferenceEngine: Could not get RenderingDevice.");

        ERR_FAIL_COND_EDMSG(
            !_operator_registry.init(_rd),
            "InferenceEngine: failed to initialize operator registry.");

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

    Ref<InferenceTask> MLInferenceEngine::run_async(uint32_t model_rid) {
        auto it = _graphs.find(model_rid);

        ERR_FAIL_COND_V_MSG(
            it == _graphs.end(), nullptr,
            "InferenceEngine: model " + String::num(model_rid) + " not found.");

        Ref<InferenceTask> task;
        task.instantiate();
        task->init(model_rid, _rd);
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

    PackedFloat32Array MLInferenceEngine::pop_task_output(
        Ref<InferenceTask> task, const String& output_name) {
        ERR_FAIL_COND_V_MSG(!task->is_done, PackedFloat32Array(),
                            "InferenceTask: Task not yet completed by GPU.");

        ERR_FAIL_COND_V_MSG(task->freed, PackedFloat32Array(),
                            "InferenceEngine: Task already resources freed. "
                            "Unable to retrieve output");

        ERR_FAIL_COND_V_MSG(
            task->output_handlers.find(output_name.utf8().get_data()) ==
                task->output_handlers.end(),
            PackedFloat32Array(),
            "InferenceEngine: Output handler for tensor " + output_name +
                " not found.");

        auto& output_handler =
            task->output_handlers[output_name.utf8().get_data()];
        godot::Variant output = output_handler->get();

        // RID sb =
        // task->activations_tm->get_buffer_rid(output_name.utf8().ptr());

        // ERR_FAIL_COND_V_MSG(!sb.is_valid(), PackedFloat32Array(),
        //                     "InferenceEngine: Buffer not found.");

        // PackedByteArray byte_array = _rd->buffer_get_data(sb);
        // PackedFloat32Array float_array;
        // float_array.resize(byte_array.size() / sizeof(float));
        // memcpy(float_array.ptrw(), byte_array.ptrw(), byte_array.size());

        // Frees all the resources
        task->activations_tm->clear();
        task->freed = true;
        return output;
    }
    void MLInferenceEngine::add_float_array_input(
        Ref<InferenceTask> task,
        const String& tensor_name,
        const PackedFloat32Array& data,
        const PackedFloat64Array& shape) {
        std::string input_name = tensor_name.utf8().get_data();

        ERR_FAIL_COND_MSG(
            task->input_handlers.find(input_name) != task->input_handlers.end(),
            "InferenceEngine: Input handler for tensor " + tensor_name +
                " already exists.");

        std::vector<int64_t> s(shape.ptr(), shape.ptr() + shape.size());

        task->input_handlers[input_name] =
            std::make_unique<ml::FloatArrayInputHandler>(
                ml::InputDesc::FloatArray{data, s});
    }
    void MLInferenceEngine::add_float_array_output(Ref<InferenceTask> task,
                                                   const String& tensor_name) {
        std::string output_name = tensor_name.utf8().get_data();

        ERR_FAIL_COND_MSG(task->input_handlers.find(output_name) !=
                              task->input_handlers.end(),
                          "InferenceEngine: Output handler for tensor " +
                              tensor_name + " already exists.");

        task->output_handlers[output_name] =
            std::make_unique<ml::FloatArrayOutputHandler>(
                ml::OutputDesc::FloatArray{tensor_name.utf8().get_data()});
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

        ERR_FAIL_COND_MSG(it == _graphs.end(),
                          "InferenceEngine: graph not found.");

        const ml::Graph& graph = it->second.graph;
        Ref<ml::TensorResourceManager> weights_tm = it->second.weights_tm;

        // Loads inputs
        for (auto& [tensor_name, input_handler] : task->input_handlers) {
            input_handler->upload(tensor_name, _rd, task->activations_tm);
        }

        int compute_list = _rd->compute_list_begin();

        // Processes the graph
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            const ml::GraphNode& node = graph.nodes[i];

            _run_node(node, compute_list, weights_tm, task->activations_tm);
            // Add a barrier unless it's the very last node
            if (i < graph.nodes.size() - 1) {
                _rd->compute_list_add_barrier(compute_list);
            }
        }

        _rd->compute_list_end();

        // Downloads the outputs
        for (auto& [tensor_name, output_handler] : task->output_handlers) {
            output_handler->download(_rd, task->activations_tm);
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
                                .compute_list = compute_list};

        op->dispatch(node, ctx);
    }

    void MLInferenceEngine::_free_all_resources() {
        for (auto& [_, graph_context] : _graphs) {
            graph_context.weights_tm->clear();
        }
        _operator_registry.destroy(_rd);
        _process_deletion_queue();
    }
    void MLInferenceEngine::_process_deletion_queue() {
        while (!_deletion_queue.empty()) {
            _deletion_queue.front()();
            _deletion_queue.pop();
        }
    }
    bool MLInferenceEngine::_has_graph(uint32_t graph_rid) {
        return _graphs.find(graph_rid) != _graphs.end();
    }
}  // namespace godot