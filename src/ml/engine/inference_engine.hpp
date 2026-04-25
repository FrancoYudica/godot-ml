#ifndef ML_INFERENCE_ENGINE_H
#define ML_INFERENCE_ENGINE_H
#include "core/core.hpp"
#include "inference_descriptor.hpp"
#include "inference_task.hpp"
#include "io/io.hpp"
#include "operators/operators.hpp"
#include "parser/parser.hpp"
#include "tensors/tensors.hpp"

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

namespace godot {

struct GraphContext {
    ml::PhysicalGraph graph;
    Ref<ml::TensorResourceManager> weights_tm;
};

class MLInferenceEngine : public RefCounted {
    GDCLASS(MLInferenceEngine, RefCounted)

  public:
    void init();
    void destroy();
    uint32_t register_model(String model_path);
    void unload_model(uint32_t model_rid);
    Ref<InferenceTask> queue_request(
        uint32_t model_rid,
        Ref<InferenceDescriptor> request);
    void print_model(uint32_t model_rid);

    void destroy_task(Ref<InferenceTask> task);
    godot::Variant get_task_output(
        Ref<InferenceTask> task,
        const String& output_name);

  protected:
    static void _bind_methods();

  private:
    void _process_pending_tasks();
    void _process_task(Ref<InferenceTask> task);
    void _run_node(
        const ml::PhysicalNode& node,
        int64_t compute_list,
        Ref<ml::TensorResourceManager> weights_tm,
        Ref<ml::TensorResourceManager> activations_tm,
        const ml::ShapeTable& shape_table);
    void _allocate_activations(
        const ml::PhysicalGraph& graph,
        const ml::ShapeTable& shape_table,
        Ref<ml::TensorResourceManager> activations_tm);

    void _free_all_resources();
    bool _has_graph(uint32_t graph_rid);

  private:
    RenderingDevice* _rd;
    ml::StorageBufferPool _sb_pool;
    ml::OperatorRegistry _operator_registry;
    ml::InputHandlerRegistry _input_registry;
    ml::OutputHandlerRegistry _output_registry;
    std::unordered_map<uint32_t, GraphContext> _graphs;
    std::vector<Ref<InferenceTask>> _pending_tasks;
    std::vector<Ref<InferenceTask>> _executing_tasks;
    bool _initialized = false;
    bool _destroying = false;

    ml::DeletionStack _frame_deletion_stack;

    uint32_t _next_graph_id = 1;
};
} // namespace godot

#endif
