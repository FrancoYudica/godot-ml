#include "element_wise_operator.hpp"

namespace ml {

bool ElementwiseOperator::init(godot::RenderingDevice* rd) {
    const String path = shader_path();
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Unable to load shader: " + shader_path);

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ElementwiseOperator::dispatch(
    const GraphNode& node,
    const OperatorContext& ctx) {

    auto resolve = [&](const std::string& name) {
        RID rid = ctx.weights_tm->get_buffer_rid(name);
        if (rid.is_valid()) return rid;
        return ctx.activations_tm->get_or_create(name);
    };

    auto resolve_shape = [&](const std::string& name) {
        std::vector<int64_t> shape = ctx.weights_tm->get_tensor_shape(name);
        if (!shape.empty()) return shape;
        return ctx.activations_tm->get_tensor_shape(name);
    };

    RID input = resolve(node.inputs[0]);
    auto in_shape = resolve_shape(node.inputs[0]);

    RID output = ctx.activations_tm->get_or_create(
        node.outputs[0],
        in_shape // Keep input shape
    );

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

    RID uniform_set_rid = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push(
        [uniform_set_rid, rd = ctx.rd]() {
            if (rd->uniform_set_is_valid(uniform_set_rid))
                rd->free_rid(uniform_set_rid);
        });

    uint32_t total_floats = 1;
    for (const auto& dim : in_shape) {
        total_floats *= static_cast<uint32_t>(dim);
    }

    PushConstants pc{total_floats};
    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());
    uint32_t local_size_x = 64;
    uint32_t total_workgroups_x = (total_floats + local_size_x - 1) / local_size_x;
    ctx.rd->compute_list_dispatch(ctx.compute_list, total_workgroups_x, 1, 1);
}

void ElementwiseOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace ml