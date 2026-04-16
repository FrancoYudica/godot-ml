#include "gemm_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace ml {

bool GemmOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/gemm.glsl";
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load GEMM shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ml::GemmOperator::dispatch(
    const ml::PhysicalNode& node,
    const OperatorContext& ctx) {
    // Resolve buffers
    auto resolve = [&](const std::string& name) -> RID {
        RID rid = ctx.activations_tm->get_buffer_rid(name);
        if (rid.is_valid()) return rid;
        return ctx.weights_tm->get_buffer_rid(name);
    };

    RID input_sb = resolve(node.inputs[0]);
    RID weight_sb = resolve(node.inputs[1]);
    RID bias_sb = resolve(node.inputs[2]);

    auto& in_shape = ctx.activations_tm->get_tensor_shape(node.inputs[0]);
    auto& w_shape = ctx.weights_tm->get_tensor_shape(node.inputs[1]);

    uint32_t K, M;

    if (in_shape.size() == 2) {
        M = in_shape[0]; // pixels
        K = in_shape[1]; // channels
    }

    // Temporal fix for images.
    else if (in_shape.size() == 4) {
        K = in_shape[1];               // channels
        M = in_shape[2] * in_shape[3]; // pixels
    }
    // N is the number of output features (rows of the weight matrix)
    uint32_t N = static_cast<uint32_t>(w_shape[0]);

    RID out_buf = ctx.activations_tm->get_or_create(
        node.outputs[0],
        {(int64_t)M, (int64_t)N});

    // Uniforms
    auto make_uniform = [&](RID rid, int binding) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(rid);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(input_sb, 0));
    uniforms.push_back(make_uniform(weight_sb, 1));
    uniforms.push_back(make_uniform(bias_sb, 2));
    uniforms.push_back(make_uniform(out_buf, 3));

    RID uniform_set_rid = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push([uniform_set_rid, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid))
            rd->free_rid(uniform_set_rid);
    });

    // Push constants
    auto& gemm = std::get<GemmAttributes>(node.attributes);
    PushConstants pc{M, N, K, gemm.alpha, gemm.beta};
    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t workgroup_count_x = M / 7 + 8;
    uint32_t workgroup_count_y = N / 7 + 8;

    ctx.rd->compute_list_dispatch(
        ctx.compute_list,
        workgroup_count_x,
        workgroup_count_y,
        1);
}

void ml::GemmOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace ml