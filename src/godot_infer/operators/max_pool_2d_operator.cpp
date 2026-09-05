#include "max_pool_2d_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace ml {

bool MaxPool2DOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/max_pool_2d.glsl";
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load Conv2D shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ml::MaxPool2DOperator::dispatch(
    const ml::Physical::Node& node,
    const OperatorContext& ctx) {

    // Resolve buffers
    auto resolve = [&](const std::string& name) -> RID {
        RID rid = ctx.activations_tm->get_buffer_rid(name);
        if (rid.is_valid()) return rid;
        return ctx.initializers_tm->get_buffer_rid(name);
    };

    RID input_sb = resolve(node.inputs[0]);

    const auto& in_shape = ctx.shape_table->at(node.inputs[0]);
    const auto& out_shape = ctx.shape_table->at(node.outputs[0]);

    ERR_FAIL_COND_MSG(in_shape.size() != 4, "MaxPool2D: input must be 4D (NCHW)");

    const auto& attrs = std::get<Physical::MaxPool2DAttrs>(node.attributes);

    uint32_t batch_size = in_shape[0];
    uint32_t channels = in_shape[1];
    uint32_t in_height = in_shape[2];
    uint32_t in_width = in_shape[3];
    uint32_t out_h = out_shape[2];
    uint32_t out_w = out_shape[3];

    RID out_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

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
    uniforms.push_back(make_uniform(out_buf, 1));

    RID uniform_set_rid = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push([uniform_set_rid, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid))
            rd->free_rid(uniform_set_rid);
    });

    // Push constants
    PushConstants pc{
        .batch_size = batch_size,
        .channels = channels,
        .in_height = in_height,
        .in_width = in_width,
        .out_height = out_h,
        .out_width = out_w,
        .kernel_h = attrs.kernel_h,
        .kernel_w = attrs.kernel_w,
        .padding_top = attrs.padding_top,
        .padding_left = attrs.padding_left,
        .stride_y = attrs.stride_y,
        .stride_x = attrs.stride_x,
        .dilation_y = attrs.dilation_y,
        .dilation_x = attrs.dilation_x};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t workgroup_count_z = batch_size * channels;

    ctx.rd->compute_list_dispatch(
        ctx.compute_list,
        (out_w + 7) / 8,
        (out_h + 7) / 8,
        workgroup_count_z);
}

void ml::MaxPool2DOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace ml