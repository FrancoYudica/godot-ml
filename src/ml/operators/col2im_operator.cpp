#include "col2im_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace ml {

bool Col2ImOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/col2im.glsl";
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load Col2Im shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ml::Col2ImOperator::dispatch(
    const ml::PhysicalNode& node,
    const OperatorContext& ctx) {

    ERR_FAIL_COND_MSG(node.outputs.size() != 1, "Col2Im: expected 1 output");
    ERR_FAIL_COND_MSG(node.inputs.size() != 2, "Col2Im: expected 2 inputs");

    RID input_sb = ctx.activations_tm->get_buffer_rid(node.inputs[0]);
    RID out_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

    // All shapes come from the pre-computed ShapeTable.
    // meta4d = [b, out_c, out_h, out_w] written by shape inference.
    const auto& attrs = std::get<Col2ImAttributes>(node.attributes);
    const auto& src = ctx.shape_table->at(attrs.source_activation);     // [b, ic, ih, iw]
    const auto& meta4d = ctx.shape_table->at(node.outputs[0] + "__4d"); // [b, oc, oh, ow]

    uint32_t in_width = src[3];
    uint32_t in_height = src[2];
    uint32_t out_w = meta4d[3];
    uint32_t out_h = meta4d[2];
    uint32_t out_channels = meta4d[1];
    uint32_t kH = attrs.kernel_shape[0];

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

    PushConstants pc{
        in_width,
        in_height,
        out_w,
        out_h,
        out_channels,
        kH,
        static_cast<uint32_t>(attrs.pads[0]),
        static_cast<uint32_t>(attrs.strides[0])};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t groups_x = (out_w + 7) / 8;
    uint32_t groups_y = (out_h + 7) / 8;
    uint32_t groups_z = out_channels;
    ctx.rd->compute_list_dispatch(ctx.compute_list, groups_x, groups_y, groups_z);
}

void ml::Col2ImOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace ml