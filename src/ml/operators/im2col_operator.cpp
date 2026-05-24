#include "im2col_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace ml {

bool Im2ColOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/im2col.glsl";
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load Im2Col shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ml::Im2ColOperator::dispatch(
    const ml::Physical::Node& node,
    const OperatorContext& ctx) {

    RID input_sb = ctx.activations_tm->get_buffer_rid(node.inputs[0]);
    RID out_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

    const auto& in_shape = ctx.shape_table->at(node.inputs[0]);
    const auto& out_shape = ctx.shape_table->at(node.outputs[0]);

    const auto& attrs = std::get<Physical::ConvAttrs>(node.attributes);

    uint32_t batch_size = in_shape[0];
    uint32_t channels = in_shape[1];
    uint32_t in_height = in_shape[2];
    uint32_t in_width = in_shape[3];

    // Compute out_h and out_w based on the input shape and convolution attributes
    int64_t kh_effective = (attrs.kernel_h - 1) * attrs.dilation_y + 1;
    int64_t kw_effective = (attrs.kernel_w - 1) * attrs.dilation_x + 1;
    int64_t pad_horizontal = attrs.padding_left + attrs.padding_right;
    int64_t pad_vertical = attrs.padding_top + attrs.padding_bottom;
    int64_t out_h = (in_height + pad_vertical - kh_effective) / attrs.stride_y + 1;
    int64_t out_w = (in_width + pad_horizontal - kw_effective) / attrs.stride_x + 1;

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
        channels,
        attrs.kernel_w,
        attrs.kernel_h,
        attrs.padding_left,
        attrs.padding_top,
        attrs.stride_x,
        attrs.stride_y,
        static_cast<uint32_t>(out_w),
        static_cast<uint32_t>(out_h)};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    UtilityFunctions::print("Attribute: out h = " + itos(out_h));
    UtilityFunctions::print("Attribute: out w = " + itos(out_w));
    UtilityFunctions::print("Attribute: channels = " + itos(channels));
    UtilityFunctions::print("Attribute: attrs.kernel_w = " + itos(attrs.kernel_w));
    UtilityFunctions::print("Attribute: attrs.kernel_h = " + itos(attrs.kernel_h));

    uint32_t total_workers = out_h * out_w * channels * attrs.kernel_w * attrs.kernel_h;

    UtilityFunctions::print("IM2COL: total workers = " + String::num(total_workers));

    uint32_t workgroup_count = (total_workers + 63) / 64;
    ctx.rd->compute_list_dispatch(ctx.compute_list, workgroup_count, 1, 1);
}

void ml::Im2ColOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace ml