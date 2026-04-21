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

    auto resolve = [&](const std::string& name) -> RID {
        RID rid = ctx.activations_tm->get_buffer_rid(name);
        if (rid.is_valid()) return rid;
        return ctx.weights_tm->get_buffer_rid(name);
    };

    auto resolve_shape = [&](const std::string& name) -> std::vector<int64_t> {
        auto shape = ctx.activations_tm->get_tensor_shape(name);
        if (!shape.empty()) return shape;
        return ctx.weights_tm->get_tensor_shape(name);
    };

    ERR_FAIL_COND_MSG(
        node.outputs.size() != 1,
        "Col2Im: expected 1 output, got: " + itos(node.outputs.size()));

    ERR_FAIL_COND_MSG(
        node.inputs.size() != 2,
        "Col2Im: expected 2 inputs, got: " + itos(node.inputs.size()));

    RID input_sb = resolve(node.inputs[0]);
    auto& in_shape = ctx.activations_tm->get_tensor_shape(node.inputs[0]);

    ERR_FAIL_COND_MSG(
        in_shape.size() != 2,
        "Col2Im requires 2D input. Got: " + itos(in_shape.size()));

    auto& attrs = std::get<Col2ImAttributes>(node.attributes);

    auto source_activation_shape = resolve_shape(attrs.source_activation);

    ERR_FAIL_COND_MSG(source_activation_shape.empty(), "Col2Im: Unable to find source activation shape.");
    ERR_FAIL_COND_MSG(
        source_activation_shape.size() != 4,
        "Col2Im: Invalid source activation shape: " +
            Utils::get_iterator_str(
                source_activation_shape.begin(),
                source_activation_shape.end()));

    uint32_t in_batch_size = source_activation_shape[0];
    uint32_t in_channels = source_activation_shape[1];
    uint32_t in_height = source_activation_shape[2];
    uint32_t in_width = source_activation_shape[3];
    uint32_t kH = attrs.kernel_shape[0];
    uint32_t kW = attrs.kernel_shape[1];

    // ConvTranspose weights: [in_C, out_C, kH, kW]
    uint32_t out_channels = ctx.weights_tm->get_tensor_shape(node.inputs[1])[0];

    // ConvTranspose output size formula
    uint32_t out_h = (in_height - 1) * attrs.strides[0] - 2 * attrs.pads[0] + kH;
    uint32_t out_w = (in_width - 1) * attrs.strides[1] - 2 * attrs.pads[1] + kW;

    // Output is [H*W, out_C] — Reshape node restores spatial dims
    std::vector<int64_t> out_shape = {
        static_cast<int64_t>(out_h * out_w),
        static_cast<int64_t>(out_channels)};

    if (node.reshape_info != nullptr) {
        node.reshape_info->shape = {
            static_cast<int64_t>(in_batch_size),
            static_cast<int64_t>(out_channels),
            static_cast<int64_t>(out_h),
            static_cast<int64_t>(out_w)};
    }

    RID out_buf = ctx.activations_tm->get_or_create(node.outputs[0], out_shape);

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