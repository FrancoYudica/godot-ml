#include "conv2d_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace ml {

bool Conv2DOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/conv2d.glsl";
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "Failed to load Conv2D shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ml::Conv2DOperator::dispatch(
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
    auto& b_shape = ctx.weights_tm->get_tensor_shape(node.inputs[2]);

    // Check for 4D (Batch, Channel, Height, Width)
    ERR_FAIL_COND_MSG(
        in_shape.size() != 4,
        "Conv2D requires 4D tensors (NCHW). Received input dimension: " + itos(in_shape.size()));

    ERR_FAIL_COND_MSG(
        w_shape.size() != 4,
        "Conv2D requires 4D tensors (NCHW). Received weight dimension: " + itos(w_shape.size()));

    auto& attrs = std::get<ConvAttributes>(node.attributes);

    uint32_t in_batch_size = in_shape[0];
    uint32_t in_channels = in_shape[1];
    uint32_t in_height = in_shape[2];
    uint32_t in_width = in_shape[3];

    // Calculate output shape for buffer allocation
    uint32_t out_h = (in_height + 2 * attrs.pads[0] - attrs.kernel_shape[0]) / attrs.strides[0] + 1;
    uint32_t out_w = (in_width + 2 * attrs.pads[1] - attrs.kernel_shape[1]) / attrs.strides[1] + 1;
    uint32_t out_channels = ctx.weights_tm->get_tensor_shape(node.inputs[1])[0]; // Takes out of bias

    std::vector<int64_t> out_shape = {
        static_cast<int64_t>(in_batch_size),
        static_cast<int64_t>(out_channels),
        static_cast<int64_t>(out_h),
        static_cast<int64_t>(out_w)};

    RID out_buf = ctx.activations_tm->get_or_create(node.outputs[0], out_shape);

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
    PushConstants pc{
        in_batch_size,
        in_width,
        in_height,
        in_channels,
        out_channels,
        static_cast<uint32_t>(attrs.kernel_shape[0]),
        static_cast<uint32_t>(attrs.pads[0]),
        static_cast<uint32_t>(attrs.strides[0]),
        static_cast<uint32_t>(attrs.strides[1]),
        out_w,
        out_h};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());
    ctx.rd->compute_list_dispatch(
        ctx.compute_list,
        (out_w + 7) / 8,
        (out_h + 7) / 8,
        in_batch_size);
}

void ml::Conv2DOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid()) {
        rd->free_rid(_pipeline);
    }
    if (_shader.is_valid()) {
        rd->free_rid(_shader);
    }
}

} // namespace ml