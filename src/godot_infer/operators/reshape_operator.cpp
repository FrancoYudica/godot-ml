#include "reshape_operator.hpp"

#include <godot_cpp/classes/rd_shader_file.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

namespace ml {

bool ReshapeOperator::init(godot::RenderingDevice* rd) {
    const String path = "shaders/matrix_transpose.glsl";
    const String& shader_path = ml::Utils::get_project_relative_path(path);
    _shader = ml::Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader.is_valid(),
        false,
        "ReshapeOperator: failed to load matrix_transpose shader.");

    _pipeline = rd->compute_pipeline_create(_shader);
    return _shader.is_valid() && _pipeline.is_valid();
}

void ReshapeOperator::dispatch(
    const Physical::Node& node,
    const OperatorContext& ctx) {

    const auto& attrs = std::get<Physical::ReshapeAttrs>(node.attributes);

    if (attrs.is_permutation) {
        _dispatch_transpose(node, ctx, attrs);
        return;
    }

    switch (attrs.mode) {

    case Physical::ReshapeMode::ImageToGemm: {
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "ImageToGemm: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "ImageToGemm: expected 1 output");

        const auto& in = ctx.shape_table->at(node.inputs[0]);
        ERR_FAIL_COND_MSG(in.size() != 4, "ImageToGemm: expects 4D input");

        int64_t b = in[0], c = in[1], h = in[2], w = in[3];

        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            {b * h * w, c});

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());

        ctx.frame_deletion_stack->push([name = node.outputs[0], tm = ctx.activations_tm]() {
            tm->remove_alias(name);
        });
        break;
    }

    case Physical::ReshapeMode::GemmToImage: {
        ERR_FAIL_COND_MSG(attrs.image_shape_ref.empty(), "GemmToImage: image_shape_ref is empty");
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "GemmToImage: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "GemmToImage: expected 1 output");

        const auto& target = ctx.shape_table->at(attrs.image_shape_ref);

        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            target);

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());
        break;
    }

    case Physical::ReshapeMode::Standard: {
        ERR_FAIL_COND_MSG(node.inputs.size() != 1, "Standard reshape: expected 1 input");
        ERR_FAIL_COND_MSG(node.outputs.size() != 1, "Standard reshape: expected 1 output");

        const auto& out_shape = ctx.shape_table->at(node.outputs[0]);
        auto result = ctx.activations_tm->create_alias(
            node.inputs[0],
            node.outputs[0],
            out_shape);

        ERR_FAIL_COND_MSG(!result.success, result.error.c_str());

        ctx.frame_deletion_stack->push([name = node.outputs[0], tm = ctx.activations_tm]() {
            tm->remove_alias(name);
        });
        break;
    }
    }
}

void ReshapeOperator::_dispatch_transpose(
    const Physical::Node& node,
    const OperatorContext& ctx,
    const Physical::ReshapeAttrs& attrs) {

    ERR_FAIL_COND_MSG(node.inputs.size() != 1, "Permute: expected 1 input");
    ERR_FAIL_COND_MSG(node.outputs.size() != 1, "Permute: expected 1 output");

    const auto& in_shape = ctx.shape_table->at(node.inputs[0]);

    uint32_t M, N;
    if (attrs.mode == Physical::ReshapeMode::ImageToGemm) {
        // BCHW input [B, C, H, W]: contiguous memory is [C, H*W] for B=1.
        // Transpose [C, H*W] -> [H*W, C] so GEMM receives [B*H*W, C].
        ERR_FAIL_COND_MSG(in_shape.size() != 4, "ImageToGemm permute: expected 4D input");
        M = static_cast<uint32_t>(in_shape[1]);
        N = static_cast<uint32_t>(in_shape[2] * in_shape[3]);
    } else {
        // GemmToImage: GEMM output [OH*OW, OC] -> transpose -> [OC, OH*OW] (BCHW).
        ERR_FAIL_COND_MSG(in_shape.size() != 2, "GemmToImage permute: expected 2D input");
        M = static_cast<uint32_t>(in_shape[0]);
        N = static_cast<uint32_t>(in_shape[1]);
    }

    auto resolve = [&](const std::string& name) -> RID {
        RID rid = ctx.activations_tm->get_buffer_rid(name);
        if (rid.is_valid()) return rid;
        return ctx.initializers_tm->get_buffer_rid(name);
    };

    RID src_buf = resolve(node.inputs[0]);
    RID dst_buf = ctx.activations_tm->get_buffer_rid(node.outputs[0]);

    auto make_uniform = [&](RID rid, int binding) {
        Ref<RDUniform> u;
        u.instantiate();
        u->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
        u->set_binding(binding);
        u->add_id(rid);
        return u;
    };

    TypedArray<RDUniform> uniforms;
    uniforms.push_back(make_uniform(src_buf, 0));
    uniforms.push_back(make_uniform(dst_buf, 1));

    RID uniform_set = ctx.rd->uniform_set_create(uniforms, _shader, 0);
    ctx.frame_deletion_stack->push([uniform_set, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set))
            rd->free_rid(uniform_set);
    });

    PushConstants pc{
        .rows = M,
        .cols = N};
    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(pc));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(pc));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t wg_x = (M + 7) / 8;
    uint32_t wg_y = (N + 7) / 8;
    ctx.rd->compute_list_dispatch(ctx.compute_list, wg_x, wg_y, 1);
}

void ReshapeOperator::destroy(godot::RenderingDevice* rd) {
    if (_pipeline.is_valid())
        rd->free_rid(_pipeline);
    if (_shader.is_valid())
        rd->free_rid(_shader);
}

} // namespace ml