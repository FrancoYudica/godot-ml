#include "texture_input_handler_compute.hpp"

#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
using namespace godot;

struct PushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t channel_count;
    float padding[1]; // Alignment to 16 bytes
};

namespace ml {

bool TextureInputHandlerCompute::init(RenderingDevice* rd) {
    // Shader and pipeline creation
    const String path = "shaders/texture_sample.glsl";
    const String& shader_path = Utils::get_project_relative_path(path);
    _shader_rid = Utils::load_shader(rd, shader_path);

    ERR_FAIL_COND_V_MSG(
        !_shader_rid.is_valid(),
        false,
        "Failed to load shader.");

    _pipeline_rid = rd->compute_pipeline_create(_shader_rid);

    ERR_FAIL_COND_V_MSG(
        !_pipeline_rid.is_valid(),
        false,
        "Failed to create compute pipeline.");

    // Creates a linear sampler
    Ref<RDSamplerState> sampler_state;
    sampler_state.instantiate();
    sampler_state->set_mag_filter(RenderingDevice::SAMPLER_FILTER_LINEAR);
    sampler_state->set_min_filter(RenderingDevice::SAMPLER_FILTER_LINEAR);
    _sampler_rid = rd->sampler_create(sampler_state);

    ERR_FAIL_COND_V_MSG(
        !_sampler_rid.is_valid(),
        false,
        "Failed to create sampler.");

    // Uniform for texture sampler and storage buffer
    Ref<RDUniform> texture_uniform;
    texture_uniform.instantiate();
    texture_uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE);
    texture_uniform->set_binding(0);

    Ref<RDUniform> tensor_uniform;
    tensor_uniform.instantiate();
    tensor_uniform->set_uniform_type(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER);
    tensor_uniform->set_binding(1);

    _uniforms.push_back(texture_uniform);
    _uniforms.push_back(tensor_uniform);

    return true;
}

void TextureInputHandlerCompute::destroy(RenderingDevice* rd) {
    _uniforms.clear();
    rd->free_rid(_sampler_rid);
    rd->free_rid(_pipeline_rid);
    rd->free_rid(_shader_rid);
}

std::vector<int64_t> TextureInputHandlerCompute::get_shape(const std::unique_ptr<InputDesc::BaseData>& desc) const {
    InputDesc::Texture* texture_desc = dynamic_cast<InputDesc::Texture*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !texture_desc,
        std::vector<int64_t>(),
        "InferenceEngine: Failed to cast InputDesc to Texture.");

    int64_t width = texture_desc->texture->get_width();
    int64_t height = texture_desc->texture->get_height();

    if (texture_desc->process_width_override > 0)
        width = texture_desc->process_width_override;
    if (texture_desc->process_height_override > 0)
        height = texture_desc->process_height_override;

    return {1, static_cast<int64_t>(texture_desc->channels), height, width};
}

bool TextureInputHandlerCompute::upload(
    const std::unique_ptr<InputDesc::BaseData>& desc,
    const InputHandlerContext& ctx) {
    InputDesc::Texture* texture_desc = dynamic_cast<InputDesc::Texture*>(desc.get());

    ERR_FAIL_COND_V_MSG(
        !texture_desc,
        false,
        "InferenceEngine: Failed to cast InputDesc to Texture.");

    _tensor_name = texture_desc->tensor_name;

    RID tensor_rid = ctx.activations_tm->get_or_create(
        texture_desc->tensor_name,
        get_shape(desc));

    RID texture_rid =
        RenderingServer::get_singleton()->texture_get_rd_texture(texture_desc->texture->get_rid());

    _uniforms[0]->clear_ids();
    _uniforms[0]->add_id(_sampler_rid);
    _uniforms[0]->add_id(texture_rid);
    _uniforms[1]->clear_ids();
    _uniforms[1]->add_id(tensor_rid);

    return true;
}

void TextureInputHandlerCompute::dispatch(const InputHandlerContext& ctx) {
    const auto& shape = ctx.activations_tm->get_tensor_shape(_tensor_name);
    uint32_t width = static_cast<uint32_t>(shape[3]);
    uint32_t height = static_cast<uint32_t>(shape[2]);
    uint32_t channels = static_cast<uint32_t>(shape[1]);

    RID uniform_set_rid = ctx.rd->uniform_set_create(
        {_uniforms[0], _uniforms[1]},
        _shader_rid,
        0);

    PushConstants pc{width, height, channels};

    PackedByteArray pc_bytes;
    pc_bytes.resize(sizeof(PushConstants));
    memcpy(pc_bytes.ptrw(), &pc, sizeof(PushConstants));

    ctx.rd->compute_list_bind_compute_pipeline(ctx.compute_list, _pipeline_rid);
    ctx.rd->compute_list_bind_uniform_set(ctx.compute_list, uniform_set_rid, 0);
    ctx.rd->compute_list_set_push_constant(ctx.compute_list, pc_bytes, pc_bytes.size());

    uint32_t x_groups = (width + 7) / 8;
    uint32_t y_groups = (height + 7) / 8;
    ctx.rd->compute_list_dispatch(ctx.compute_list, x_groups, y_groups, 1);

    ctx.frame_deletion_stack->push([uniform_set_rid, rd = ctx.rd]() {
        if (rd->uniform_set_is_valid(uniform_set_rid)) rd->free_rid(uniform_set_rid);
    });
}

} // namespace ml