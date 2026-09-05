#pragma once
#include "io/output_handler.hpp"

namespace ml {

class TextureOutputHandlerCompute : public IOutputHandler {
  public:
    bool init(godot::RenderingDevice* rd) override;
    void destroy(RenderingDevice* rd) override;
    void dispatch(
        const std::unique_ptr<OutputDesc::BaseData>& desc,
        const OutputHandlerContext& ctx) override;
    godot::Variant download(
        const std::unique_ptr<OutputDesc::BaseData>& desc,
        godot::RenderingDevice* rd,
        Ref<TensorResourceManager> activations_tm) override;

  private:
    godot::RID _shader_rid;
    godot::RID _pipeline_rid;
    std::vector<godot::Ref<godot::RDUniform>> _uniforms;
    uint32_t _texture_width, _texture_height, _texture_channels;
};

} // namespace ml