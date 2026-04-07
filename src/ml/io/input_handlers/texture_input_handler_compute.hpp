#pragma once
#include "io/input_handler.hpp"
#include "io/io_descriptor.hpp"

namespace ml {
    class TextureInputHandlerCompute : public IInputHandler {
    public:
        bool init(godot::RenderingDevice* rd) override;

        void destroy(godot::RenderingDevice* rd) override;

        std::vector<int64_t> upload(
            const std::unique_ptr<InputDesc::BaseData>& desc,
            const InputHandlerContext& ctx) override;

        void dispatch(const InputHandlerContext& ctx) override;

    private:
        godot::RID _sampler_rid;
        godot::RID _pipeline_rid;
        godot::RID _shader_rid;
        std::vector<godot::Ref<godot::RDUniform>> _uniforms;
        uint32_t _texture_width, _texture_height, _texture_channels;
    };

}  // namespace ml